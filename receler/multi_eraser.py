import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import BasicTransformerBlock

from receler.erasers.diffusers_erasers import diffuser_prefix_name
from receler.erasers.utils import AdapterEraser


def parse_csv_list(value, item_type=str):
    if value is None:
        return None
    items = [item.strip() for item in value.split(',')]
    items = [item for item in items if item]
    return [item_type(item) for item in items]


def load_fusion_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    erasers = config.get('erasers')
    if not erasers:
        raise ValueError(f'fusion config must contain a non-empty "erasers" list: {config_path}')
    eraser_paths = []
    weights = []
    for idx, eraser in enumerate(erasers):
        if 'path' not in eraser:
            raise ValueError(f'fusion config eraser #{idx} is missing "path"')
        eraser_paths.append(eraser['path'])
        weights.append(float(eraser.get('weight', 1.0)))
    fusion_scale = float(config.get('fusion_scale', 1.0))
    return eraser_paths, weights, fusion_scale


def normalize_fusion_inputs(eraser_paths=None, fusion_weights=None, fusion_config=None, fusion_scale=None):
    if fusion_config and eraser_paths:
        raise ValueError('--fusion_config and --eraser_paths are mutually exclusive.')
    if fusion_config:
        eraser_paths, fusion_weights, config_fusion_scale = load_fusion_config(fusion_config)
        if fusion_scale is None:
            fusion_scale = config_fusion_scale
    elif eraser_paths:
        eraser_paths = parse_csv_list(eraser_paths, str)
        fusion_weights = parse_csv_list(fusion_weights, float) if fusion_weights else None
    else:
        return None, None

    if not eraser_paths:
        raise ValueError('At least one eraser path is required for multi-Eraser fusion.')
    if fusion_weights is None:
        fusion_weights = [1.0] * len(eraser_paths)
    if len(fusion_weights) != len(eraser_paths):
        raise ValueError(
            f'Expected {len(eraser_paths)} fusion weights, but got {len(fusion_weights)}.'
        )
    if fusion_scale is None:
        fusion_scale = 1.0
    fusion_scale = float(fusion_scale)
    if fusion_scale < 0:
        raise ValueError('fusion_scale must be non-negative.')
    return eraser_paths, fusion_weights, fusion_scale


class MultiEraserWrapper(nn.Module):
    def __init__(
            self,
            eraser_paths,
            fusion_weights=None,
            fusion_scale=1.0,
            trainable_weights=False,
            device=None,
            dtype=None,
        ):
        super().__init__()
        if not eraser_paths:
            raise ValueError('MultiEraserWrapper requires at least one eraser path.')
        self.eraser_paths = [str(path) for path in eraser_paths]
        self.fusion_weights = self._prepare_weights(fusion_weights, len(self.eraser_paths))
        self.fusion_scale = float(fusion_scale)
        if self.fusion_scale < 0:
            raise ValueError('fusion_scale must be non-negative.')
        self.trainable_weights = trainable_weights
        self.device = device
        self.dtype = dtype
        self.enabled = True
        self.handles = []
        self.adapters = nn.ModuleDict()
        self.layer_to_adapter_keys = {}
        self.layer_to_eraser_indices = {}
        self.eraser_configs = []
        self.eraser_checkpoints = []
        self.layer_names = set()

        logits = torch.log(torch.tensor(self.fusion_weights, dtype=torch.float32).clamp_min(1e-12))
        self.logits = nn.Parameter(logits, requires_grad=trainable_weights)
        self._load_checkpoints()

    @staticmethod
    def _prepare_weights(fusion_weights, num_erasers):
        if fusion_weights is None:
            return [1.0] * num_erasers
        if len(fusion_weights) != num_erasers:
            raise ValueError(f'Expected {num_erasers} fusion weights, but got {len(fusion_weights)}.')
        weights = [float(weight) for weight in fusion_weights]
        if any(weight < 0 for weight in weights):
            raise ValueError('Fusion weights must be non-negative.')
        if sum(weights) <= 0:
            raise ValueError('At least one fusion weight must be greater than zero.')
        return weights

    @staticmethod
    def _adapter_key(eraser_idx, layer_name):
        return f'{eraser_idx}__{layer_name.replace(".", "_")}'

    @staticmethod
    def _load_eraser_folder(eraser_path):
        weights_path = os.path.join(eraser_path, 'eraser_weights.pt')
        config_path = os.path.join(eraser_path, 'eraser_config.json')
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f'Missing eraser_weights.pt: {weights_path}')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f'Missing eraser_config.json: {config_path}')
        with open(config_path) as f:
            config = json.load(f)
        if config.get('eraser_type', 'adapter') != 'adapter':
            raise ValueError(f'Only adapter erasers are supported: {eraser_path}')
        checkpoint = torch.load(weights_path, map_location='cpu')
        if not checkpoint:
            raise ValueError(f'Eraser checkpoint has no layers: {weights_path}')
        return config, checkpoint

    def _load_checkpoints(self):
        for eraser_path in self.eraser_paths:
            config, checkpoint = self._load_eraser_folder(eraser_path)
            self.eraser_configs.append(config)
            self.eraser_checkpoints.append(checkpoint)
            self.layer_names.update(checkpoint.keys())

    def _build_adapter(self, attn_module, eraser_idx, layer_name):
        config = self.eraser_configs[eraser_idx]
        checkpoint = self.eraser_checkpoints[eraser_idx]
        eraser_rank = int(config['eraser_rank'])
        dim = attn_module.to_out[0].weight.shape[1]
        adapter = AdapterEraser(dim, eraser_rank)
        adapter.load_state_dict(checkpoint[layer_name])
        if self.device is not None or self.dtype is not None:
            adapter = adapter.to(device=self.device, dtype=self.dtype)
        for param in adapter.parameters():
            param.requires_grad = False
        return adapter

    def register(self, unet):
        if self.handles:
            raise RuntimeError('MultiEraserWrapper hooks are already registered.')

        try:
            matched_layers = set()
            for block_name, module in unet.named_modules():
                if not isinstance(module, BasicTransformerBlock):
                    continue
                layer_name = diffuser_prefix_name(block_name)
                active_erasers = [
                    eraser_idx for eraser_idx, checkpoint in enumerate(self.eraser_checkpoints)
                    if layer_name in checkpoint
                ]
                if not active_erasers:
                    continue

                matched_layers.add(layer_name)
                self.layer_to_adapter_keys[layer_name] = []
                self.layer_to_eraser_indices[layer_name] = active_erasers
                for eraser_idx in active_erasers:
                    key = self._adapter_key(eraser_idx, layer_name)
                    if key not in self.adapters:
                        self.adapters[key] = self._build_adapter(module.attn2, eraser_idx, layer_name)
                    self.layer_to_adapter_keys[layer_name].append(key)

                print(f'Load multi eraser hook at: {block_name} ({len(active_erasers)} active erasers)')
                self.handles.append(module.attn2.register_forward_hook(self._make_hook(layer_name)))

            missing_layers = sorted(self.layer_names - matched_layers)
            if missing_layers:
                raise ValueError(
                    'Some eraser checkpoint layers did not match the UNet: '
                    + ', '.join(missing_layers[:10])
                    + (' ...' if len(missing_layers) > 10 else '')
                )
            if not self.handles:
                raise ValueError('No matching cross-attention layers found for the provided erasers.')
        except Exception:
            self.remove()
            raise
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @contextmanager
    def disabled(self):
        old_enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = old_enabled

    def active_weights(self, eraser_indices):
        indices = torch.tensor(eraser_indices, device=self.logits.device, dtype=torch.long)
        logits = self.logits.index_select(0, indices)
        return F.softmax(logits, dim=0)

    def _make_hook(self, layer_name):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            if isinstance(output, tuple):
                hidden_states = output[0]
                suffix = output[1:]
            else:
                hidden_states = output
                suffix = None

            eraser_indices = self.layer_to_eraser_indices[layer_name]
            weights = self.active_weights(eraser_indices).to(device=hidden_states.device, dtype=hidden_states.dtype)
            fused_delta = torch.zeros_like(hidden_states)
            for weight, adapter_key in zip(weights, self.layer_to_adapter_keys[layer_name]):
                adapter = self.adapters[adapter_key]
                fused_delta = fused_delta + weight * adapter(hidden_states)

            fused = hidden_states + (self.fusion_scale * fused_delta)
            if suffix is not None:
                return (fused,) + suffix
            return fused
        return hook

    def normalized_weights(self):
        weights = F.softmax(self.logits.detach().cpu(), dim=0)
        return [float(weight) for weight in weights]

    def to_fusion_config(self):
        weights = self.normalized_weights()
        return {
            'fusion_type': 'active_softmax',
            'fusion_scale': self.fusion_scale,
            'erasers': [
                {'path': path, 'weight': weight}
                for path, weight in zip(self.eraser_paths, weights)
            ],
        }

    def save_fusion_config(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_fusion_config(), f, indent=4)
