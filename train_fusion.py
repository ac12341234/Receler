import argparse
import os
import random

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from receler.multi_eraser import MultiEraserWrapper, normalize_fusion_inputs


def infer_concepts_from_eraser_paths(eraser_paths):
    concepts = []
    for eraser_path in eraser_paths:
        folder_name = os.path.basename(os.path.normpath(eraser_path))
        marker = 'word_'
        if marker not in folder_name:
            continue
        concept_part = folder_name.split(marker, 1)[1].split('-', 1)[0]
        concept = concept_part.replace('_', ' ').strip()
        if concept:
            concepts.append(concept)
    return concepts


def encode_prompts(pipeline, prompts, device):
    text_inputs = pipeline.tokenizer(
        prompts,
        padding='max_length',
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
    )
    input_ids = text_inputs.input_ids.to(device)
    return pipeline.text_encoder(input_ids)[0]


def freeze_pipeline(pipeline):
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()


def train_fusion(
        concept,
        eraser_paths,
        fusion_weights=None,
        output_path='fusion_config.json',
        base_model='CompVis/stable-diffusion-v1-4',
        iterations=500,
        lr=1e-2,
        negative_guidance=1.0,
        image_size=512,
        seed=None,
    ):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline = StableDiffusionPipeline.from_pretrained(base_model, safety_checker=None)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder='scheduler')
    pipeline = pipeline.to(device)
    freeze_pipeline(pipeline)

    wrapper = MultiEraserWrapper(
        eraser_paths,
        fusion_weights=fusion_weights,
        trainable_weights=True,
        device=device,
    )
    wrapper.register(pipeline.unet)
    wrapper.to(device)
    for param in wrapper.adapters.parameters():
        param.requires_grad = False
    wrapper.logits.requires_grad = True

    optimizer = torch.optim.Adam([wrapper.logits], lr=lr)

    if concept is None:
        concepts = infer_concepts_from_eraser_paths(eraser_paths)
    elif ',' in concept:
        concepts = [item.strip() for item in concept.split(',') if item.strip()]
    else:
        concepts = [concept]
    if not concepts:
        raise ValueError(
            '--concept was not provided and concepts could not be inferred from eraser folder names. '
            'Use folders like "receler-word_cat-..." or pass --concept explicitly.'
        )

    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    latent_size = image_size // vae_scale_factor
    latent_shape = (1, pipeline.unet.config.in_channels, latent_size, latent_size)

    pbar = tqdm(range(iterations))
    for _ in pbar:
        word = random.choice(concepts)
        emb_0 = encode_prompts(pipeline, [''], device)
        emb_p = encode_prompts(pipeline, [word], device)

        latents = torch.randn(latent_shape, device=device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (1,),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad(), wrapper.disabled():
            e_0 = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=emb_0).sample
            e_p = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=emb_p).sample

        wrapper.enabled = True
        e_n = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=emb_p).sample
        target = e_0 - (negative_guidance * (e_p - e_0))
        loss = F.mse_loss(e_n, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({
            'loss': float(loss.detach().cpu()),
            'weights': ','.join(f'{weight:.3f}' for weight in wrapper.normalized_weights()),
        }, refresh=False)

    wrapper.save_fusion_config(output_path)
    print(f'Saved fusion config to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='TrainFusion',
        description='Train learnable fusion weights for multiple Receler erasers using Diffusers.',
    )
    parser.add_argument('--concept', help='concept(s) to erase; comma-separated concepts are sampled during training. If omitted, inferred from eraser folder names.', type=str, default=None)
    parser.add_argument('--eraser_paths', help='comma-separated Receler eraser folders', type=str, default=None)
    parser.add_argument('--fusion_weight', help='comma-separated initial fusion weights for --eraser_paths', type=str, default=None)
    parser.add_argument('--fusion_config', help='existing fusion_config.json to initialize paths and weights', type=str, default=None)
    parser.add_argument('--output_path', help='path to save learned fusion_config.json', type=str, default='fusion_config.json')
    parser.add_argument('--base_model', help='Diffusers base model name or path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--iterations', help='number of fusion training iterations', type=int, default=500)
    parser.add_argument('--lr', help='learning rate for fusion logits', type=float, default=1e-2)
    parser.add_argument('--negative_guidance', help='negative guidance used in Receler erase loss', type=float, default=1.0)
    parser.add_argument('--image_size', help='image size used to infer latent resolution', type=int, default=512)
    parser.add_argument('--seed', help='optional random seed', type=int, default=None)

    args = parser.parse_args()
    try:
        eraser_paths, fusion_weights = normalize_fusion_inputs(
            eraser_paths=args.eraser_paths,
            fusion_weights=args.fusion_weight,
            fusion_config=args.fusion_config,
        )
    except ValueError as error:
        parser.error(str(error))
    if not eraser_paths:
        parser.error('--eraser_paths or --fusion_config is required.')

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    train_fusion(
        concept=args.concept,
        eraser_paths=eraser_paths,
        fusion_weights=fusion_weights,
        output_path=args.output_path,
        base_model=args.base_model,
        iterations=args.iterations,
        lr=args.lr,
        negative_guidance=args.negative_guidance,
        image_size=args.image_size,
        seed=args.seed,
    )
