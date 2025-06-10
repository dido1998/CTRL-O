import argparse
import json
import logging
import os
import pathlib
import pickle
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision
import tqdm
from ocl import visualizations
from ocl.cli import train
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import textwrap


from llm2vec import LLM2Vec

logging.getLogger().setLevel(logging.INFO)


# TODO: Use CVPR submission checkpoints --- these checkpoints are recent I suppose
CHECKPOINTS = {
    "checkpoint": "pretrained_models/ctrlo/pretrained_model.ckpt",
    "config": "pretrained_models/ctrlo/config.yaml",
}


l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)


def get_shard_pattern(path: str):
    base_pattern: str = "shard-%06d.tar"
    return os.path.join(path, base_pattern)

class FeatureExtractor:
    """Handles feature extraction for multiple vision models."""

    def __init__(self, device="cuda", batch_size=32):
        self.device = device
        self._init_models()
        self._init_transforms()

    def _init_models(self):
        # Initialize Ctrlo
        config_path = CHECKPOINTS[
            "config"
        ]  
        encoder_checkpoint_path = CHECKPOINTS[
            "checkpoint"
        ]
        oclf_config = OmegaConf.load(config_path)
        self.model = train.build_model_from_config(
            oclf_config, encoder_checkpoint_path
        ).to(self.device)
        self.model.eval()

    def _init_transforms(self):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def extract_features_batch(self, images, prompts):
        """Extract features for a batch of images."""
        images = torch.stack([self.base_transform(image) for image in images]).to(
            self.device
        )  # TODO: check if this is correct
        name_embeddings = torch.stack([l2v.encode(prompt) for prompt in prompts]).to(self.device)
        bsz = images.shape[0]
        inputs = {
            "image": images,
            "bbox_centroids": torch.tensor([[-1, -1]] * 7, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device),
            # "contrastive_loss_mask": torch.zeros(7).repeat(bsz, 1).to(self.device),
            # "name_embedding": torch.randn(bsz, 7, 512).to(
            #     self.device
            # ),  # TODO: replace with actual name embeddings
            "contrastive_loss_mask": torch.stack([torch.tensor([int(p != "other") for p in prompt]) for prompt in prompts]).to(self.device),
            "name_embedding": name_embeddings,
            "instance_bbox": torch.tensor(
                [[-1, -1, -1, -1]] * 7, dtype=torch.float32
            )
            .repeat(bsz, 1, 1)
            .to(self.device),  # TODO: this field was not before
            "batch_size": bsz,
        }
        outputs = self.model(inputs)
        features = outputs["perceptual_grouping"].objects

        # make sure feature shape makes sense
        return outputs

def visualize_features(outputs, prompts, images, device="cuda"):
    fig, axes = plt.subplots(len(prompts), 8, figsize=(24, len(prompts) * 5))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)  # Reduce horizontal and vertical spacing for a tighter fit

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    pixel_values_vit_1 = transform(images[0]).unsqueeze(0).repeat(len(prompts[0]), 1, 1, 1).to(device)
    pixel_values_vit_2 = transform(images[1]).unsqueeze(0).repeat(len(prompts[1]), 1, 1, 1).to(device)

    pixel_values_vit = torch.cat([pixel_values_vit_1, pixel_values_vit_2], dim=0)

    # Colors for different masks to visually differentiate them
    colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (1, 1, 0),  # Yellow
        (1, 0, 1),  # Magenta
        (0, 1, 1),  # Cyan
        (0.75, 0.75, 0.75),  # Bright Gray
    ]

    # Visualize for each prompt
    for i, prompt in enumerate(prompts):
        print(prompt)
        # Original image (resized for visualization)
        axes[i, 0].imshow(cv2.resize(pixel_values_vit[i * 7].squeeze(0).permute(1, 2, 0).cpu().numpy(), (768, 768), interpolation=cv2.INTER_NEAREST))
        axes[i, 0].axis('off')

        # Get masks for the current prompt
        image_shape = pixel_values_vit[i:i + 1].shape[2:]
        masks_as_image = outputs['object_decoder'].masks_as_image[i]
        masks = masks_as_image.view(-1, 1, *image_shape)
        
        # For each mask, visualize it superimposed on the original image
        for j in range(masks.shape[0]):
            
            # Superimpose the mask on the image
            curr_mask = masks[j].squeeze(0).cpu().detach().numpy()
            image_np = pixel_values_vit[i * 7 + j].squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Create a colored mask (using different colors for each mask)
            colored_mask = np.zeros((*curr_mask.shape, 3), dtype=np.float32)
            colored_mask[:, :, 0] = curr_mask * colors[j % len(colors)][0] * 1.5  # Amplify intensity by multiplying
            colored_mask[:, :, 1] = curr_mask * colors[j % len(colors)][1] * 1.5  # Amplify intensity by multiplying
            colored_mask[:, :, 2] = curr_mask * colors[j % len(colors)][2] * 1.5  # Amplify intensity by multiplying
            
            # Clip values to be in range [0, 1]
            colored_mask = np.clip(colored_mask, 0, 1)
            
            # Blend the image and the colored mask with a higher emphasis on the mask
            alpha = 0.45  # Increase this value to make the mask more prominent
            blended = image_np * (1 - alpha) + colored_mask * alpha
            
            # Resize to be larger (768x768)
            blended = cv2.resize(blended, (768, 768), interpolation=cv2.INTER_NEAREST)
            wrapped_text = textwrap.fill(prompt[j], width=20) if prompt[j] != "other" else ""

            # Plot the blended image
            axes[i, j + 1].imshow(blended)
            axes[i, j + 1].set_title(wrapped_text, fontsize=19, pad=7.5, fontweight='bold')  # Add padding to improve readability
            axes[i, j + 1].axis('off')

    # Add a border around subplots for better separation
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

    # Save the combined visualization
    combined_image_path = "images/vg_demo.png"
    try:
        plt.savefig(combined_image_path, bbox_inches='tight', dpi=300, pad_inches=0, transparent=True)  # Adjust dpi to balance quality and file size
        logging.info(f"Combined visualization saved to {combined_image_path}")
    except Exception as e:
        logging.error(f"Error saving visualization: {e}")
        raise

    print(f"Combined visualization saved to {combined_image_path}")


# you can specify upto 7 regions or objects phrases, the rest will be "other"
prompts = [
    ["The brocolli in the lunchbox", "the mashed potatoes", "other", "other", "other", "other", "other"],
    ["The girrafe eating the leaves", "The trees", "other", "other", "other", "other", "other"],
]
images = [
    Image.open("example_images/000000000009.jpg").convert("RGB"),
    Image.open("example_images/000000000025.jpg").convert("RGB"),
]

feature_extractor = FeatureExtractor()
outputs = feature_extractor.extract_features_batch(images, prompts)
visualize_features(outputs, prompts, images)
