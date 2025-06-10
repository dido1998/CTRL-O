"""Utilities related to masking."""
import math
from typing import Optional

import torch
from torch import nn


class CreateSlotMask(nn.Module):
    """Module intended to create a mask that marks empty slots.

    Module takes a tensor holding the number of slots per batch entry, and returns a binary mask of
    shape (batch_size, max_slots) where entries exceeding the number of slots are masked out.
    """

    def __init__(self, max_slots: int):
        super().__init__()
        self.max_slots = max_slots

    def forward(self, n_slots: torch.Tensor) -> torch.Tensor:
        (batch_size,) = n_slots.shape

        # Create mask of shape B x K where the first n_slots entries per-row are false, the rest true
        indices = torch.arange(self.max_slots, device=n_slots.device)
        masks = indices.unsqueeze(0).expand(batch_size, -1) >= n_slots.unsqueeze(1)

        return masks


class CreateRandomMaskPatterns(nn.Module):
    """Create random masks.

    Useful for showcasing behavior of metrics.
    """

    def __init__(self, pattern: str, n_slots: Optional[int] = None, n_cols: int = 2):
        super().__init__()
        if pattern not in ("random", "blocks"):
            raise ValueError(f"Unknown pattern {pattern}")
        self.pattern = pattern
        self.n_slots = n_slots
        self.n_cols = n_cols

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if self.pattern == "random":
            rand_mask = torch.rand_like(masks)
            return rand_mask / rand_mask.sum(1, keepdim=True)
        elif self.pattern == "blocks":
            n_slots = masks.shape[1] if self.n_slots is None else self.n_slots
            height, width = masks.shape[-2:]
            new_masks = torch.zeros(
                len(masks), n_slots, height, width, device=masks.device, dtype=masks.dtype
            )
            blocks_per_col = int(n_slots // self.n_cols)
            remainder = n_slots - (blocks_per_col * self.n_cols)
            slot = 0
            for col in range(self.n_cols):
                rows = blocks_per_col if col < self.n_cols - 1 else blocks_per_col + remainder
                for row in range(rows):
                    block_width = math.ceil(width / self.n_cols)
                    block_height = math.ceil(height / rows)
                    x = col * block_width
                    y = row * block_height
                    new_masks[:, slot, y : y + block_height, x : x + block_width] = 1
                    slot += 1
            assert torch.allclose(new_masks.sum(1), torch.ones_like(masks[:, 0]))
            return new_masks


class FilterAndExpandMasks(nn.Module):
    """Keep largest masks and expand them to fill the whole image.

    Useful for evaluation.
    """

    def __init__(self, keep_n_largest: Optional[int] = 7):
        super().__init__()
        self.keep_n_largest = keep_n_largest

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        import skimage

        assert masks.ndim == 4

        masks = masks.detach()

        out_masks = []
        for mask in masks:
            bin_mask = mask > 0
            if self.keep_n_largest is not None:
                mask_sizes = bin_mask.sum((-2, -1))
                idxs = torch.argsort(mask_sizes, descending=True, stable=True)[: self.keep_n_largest]
                bin_mask = bin_mask[idxs]

            # Use increasing integers for different masks (label image format of skimage)
            bin_mask = bin_mask * (
                torch.arange(len(bin_mask), device=mask.device)[:, None, None] + 1
            )
            bin_mask_dense = bin_mask.max(0).values
            bin_mask_dense = bin_mask_dense.cpu().numpy()
            out_mask_dense = skimage.segmentation.expand_labels(bin_mask_dense, distance=1e7)

            out_mask_dense = torch.from_numpy(out_mask_dense).to(bin_mask.device)
            out_mask = nn.functional.one_hot(out_mask_dense, num_classes=len(bin_mask) + 1)
            out_mask = out_mask.transpose(-1, -2).transpose(-2, -3)[1:]

            out_mask_ = torch.zeros_like(mask)
            out_mask_[: len(out_mask)] = out_mask
            assert torch.all(out_mask_.sum(0) > 0)
            out_masks.append(out_mask_)

        return (torch.stack(out_masks) > 0).to(torch.float32)
