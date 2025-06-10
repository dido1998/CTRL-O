"""Utils for filtering checkpoints before loading them into the model."""
from typing import Dict, Tuple

import timm
import torch


class ResamplePositionEmbedding:
    """Resample a positional embedding in a checkpoint."""

    def __init__(self, path: str, size: Tuple[int, int]):
        self.path = path
        self.size = size

    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, val in state_dict.items():
            if key == self.path:
                if val.ndim != 3:
                    raise ValueError(
                        f"Pos embed to resample under path {key} should have shape (1, n_tokens, "
                        f"dims), but has shape {val.shape}"
                    )
                state_dict[key] = timm.layers.resample_abs_pos_embed(
                    val, self.size, num_prefix_tokens=0
                )
                break
        else:
            raise ValueError(
                f"Did not find pos embed under path {self.path} in state dict. "
                f"Available options are {list(state_dict.keys())}"
            )

        return state_dict
