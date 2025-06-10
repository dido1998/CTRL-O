"""Types used in object centric learning framework."""
import dataclasses
from typing import Dict, Iterable, Optional, Union

import torch
from torchtyping import TensorType

# Input data types.
ImageData = TensorType["batch size", "channels", "height", "width"]  # noqa: F821
VideoData = TensorType["batch size", "frames", "channels", "height", "width"]  # noqa: F821
ImageOrVideoData = Union[VideoData, ImageData]  # noqa: F821
TextData = TensorType["batch_size", "max_tokens"]  # noqa: F821

# Feature data types.
CNNImageFeatures = ImageData
TransformerImageFeatures = TensorType[
    "batch_size", "n_spatial_features", "feature_dim"
]  # noqa: F821
ImageFeatures = TransformerImageFeatures
VideoFeatures = TensorType["batch_size", "frames", "n_spatial_features", "feature_dim"]  # noqa: F821
ImageOrVideoFeatures = Union[ImageFeatures, VideoFeatures]
Positions = Union[
    TensorType["n_spatial_features", "spatial_dims"],  # noqa: F821
    TensorType["batch_size", "n_spatial_features", "spatial_dims"],  # noqa: F821
]
PooledFeatures = TensorType["batch_size", "feature_dim"]

# Object feature types.
ObjectFeatures = TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
EmptyIndicator = TensorType["batch_size", "n_objects"]  # noqa: F821
ObjectFeatureAttributions = TensorType["batch_size", "n_objects", "n_spatial_features"]  # noqa: F821

# Module output types.
ConditioningOutput = TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
"""Output of conditioning modules."""


@dataclasses.dataclass
class FrameFeatures:
    """Features associated with a single frame."""

    features: ImageFeatures
    positions: Positions


@dataclasses.dataclass
class FeatureExtractorOutput:
    """Output of feature extractor."""

    features: ImageOrVideoFeatures
    positions: Positions
    aux_features: Optional[Dict[str, torch.Tensor]] = None

    def __iter__(self) -> Iterable[ImageFeatures]:
        """Iterate over features and positions per frame."""
        if self.features.ndim == 4:
            yield FrameFeatures(self.features, self.positions)
        else:
            for frame_features in torch.split(self.features, 1, dim=1):
                yield FrameFeatures(frame_features.squeeze(1), self.positions)


@dataclasses.dataclass
class PerceptualGroupingOutput:
    """Output of a perceptual grouping algorithm."""

    objects: ObjectFeatures
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821


@dataclasses.dataclass
class ControllablePerceptualGroupingOutput:
    """Output of a perceptual grouping algorithm."""

    objects: ObjectFeatures
    dual_objects: ObjectFeatures
    point_objects: ObjectFeatures
    lang_objects: ObjectFeatures
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    dual_feature_attributions: Optional[ObjectFeatureAttributions] = None
    point_feature_attributions: Optional[ObjectFeatureAttributions] = None
    lang_feature_attributions: Optional[ObjectFeatureAttributions] = None
    dual_attn_1: Optional[ObjectFeatureAttributions] = None
    dual_attn_2: Optional[ObjectFeatureAttributions] = None
    dual_attn_3: Optional[ObjectFeatureAttributions] = None
