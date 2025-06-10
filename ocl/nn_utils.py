from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

import ocl.typing


class ConcatFeatures(nn.Module):
    def forward(
        self, x1: ocl.typing.FeatureExtractorOutput, x2: ocl.typing.FeatureExtractorOutput
    ) -> ocl.typing.FeatureExtractorOutput:
        return ocl.typing.FeatureExtractorOutput(
            features=torch.cat((x1.features, x2.features), dim=-1),
            positions=x1.positions,
            aux_features=None,
        )
