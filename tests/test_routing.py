import torch

import routed
from ocl.utils import routing


def test_combined_topological_order():
    modules = {
        "linear2": routed.torch.nn.Linear(3, 2, input_path="batch.y"),
        "sigmoid1": routed.torch.nn.Sigmoid(input_path="linear1"),
        "bilinear": routed.torch.nn.Bilinear(
            4, 2, 5, input1_path="sigmoid1", input2_path="sigmoid2"
        ),
        "linear1": routed.torch.nn.Linear(2, 4, input_path="batch.x"),
        "sigmoid2": routed.torch.nn.Sigmoid(input_path="linear2"),
    }
    combined = routing.Combined(**modules)

    inputs = {"batch": {"x": torch.randn(4, 2), "y": torch.randn(4, 3)}}

    outputs = combined(inputs=inputs)
    assert outputs["bilinear"].shape == (4, 5)
