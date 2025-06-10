import os
from typing import Sequence, Union

import pytest
import torch


@pytest.fixture
def assert_tensors_equal():
    def _assert_tensors_equal(
        tensor: Union[torch.Tensor, Sequence[torch.Tensor]],
        expected_tensor: Union[torch.Tensor, Sequence[torch.Tensor]],
    ):
        if not isinstance(tensor, Sequence):
            tensor = [tensor]
        if not isinstance(expected_tensor, Sequence):
            expected_tensor = [expected_tensor]
        assert len(tensor) == len(expected_tensor)

        for t, t_exp in zip(tensor, expected_tensor):
            assert isinstance(t, torch.Tensor)
            assert t.shape == t_exp.shape
            assert torch.allclose(t, t_exp)

    return _assert_tensors_equal


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_generate_tests(metafunc):
    # Set variable that tests are running.
    os.environ["RUNNING_TESTS"] = "true"
