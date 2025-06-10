import ast
import math
import os
import re
from distutils.util import strtobool
from functools import reduce
from typing import Any, Callable

import yaml
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def torchvision_interpolation_mode(mode: str):
    import torchvision

    return torchvision.transforms.InterpolationMode[mode.upper()]


def torch_dtype(dtype: str):
    import torch

    dtype = getattr(torch, dtype)
    if dtype is None:
        return ValueError(f"torch.{dtype} does not exist")
    if not isinstance(dtype, torch.dtype):
        return ValueError(f"torch.{dtype} is not a datatype")

    return dtype


def lambda_string_to_function(function_string: str) -> Callable[..., Any]:
    """Convert string of the form "lambda x: x" into a callable Python function."""
    # This is a bit hacky but ensures that the syntax of the input is correct and contains
    # a valid lambda function definition without requiring to run `eval`.
    parsed = ast.parse(function_string)
    is_lambda = isinstance(parsed.body[0], ast.Expr) and isinstance(parsed.body[0].value, ast.Lambda)
    if not is_lambda:
        raise ValueError(f"'{function_string}' is not a valid lambda definition.")

    return eval(function_string)


class ConfigDefinedLambda:
    """Lambda function defined in the config.

    This allows lambda functions defined in the config to be pickled.
    """

    def __init__(self, function_string: str):
        self.__setstate__(function_string)

    def __getstate__(self) -> str:
        return self.function_string

    def __setstate__(self, function_string: str):
        self.function_string = function_string
        self._fn = lambda_string_to_function(function_string)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def eval_lambda(function_string, *args):
    lambda_fn = lambda_string_to_function(function_string)
    return lambda_fn(*args)


def resolver_eval(fn: str, *args):
    params, _, body = fn.partition(":")
    if body == "":
        body = params
        params = ""

    if len(params) == 0:
        arg_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        assert len(args) <= len(arg_names), f"Only up to {len(arg_names)} arguments are supported"
        params = ",".join(arg_names[: len(args)])

    if not params.startswith("lambda "):
        params = "lambda " + params

    return eval(f"{params}: {body}")(*args)


def make_slice(expr):
    if isinstance(expr, int):
        return expr

    pieces = [s and int(s) or None for s in expr.split(":")]
    if len(pieces) == 1:
        return slice(pieces[0], pieces[0] + 1)
    else:
        return slice(*pieces)


def slice_string(string: str, split_char: str, slice_str: str) -> str:
    """Split a string according to a split_char and slice.

    If the output contains more than one element, join these using the split char again.
    """
    sl = make_slice(slice_str)
    res = string.split(split_char)[sl]
    if isinstance(res, list):
        res = split_char.join(res)
    return res


def read_yaml(path):
    with open(to_absolute_path(path), "r") as f:
        return yaml.safe_load(f)


def when_testing(output_testing, output_otherwise):
    running_tests = bool(strtobool(os.environ.get("RUNNING_TESTS", "false")))
    return output_testing if running_tests else output_otherwise


def isqrt(a: float, safe: bool = True) -> int:
    a_sqrt = math.sqrt(a)
    if int(a_sqrt) != a_sqrt:
        raise ValueError(f"{a} does not have a perfect square (sqrt(a) == {a_sqrt})")
    return int(a_sqrt)


def get_timm_model_dim(model: str) -> int:
    if "vit_" in model:
        vit_sizes = {
            "tiny": 192,
            "small": 384,
            "medium": 512,
            "base": 768,
            "large": 1024,
            "huge": 1280,
            "giant_patch14_dinov2": 1536,
            "giant_patch14_reg4_dinov2": 1536,
            "giant": 1408,
            "gigantic": 1664,
        }
        for size, dim in vit_sizes.items():
            if size in model:
                return dim

    raise ValueError(f"Dimension for model {model} is unknown")


def get_timm_model_num_patches(model: str, image_size: int) -> int:
    patch_size = get_timm_model_patch_size(model)
    if image_size % patch_size != 0:
        raise ValueError(f"Image size {image_size} does not work with patch size {patch_size}")
    return (image_size // patch_size) ** 2


def get_timm_model_patch_size(model: str) -> int:
    match = re.match(r".*vit.+patch(\d+).*", model)
    if match and len(match.groups()) == 1:
        return int(match.group(1))

    raise ValueError(f"Patch size for model {model} is unknown")


OmegaConf.register_new_resolver("torchvision_interpolation_mode", torchvision_interpolation_mode)
OmegaConf.register_new_resolver("torch_dtype", torch_dtype)
OmegaConf.register_new_resolver("lambda_fn", ConfigDefinedLambda)
OmegaConf.register_new_resolver("eval_lambda", eval_lambda)
OmegaConf.register_new_resolver("slice", slice_string)
OmegaConf.register_new_resolver("read_yaml", read_yaml)
OmegaConf.register_new_resolver("when_testing", when_testing)
OmegaConf.register_new_resolver("timm_model_dim", get_timm_model_dim)
OmegaConf.register_new_resolver("timm_model_num_patches", get_timm_model_num_patches)
OmegaConf.register_new_resolver("timm_model_patch_size", get_timm_model_patch_size)
OmegaConf.register_new_resolver("eval", resolver_eval)
OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda *args: reduce(lambda prod, cur: prod * cur, args, 1))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("isqrt", lambda a, safe=True: isqrt(a, safe))
OmegaConf.register_new_resolver("min", lambda *args: min(*args))
OmegaConf.register_new_resolver("max", lambda *args: max(*args))
