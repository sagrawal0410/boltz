import torch
import torch.nn as nn
from typing import Union, Any
import random
import numpy as np
from einops import einsum

compile_mode="default"

def custom_compile(**args):
    def wrapper(function):
        if compile_mode == "none":
            return function
        kwargs = {
            "mode": compile_mode,
        }
        kwargs.update(args)
        return torch.compile(function, **kwargs)
    return wrapper



def sg(x):
    return x.clone().detach()

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


