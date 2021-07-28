from math import prod
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as fn
    from torch import Tensor

    PYTORCH_IS_IMPORTED = True
except ImportError:
    # TODO(simaki): Lazily raise ImportError
    Tensor = object  # For typing
    PYTORCH_IS_IMPORTED = False

from .. import base


def is_floattype(dtype: torch.dtype):
    return dtype in (torch.float16, torch.float32, torch.float64)


def fdiff_coef(d: float, window: int) -> Tensor:
    """Returns sequence of coefficients in fracdiff operator.

    Args:
        d : float
            Order of differentiation.
        window : int
            Number of terms.

    Returns:
        torch.Tensor

    Examples:
        >>> fdiff_coef(0.5, 4)
        tensor([ 1.0000, -0.5000, -0.1250, -0.0625], dtype=torch.float64)
        >>> fdiff_coef(1.0, 4)
        tensor([ 1., -1.,  0., -0.], dtype=torch.float64)
        >>> fdiff_coef(1.5, 4)
        tensor([ 1.0000, -1.5000,  0.3750,  0.0625], dtype=torch.float64)
    """
    return torch.as_tensor(base.fdiff_coef(d, window))


def fdiff(
    input: Tensor,
    d: float,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
    window: int = 10,
    mode: str = "same",
):
    """Return the `d`-th differentiation along the given axis.

    Args:
        input (torch.Tensor): The input tensor.
        d (float):
        dim (int, default=-1): ...
        mode (str, default="same"): "same" or "valid".

    Returns:
        torch.Tensor

    Shape:
        - input: :math:`(N, *, L_{\\mathrm{in}})` where :math:`N` is ...
        - output: :math:`(N, *, L_{\\mathrm{out}})` where ...

    Examples:

        >>> from fracdiff.torch import fdiff
        >>> input = torch.tensor([1, 2, 4, 7, 0])
        >>> fdiff(input, 0.5, mode="same", window=3)
        tensor([ 1.0000,  1.5000,  2.8750,  4.7500, -4.0000])

        >>> fdiff(input, 0.5, mode="valid", window=3)
        tensor([ 2.8750,  4.7500, -4.0000])

        >>> fdiff(input, 0.5, mode="valid", window=3, prepend=[1, 1])
        tensor([ 0.3750,  1.3750,  2.8750,  4.7500, -4.0000])

        >>> input = torch.arange(10).reshape(2, 5)
        >>> fdiff(input, 0.5)
        tensor([[0.0000, 1.0000, 1.5000, 1.8750, 2.1875],
                [5.0000, 3.5000, 3.3750, 3.4375, 3.5547]])
    """
    if not PYTORCH_IS_IMPORTED:
        raise ImportError("`fracdiff.torch.fdiff` needs `torch` to be installed.")

    if dim != -1:
        # TODO(simaki): Implement dim != -1. PR welcomed!
        raise ValueError("Only supports dim == -1.")

    if not is_floattype(input.dtype):
        input = input.to(torch.get_default_dtype())

    # Return `np.diff(...)` if d is integer
    if isinstance(d, int) or d.is_integer():
        # TODO(simaki): PyTorch implementation of diff
        return torch.as_tensor(
            np.diff(input, n=int(n), axis=dim, prepend=prepend, append=append)
        )

    combined = []
    if prepend is not None:
        prepend = torch.as_tensor(prepend).to(input)
        if prepend.ndim == 0:
            size = list(input.size())
            size[dim] = 1
            prepend = prepend.broadcast_to(torch.Size(size))
        combined.append(prepend)

    combined.append(input)

    if append is not None:
        append = torch.as_tensor(append).to(input)
        if append.ndim == 0:
            size = list(input.size())
            size[dim] = 1
            append = append.broadcast_to(torch.Size(size))
        combined.append(append)

    if len(combined) > 1:
        input = torch.cat(combined, dim)

    input_size = input.size()
    input = input.reshape(prod(input_size[:-1]), 1, input_size[-1])
    input = fn.pad(input, (window - 1, 0))

    # TODO(simaki): PyTorch Implementation to create weight
    weight = fdiff_coef(d, window).to(input).reshape(1, 1, -1).flip(-1)

    output = fn.conv1d(input, weight)

    if mode == "same":
        size = input_size[-1]
    elif mode == "valid":
        size = input_size[-1] - window + 1
    else:
        raise ValueError("Invalid mode: " + str(mode))

    output_size = input_size[:-1] + (size,)

    return output[..., -size:].reshape(output_size)
