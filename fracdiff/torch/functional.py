from typing import Optional

try:
    import torch
    from torch import Tensor
    from torch.functional import conv1d
    PYTORCH_IS_IMPORTED = True
except ImportError:
    # TODO(simaki): Lazily raise ImportError
    Tensor = object  # For typing
    PYTORCH_IS_IMPORTED = False

from ..base import fdiff_coef


def fdiff(
    input: Tensor,
    n: int = 1.0,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
    window: int = 10,
    mode: str = "full",
):
    """Return the `n`-th differentiation along the given axis.

    Args:
        input (torch.Tensor): The input tensor.
        n (float):
        dim (int, default=-1): ...

    Returns:
        torch.Tensor

    Shape:
        - input: :math:`(N, *, L_{\\mathrm{in}})` where :math:`N` is ...
        - output: :math:`(N, *, L_{\\mathrm{out}})` where ...

    Examples:

        >>> from fracdiff.torch import fdiff
        >>> input = torch.tensor([1, 2, 4, 7, 0])
        >>> fdiff(input, 0.5)
    """
    if prepend is not None:
        # TODO(simaki): Implement prepend. PR welcomed!
        raise ValueError("prepend is not supported.")
    if append is not None:
        # TODO(simaki): Implement append. PR welcomed!
        raise ValueError("prepend is not supported.")
    if dim != -1:
        # TODO(simaki): Implement dim != -1. PR welcomed!
        raise ValueError("Only supports dim == -1.")
    if mode != "full":
        # TODO(simaki): Implement mode != "full". PR welcomed!
        raise ValueError('Only supports mode == "full"')

    # Return `np.diff(...)` if n is integer
    if isinstance(n, int) or n.is_integer():
        # TODO(simaki): PyTorch implementation of diff
        return np.diff(a, n=int(n), axis=axis, prepend=prepend, append=append)

    input_size = input.size()
    input = input.reshape((sum(input_size[:-1]), 1, input_size[-1]))

    # TODO(simaki): PyTorch Implementation to create weight
    weight = torch.as_tensor(fdiff_coef(n, window)).to(input).reshape(1, 1, -1)

    return conv1d(input, weight)
