from typing import Optional

import torch
from torch import Tensor

from fracdiff.fdiff import fdiff_coef as fdiff_coef_numpy


def fdiff_coef(d: float, window: int) -> Tensor:
    """Returns sequence of coefficients in fracdiff operator.

    Args:
        d (float): Order of differentiation.
        window (int): Number of terms.

    Returns:
        torch.Tensor

    Examples:
        >>> from fracdiff.torch import fdiff_coef
        >>>
        >>> fdiff_coef(0.5, 4)
        tensor([ 1.0000, -0.5000, -0.1250, -0.0625], dtype=torch.float64)
        >>> fdiff_coef(1.0, 4)
        tensor([ 1., -1.,  0., -0.], dtype=torch.float64)
        >>> fdiff_coef(1.5, 4)
        tensor([ 1.0000, -1.5000,  0.3750,  0.0625], dtype=torch.float64)
    """
    return torch.as_tensor(fdiff_coef_numpy(d, window))


def fdiff(
    input: Tensor,
    n: float,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
    window: int = 10,
    mode: str = "same",
) -> Tensor:
    r"""Computes the ``n``-th differentiation along the given dimension.

    This is an extension of :func:`torch.diff` to fractional differentiation.
    See :class:`fracdiff.torch.Fracdiff` for details.

    Note:
        For integer ``n``, the output is the same as :func:`torch.diff`
        and the parameters ``window`` and ``mode`` are ignored.

    Shape:
        - input: :math:`(N, *, L_{\mathrm{in}})`, where where :math:`*` means any
          number of additional dimensions.
        - output: :math:`(N, *, L_{\mathrm{out}})`, where :math:`L_{\mathrm{out}}`
          is given by :math:`L_{\mathrm{in}}` if `mode="same"` and
          :math:`L_{\mathrm{in}} - \mathrm{window} - 1` if `mode="valid"`.
          If `prepend` and/or `append` are provided, then :math:`L_{\mathrm{out}}`
          increases by the number of elements in each of these tensors.

    Examples:
        >>> from fracdiff.torch import fdiff
        ...
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
    # Calls torch.diff if n is an integer
    if isinstance(n, int) or n.is_integer():
        return input.diff(n=int(n), dim=dim, prepend=prepend, append=append)

    if dim != -1:
        # TODO(simaki): Implement dim != -1. PR welcomed!
        raise ValueError("Only supports dim == -1.")

    if not input.is_floating_point():
        input = input.to(torch.get_default_dtype())

    combined = []
    if prepend is not None:
        prepend = torch.as_tensor(prepend).to(input)
        if prepend.dim() == 0:
            size = list(input.size())
            size[dim] = 1
            prepend = prepend.broadcast_to(torch.Size(size))
        combined.append(prepend)

    combined.append(input)

    if append is not None:
        append = torch.as_tensor(append).to(input)
        if append.dim() == 0:
            size = list(input.size())
            size[dim] = 1
            append = append.broadcast_to(torch.Size(size))
        combined.append(append)

    if len(combined) > 1:
        input = torch.cat(combined, dim=dim)

    input_size = input.size()
    input = input.reshape(input[..., 0].numel(), 1, input_size[-1])
    input = torch.nn.functional.pad(input, (window - 1, 0))

    # TODO(simaki): PyTorch Implementation to create weight
    weight = fdiff_coef(n, window).to(input).reshape(1, 1, -1).flip(-1)

    output = torch.nn.functional.conv1d(input, weight)

    if mode == "same":
        size_lastdim = input_size[-1]
    elif mode == "valid":
        size_lastdim = input_size[-1] - window + 1
    else:
        raise ValueError("Invalid mode: " + str(mode))

    output_size = input_size[:-1] + (size_lastdim,)
    output = output[..., -size_lastdim:].reshape(output_size)

    output = output.transpose(dim, -1)

    return output
