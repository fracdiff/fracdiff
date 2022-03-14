import numpy as np
import pytest
import torch
from torch.testing import assert_close

import fracdiff
from fracdiff.torch import Fracdiff
from fracdiff.torch import fdiff


class TestTorchFracdiff:
    @pytest.mark.parametrize("d", [0.1, 0.5, 1])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_torch_fdiff(self, d, mode):
        torch.manual_seed(42)
        input = torch.randn(10, 100)

        result = fdiff(input, d, mode=mode)
        expect = torch.from_numpy(fracdiff.fdiff(input, d, mode=mode))
        assert_close(result, expect, check_stride=False)

        result = Fracdiff(d, mode=mode)(input)
        expect = torch.from_numpy(fracdiff.fdiff(input, d, mode=mode))
        assert_close(result, expect, check_stride=False)

    @pytest.mark.parametrize("d", [0.1, 0.5, 1])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_torch_fdiff_int(self, d, mode):
        torch.manual_seed(42)
        input = torch.randint(5, size=(10, 100))

        result = fdiff(input, d, mode=mode)
        expect = torch.from_numpy(fracdiff.fdiff(np.array(input), d, mode=mode))
        assert_close(result, expect, check_stride=False, check_dtype=False)

        result = Fracdiff(d, mode=mode)(input)
        expect = torch.from_numpy(fracdiff.fdiff(np.array(input), d, mode=mode))
        assert_close(result, expect, check_stride=False, check_dtype=False)

    @pytest.mark.parametrize("d", [0.1, 0.5, 1])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_torch_prepend_append(self, d, mode):
        torch.manual_seed(42)
        input = torch.randn(10, 100)
        prepend = torch.randn(10, 50)
        append = torch.randn(10, 50)

        expect = torch.from_numpy(
            fracdiff.fdiff(input, d, mode=mode, prepend=prepend, append=append)
        )
        result = fdiff(input, d, mode=mode, prepend=prepend, append=append)
        assert_close(result, expect, check_stride=False)

        expect = torch.from_numpy(
            fracdiff.fdiff(input, d, mode=mode, prepend=prepend, append=append)
        )
        result = Fracdiff(d, mode=mode)(input, prepend=prepend, append=append)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.parametrize("d", [0.1, 0.5, 1])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_torch_prepend_append_dim0(self, d, mode):
        torch.manual_seed(42)
        input = torch.randn(10, 100)
        prepend = torch.tensor([[1]]).expand(10, 1)
        append = torch.tensor([[2]]).expand(10, 1)

        expect = torch.from_numpy(
            fracdiff.fdiff(input, d, mode=mode, prepend=prepend, append=append)
        )
        result = fdiff(input, d, mode=mode, prepend=prepend, append=append)
        assert_close(result, expect, check_stride=False, check_dtype=False)

        result = Fracdiff(d, mode=mode)(input, prepend=prepend, append=append)
        assert_close(result, expect, check_stride=False, check_dtype=False)

    def test_repr(self):
        m = Fracdiff(0.1, dim=-1, window=10, mode="same")
        result = repr(m)
        expect = "Fracdiff(0.1, dim=-1, window=10, mode='same')"

        assert result == expect

    # torch.diff for n < -1 returns input

    # def test_invalid_n(self):
    #     with pytest.raises(ValueError):
    #         input = torch.empty(10, 100)
    #         _ = fdiff(input, -1)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            input = torch.empty(10, 100)
            _ = fdiff(input, 0.5, mode="invalid")

    # def test_invalid_dim(self):
    #     with pytest.raises(ValueError):
    #         input = torch.empty(10, 100)
    #         _ = fdiff(input, 0.5, dim=0)
