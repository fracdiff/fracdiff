import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fracdiff import Fracdiff
from fracdiff import fdiff


class TestFracdiff:
    """
    Test `Fracdiff`.
    """

    def test_repr(self):
        fracdiff = Fracdiff(0.5, window=10, mode="full", window_policy="fixed")
        expected = "Fracdiff(d=0.5, window=10, mode=full, window_policy=fixed)"
        assert repr(fracdiff) == expected

    @pytest.mark.parametrize("d", [0.5])
    @pytest.mark.parametrize("window", [10])
    @pytest.mark.parametrize("mode", ["full", "valid"])
    def test_transform(self, d, window, mode):
        np.random.seed(42)
        X = np.random.randn(100, 200)
        fracdiff = Fracdiff(d=d, window=window, mode=mode)
        out = fdiff(X, n=d, axis=0, window=window, mode=mode)
        assert_array_equal(fracdiff.fit_transform(X), out)
