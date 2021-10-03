import builtins
import sys
from unittest.mock import patch

import numpy as np
import pytest

builtin_import = builtins.__import__


def import_except_torch(name, *args, **kwargs):
    if name == "torch":
        raise ImportError
    else:
        return builtin_import(name, *args, **kwargs)


def test_torch_importerror():
    with patch("builtins.__import__", import_except_torch):
        # one can use fdiff and Fracdiff since torch is optional
        import fracdiff
        from fracdiff import fdiff
        from fracdiff.sklearn import Fracdiff

        a = np.random.randn(10, 100)
        _ = fdiff(a, 0.5)
        a = np.random.randn(10, 100)
        _ = Fracdiff(0.5).fit(a).transform(a)
