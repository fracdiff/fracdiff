from concurrent.futures import ProcessPoolExecutor

import numpy
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from .base import fdiff
from .fracdiff import Fracdiff
from .stat import StatTester


class FracdiffStat(TransformerMixin, BaseEstimator):
    """
    Carry out fractional derivative with the minumum order
    with which the differentiation becomes stationary.

    Parameters
    ----------
    window : int > 0 or None, default 10
        Number of observations to compute each element in the output.
    mode : {"full", "valid"}, default "full"
        "full" (default) :
            Return elements where at least one coefficient is used.
            Shape of a transformed array is the same with the original array.
            At the beginning of a transformed array, boundary effects may be seen.
        "valid" :
            Return elements where all coefficients are used.
            Output size along axis 1 is `n_features - window`.
            At the beginning of a time-series, boundary effects is not seen.
    window_policy : {"fixed"}, default "fixed"
        If "fixed" :
            Fixed window method.
            Every term in the output is evaluated using `window` observations.
            In other words, a fracdiff operator, which is a polynominal of a backshift
            operator, is truncated up to the `window`-th term.
            The beginning `window - 1` elements in output are filled with
            ``numpy.nan``.
        If "expanding" (not available) :
            Expanding window method.
            Every term in fracdiff time-series is evaluated using at least `window`
            observations.
            The beginning `window - 1` elements in output are filled with
            ``numpy.nan``.
    stattest : {"ADF"}, default "ADF"
        Method of stationarity test.
    pvalue : float, default 0.05
        Threshold of p-value to judge stationarity.
    precision : float, default .01
        Precision for the order of differentiation.
    upper : float, default 1.0
        Upper limit of the range to search the order.
    lower : float, default 0.0
        Lower limit of the range to search the order.
    n_jobs : int, default None
        The number of jobs to run `fit` in parallel. -1 means using all processors

    Attributes
    ----------
    d_ : numpy.array, shape (n_features,)
        Minimum order of fractional differentiation
        that makes time-series stationary.

    Note
    ----
    If `upper`th differentiation of series is still non-stationary,
    order_ is set to ``numpy.nan``.
    If `lower`th differentiation of series is already stationary,
    order_ is set to `lower`, but the true value may be smaller.

    Examples
    --------
    >>> numpy.random.seed(42)
    >>> X = numpy.random.randn(100, 4).cumsum(0)
    >>> f = FracdiffStat().fit(X)
    >>> f.d_
    array([0.140625 , 0.5078125, 0.3984375, 0.140625 ])
    >>> X = f.transform(X)
    """

    def __init__(
        self,
        window=10,
        mode="full",
        window_policy="fixed",
        stattest="ADF",
        pvalue=0.05,
        precision=0.01,
        upper=1.0,
        lower=0.0,
        n_jobs=None,
    ):
        self.window = window
        self.mode = mode
        self.window_policy = window_policy
        self.stattest = stattest
        self.pvalue = pvalue
        self.precision = precision
        self.upper = upper
        self.lower = lower
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Fit the model with `X`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Time-series to perform fractional differentiation.
            Here `n_samples` is the number of samples and `n_features` is the number of
            features.
        y : array_like, optional
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_array(X)

        self.d_ = self._find_features_d(X)

        return self

    def transform(self, X, y=None) -> numpy.array:
        """
        Return the fractional differentiation of `X`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_series)
            Time-series to perform fractional differentiation.
            Raises ValueError if `n_samples < self.window`.
        y : array_like, optional
            Ignored.

        Returns
        -------
        fdiff : ``numpy.array``, shape (n_samples, n_series)
            The fractional differentiation of `X`.
        """
        check_is_fitted(self, ["d_"])
        check_array(X)

        prototype = Fracdiff(0.5, window=self.window, mode=self.mode).fit_transform(X)
        out = numpy.empty_like(prototype[:, :0])

        for i in range(X.shape[1]):
            f = Fracdiff(self.d_[i], window=self.window, mode=self.mode)
            d = f.fit_transform(X[:, [i]])[-out.shape[0] :]
            out = numpy.concatenate((out, d), 1)

        return out

    def _is_stat(self, x) -> bool:
        return StatTester(method=self.stattest).is_stat(x, pvalue=self.pvalue)

    def _find_features_d(self, X) -> numpy.ndarray:
        features = (X[:, i] for i in range(X.shape[1]))

        if self.n_jobs is not None and self.n_jobs != 1:
            # If n_jobs == -1, use all CPUs
            max_workers = self.n_jobs if self.n_jobs != -1 else None
            with ProcessPoolExecutor(max_workers=max_workers) as exec:
                d_ = exec.map(self._find_d, features)
        else:
            d_ = map(self._find_d, features)

        return numpy.array(list(d_))

    def _find_d(self, x) -> float:
        """
        Carry out binary search of minimum order of fractional
        differentiation to make the time-series stationary.

        Parameters
        ----------
        x : array, shape (n,)

        Returns
        -------
        d : float
        """

        def diff(d):
            return fdiff(x, d, window=self.window, mode=self.mode)

        if not self._is_stat(diff(self.upper)):
            return numpy.nan
        if self._is_stat(diff(self.lower)):
            return self.lower

        upper, lower = self.upper, self.lower
        while upper - lower > self.precision:
            m = (upper + lower) / 2
            if self._is_stat(diff(m)):
                upper = m
            else:
                lower = m

        return upper
