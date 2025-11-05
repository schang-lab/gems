from typing import TYPE_CHECKING, Optional, Union, Callable, Tuple, List, Literal

import numpy as np
import ot  # python optimal transport library


def _wasserstein_distance_nd(P: np.ndarray,
                            Q: np.ndarray,
                            ordinality: List[np.ndarray],
                            metric: Union[str, Callable] = "euclidean",
                            sinkhorn_eps: Optional[float] = None,
                            return_log: bool = False,
                            jitter: float = 1e-10,
                            ) -> Union[float, Tuple[float, dict]]:
    """
    Wasserstein-1 distance between two probability tensors on an n-D grid.
    ----------
    [Parameters]
    P, Q : np.ndarray, same shape (d1, d2, …, dn)
        Discrete probability tensors whose entries sum to 1.
        Each dimension i has length d_i (number of bins).
    ordinality : List[np.ndarray]
        List of nD elements, where each element is list of length di floats.
    metric : {"euclidean", "cityblock"} | callable, optional
        How to measure ground distance between grid points.
        • "euclidean" -> L2 distance  (default)
        • "cityblock" -> L1 distance
        • callable(x, y) -> returns distance between two 1-D coordinate vectors.
    sinkhorn_eps : float | None, optional
        If given, the distance is approximated with entropic-regularised
        Sinkhorn divergence (faster for large grids).  Typical values:
        1e-2 … 1e-1.  If None (default), the exact EMD is computed.
    return_log : if True (only for Sinkhorn) also return the POT log dict
    jitter : added to every bin to avoid 0-mass entries
    -------
    [Returns]
    float
        The Wasserstein-1 distance (same units as the chosen metric).
    """
    assert isinstance(P, np.ndarray), "P must be a numpy array"
    assert isinstance(Q, np.ndarray), "Q must be a numpy array"
    assert P.shape == Q.shape, "P and Q must have the same shape"
    assert len(ordinality) == P.ndim, "ord. - dist. dimension mismatch"

    P = P.astype(float, copy=True) + jitter
    Q = Q.astype(float, copy=True) + jitter
    P /= P.sum()
    Q /= Q.sum()

    shape, ndim = P.shape, P.ndim
    axes = [np.asarray(ax, dtype=float) for ax in ordinality]
    coords = np.stack(
        np.meshgrid(*axes, indexing="ij"), axis=-1
    ).reshape(-1, ndim)
    P_flat = P.ravel()
    Q_flat = Q.ravel()
    C = ot.utils.dist(coords, coords, metric=metric)

    if sinkhorn_eps is None:
        distance, log = ot.emd2(P_flat, Q_flat, C), None
    else:
        distance, log = ot.sinkhorn2(
            P_flat, Q_flat, C, reg=sinkhorn_eps, log=return_log
        )
    return (
        float(distance / C.max()), log
    ) if return_log else float(distance / C.max())


def _correlation(P: np.ndarray,
                 ordinality: List[np.ndarray],
                 weights: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
    """
    Compute the correlation matrix of P, which has shape (n_respondent, n_items).
    Each column of P is a random variable (each item).
    The correlation matrix is a square matrix of shape (n_items, n_items).
    Ordinality is a list of n_items 1D arrays, where each array is
    the ordinality of the corresponding item. (example: [0.0, 1.0, 0.5])
    Weights is a 1D array of shape (n_respondent,).
    """
    assert P.ndim == 2, "P must be a 2D array"
    n_respondent, n_items = P.shape
    if weights is not None:
        assert (
            weights.ndim == 1
            and weights.shape[0] == n_respondent
        ), "This should not happen!"
        weights = weights.astype(float, copy=True)
        weights /= np.sum(weights)
    else:
        weights = np.ones(P.shape[0], dtype=float) / P.shape[0]

    P_mapped = np.empty_like(P, dtype=float)
    for j in range(n_items):
        mapping = np.asarray(ordinality[j], dtype=float)
        col = P[:, j]
        assert col.dtype == np.int64, (
            f"Column {j} of response data is not inttype. "
            "Initialization of Distribution has encountered an error."
        )
        assert np.all((col >= 0) & (col < mapping.size)), (
            f"Column {j} of response data has values outside the allowed range"
        )
        P_mapped[:, j] = mapping[col]

    mean_w = np.sum(P_mapped * weights[:, np.newaxis], axis=0)
    P_centered = P_mapped - mean_w
    cov_w = np.dot(
            (P_centered * weights[:, None]).T, 
            P_centered,
        ) / (1 - np.sum(weights ** 2))
    stddev_w = np.sqrt(np.diag(cov_w))
    corr_w = cov_w / np.outer(stddev_w, stddev_w)
    return corr_w


class Distribution:

    def __init__(self,
                 data: np.ndarray,
                 ordinality: List[List[float]],
                 datatype: Literal["individual", "distribution"] = "individual",
                 weights: Optional[np.ndarray] = None,
                 qkey: Optional[Union[str, List[str]]] = None):
        """
        Initialization.
        ----------
        [Parameters]
        data: individual responses or pre-calculated distribution
            when datatype = individual, a 2D array of shape (n_respondent, n_items)
                values appearing in each column (length n_respondent) is
                guaranteed to be contiguous integers.
            when datatype = distribution, a nD array of shape (d1, d2, …, dn)
                where di is the number of bins in dimension (question) i
        datatype
            whether data is a set of individual responses or a pre-calculated distribution
        ordinality
            a list of nD elements, where each element is list of length di floats
            must be provided, as survey question options have a natural order.
            For categorical questions, must pre-define distance (cost) matrix.
        weights
            only provided when datatype = individual
            a 1D array of shape (n_respondent,)
        qkey
            a string or list of strings to identify the origin of the distribution
            only used for the tracking purpose (which question the distribution is from)
        """
        assert datatype in ["individual", "distribution"], "invalid datatype"
        self.datatype = datatype
        self.qkey = [qkey] if isinstance(qkey, str) else qkey

        if datatype == "individual":
            self.n_respondent, self.n_items = data.shape
            if weights is None:
                self.weights = np.ones(self.n_respondent, dtype=float)
            else:
                assert len(weights) == self.n_respondent, "weight - data length mismatch"
                self.weights = weights.astype(float)
            self.weights /= np.sum(self.weights)
            self.data = (data - data.min(axis=0)).astype(int)
            self.cardinality = (self.data.max(axis=0) + 1).astype(int)
            self.pmf = np.zeros(self.cardinality, dtype=float)
            flat_idx = np.ravel_multi_index(self.data.T, dims=self.cardinality)
            np.add.at(self.pmf.ravel(), flat_idx, self.weights)
            self.pmf /= self.pmf.sum()
    
        else: # datatype == "distribution"
            assert weights is None, "no weights for pre-calculated distribution"
            self.n_respondent, self.n_items = -1, data.ndim
            self.weights = None
            self.data = None
            self.cardinality = np.array(data.shape, dtype=int)
            self.pmf = data / data.sum()

        assert (
            (datatype == "individual" and len(ordinality) == data.shape[1])
            or (datatype == "distribution" and len(ordinality) == data.ndim)
        ), "ordinality dimension not matching the data dimension"
        for i, ord in enumerate(ordinality):
                assert len(ord) == self.cardinality[i], (
                f"ordinality information not matching the cardinality in dimension {i}"
            )
        self.ordinality: List[np.ndarray] = []
        for ord in ordinality:
            ord = np.array(ord, dtype=float)
            self.ordinality.append(ord - ord.min())
    
    def __repr__(self):
        return (f"Distribution(qkey={self.qkey}\n"
                + f"datatype={self.datatype}\n"
                + f"dimension={self.cardinality}\n"
                + f"ordinality={self.ordinality}\n"
                + f"pmf=\n{self.pmf})"
        )

    def correlation(self,
                    dim1: Optional[int] = None,
                    dim2: Optional[int] = None,
                    use_weights: bool = True,
                    **kwargs
                    ) -> Union[np.ndarray, float]:
        assert self.data is not None, "Distribution is initialized with pre-calculated dist."
        assert (dim1 is None) == (dim2 is None), "dim 1 & 2 must be both provided or None"
        if dim1 is not None:
            assert dim1 < self.n_items, "dim1 must be less than n_items"
            assert dim2 < self.n_items, "dim2 must be less than n_items"
        corr = _correlation(self.data,
                            self.ordinality,
                            self.weights if use_weights else None,
                            )
        if dim1 is not None:
            return corr[dim1, dim2]
        return corr

    def entropy(self,
                base: float = 2.0,
                normalize: bool = True,
                **kwargs,
                ) -> float:
        """
        Entropy of the distribution (self).
        ----------
        [Parameters]
        base : float. Base of the logarithm.  Default: 2.0
        normalize: bool. If True, normalize by the maximum entropy.
        -------
        [Returns]
        float, Shannon entropy of the distribution.
        """
        assert self.pmf is not None, "Distribution is initialized with pre-calculated dist."
        entropy = -np.sum(self.pmf * np.log(self.pmf + 1e-10)) / np.log(base)
        if normalize:
            entropy /= (np.log(self.cardinality.prod()) / np.log(base))
        return entropy

    def emd(self, 
            other: "Distribution",
            metric: Union[str, Callable] = "euclidean",
            sinkhorn_eps: Optional[float] = None,
            return_log: bool = False,
            jitter: float = 1e-10,
            **kwargs,
            ) -> Union[float, Tuple[float, dict]]:
        """
        Earth-Mover distance to another Distribution.
        ----------
        [Parameters]
        other : Distribution
            Distribution to compare against.  Must share the same grid
            (i.e. identical `cardinality` / PMF shape).
        metric, sinkhorn_eps, return_log, jitter
            Forwarded to `_wasserstein_distance_nd`.
        -------
        [Returns]
        float or tuple
            Distance, and—when `sinkhorn_eps` is not None and
            `return_log` is True—the POT log dictionary as well.
        """
        assert isinstance(other, Distribution), "other must be a Distribution instance"
        assert tuple(self.cardinality) == tuple(other.cardinality), (
            f"Distributions live on different supports: "
            f"{tuple(self.cardinality)} vs {tuple(other.cardinality)}"
        )
        assert (
            len(self.ordinality) == len(other.ordinality)
            and all(
                np.array_equal(x,y)
                for x,y in zip(self.ordinality, other.ordinality)
            )
        ), "Distributions have different ordinality"
        return _wasserstein_distance_nd(
            self.pmf,
            other.pmf,
            ordinality=self.ordinality,
            metric=metric,
            sinkhorn_eps=sinkhorn_eps,
            return_log=return_log,
            jitter=jitter,
        )

    @staticmethod
    def bootstrap_stats(
        stat_func: Callable,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_boot: int = 1000,
        ci: float = 0.95,
        random_state: Optional[np.random.Generator] = None,
        **stat_kwargs,
    ) -> Tuple[float, float]:
        """
        Generic non-parametric bootstrap for a statistic that can accept
        ``(data, weights, **kwargs)`` as its first two arguments.
        -------
        [Returns]
        (low, high) : Tuple[float, float]
            Two-sided (1-alpha)·100 % percentile interval, where alpha = 1-ci.
        """

        if stat_func == "entropy":
            def stat_func(d, w=None, **kw):
                return Distribution(
                    data=d, weights=w, datatype="individual"
                ).entropy(**kw)
        else:
            raise NotImplementedError()

        assert 0.0 < ci < 1.0, "Confidence level must be in (0, 1)"
        if weights is not None:
            assert len(weights) == len(data), "weights and data length mismatch"
            sampling_prob = np.array(weights, dtype=float, copy=True)
            sampling_prob /= sampling_prob.sum()
        else:
            sampling_prob = None
        rng = (
            random_state if (
                random_state is not None
                and isinstance(random_state, np.random.Generator)
            ) else np.random.default_rng()
        )
        estimates = np.empty(n_boot, dtype=float)
        n = len(data)

        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True, p=sampling_prob)
            sample_data = data[idx]
            if weights is not None:
                sample_weights = weights[idx]
                estimates[b] = stat_func(sample_data, sample_weights, **stat_kwargs)
            else:
                estimates[b] = stat_func(sample_data, **stat_kwargs)

        alpha = 0.5 * (1.0 - ci)
        low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
        return float(low), float(high)
