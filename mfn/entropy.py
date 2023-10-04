from mfn.bootstrap import lbb
from ordpy import fisher_shannon

def _miee(perm_entropy, fisher_info) -> float:
    miee = 1 - (perm_entropy - fisher_info)
    return miee


def miee(data, dx=3, taux=1, probs=False, tie_precision=None) -> float:
    """
    Calculates the Macroeconophysics indicator of economic efficiency (MIEE)\\ [#fernandes2021]_ .
    This consists of a specific linear combination of permutation entropy and Fisher information.
    It was shown empirically in \\ [#martins2023]_ that it comprises information from the 
    Fisher-Shannon Causality plane in an efficient manner.
    It should be used for univariate time series only.

    Parameters
    ----------
    data : array
              Array object in the format :math:`[x_{1}, x_{2}, x_{3}, \\ldots ,x_{n}]`
                or  :math:`[[x_{11}, x_{12}, x_{13}, \\ldots, x_{1m}],
                \\ldots, [x_{n1}, x_{n2}, x_{n3}, \\ldots, x_{nm}]]`.
    dx : int
            Embedding dimension (horizontal axis) (default: 3).
    taux : int
            Embedding delay (horizontal axis) (default: 1).
    probs : boolean
            If `True`, it assumes **data** is an ordinal probability distribution.
            If `False`, **data** is expected to be a one- or two-dimensional array
            (default: `False`).
    tie_precision : None, int
                    If not `None`, **data** is rounded with `tie_precision`
                    decimal numbers (default: `None`).

    Returns
    -------
        : float
            The value of MIEE.

    Examples
    --------
    """
    perm_entropy, fisher_info = fisher_shannon(
        data,
        dx=dx, dy=1,
        taux=taux, tauy=1,
        probs=probs, tie_precision=tie_precision
    )
    miee = _miee(perm_entropy, fisher_info)
    return miee


def MFN(
    data,
    b:int, B:float, size:int,
    dx=3, taux=1, probs=False, tie_precision=None,
    metric_to_return=["permutation entropy", "fisher information", "miee"],
) -> dict:
    """
    Generates the results from the desired metrics using the MFN method \\ [#martins2023]_ .
    This method permits to explore the empirical distribution instead of single point estimates
    of the desired metrics via bootstrap methods applied to the original data.
    It should only be used for univariate time series.

    Parameters
    ----------
    data : array
              Array object in the format :math:`[x_{1}, x_{2}, x_{3}, \\ldots ,x_{n}]`
                or  :math:`[[x_{11}, x_{12}, x_{13}, \\ldots, x_{1m}],
                \\ldots, [x_{n1}, x_{n2}, x_{n3}, \\ldots, x_{nm}]]`.
    dx : int
            Embedding dimension (horizontal axis) (default: 3).
    taux : int
            Embedding delay (horizontal axis) (default: 1).
    probs : boolean
            If `True`, it assumes **data** is an ordinal probability distribution.
            If `False`, **data** is expected to be a one- or two-dimensional array
            (default: `False`).
    tie_precision : None, int
                    If not `None`, **data** is rounded with `tie_precision`
                    decimal numbers (default: `None`).
    b : int
        block size for the bootstrap method. Must be between 0 and `len(series)`.
    B : float
        size of the neighborhood for the bootstrap method. Must be between 0 and 1.
    size : int
        how many bootstrap replicas you want.
    metric_to_return : list
        list of metrics to return. Must be one or more of the following:
        "permutation entropy", "fisher information", "miee".

    Returns
    -------
        : dict
            Dictionary containing the results for each metric. The keys are the
            names of the metrics and the values are lists of size `size` containing
            the results for each bootstrap replica.

    Examples
    --------

    """
    # sanity checks
    if b < 0 or b > len(data):
        raise ValueError("b must be between 0 and len(series)")
    if B < 0 or B > 1:
        raise ValueError("B must be between 0 and 1")
    if not isinstance(metric_to_return, list):
        raise TypeError("metric_to_return must be a list")
    if len(metric_to_return) == 0:
        raise ValueError("metric_to_return must have at least one element")
    for metric in metric_to_return:
        if metric not in ["permutation entropy", "fisher information", "miee"]:
            raise ValueError("metric_to_return must be one or more of the following: " +
                             "'permutation entropy', 'fisher information', 'miee'")


    lbb_samples = lbb(
        series=data,
        b=b,
        B=B,
        size=size
    )

    results = {}
    for metric in metric_to_return:
        results[metric] = []

    for lbb_sample in lbb_samples:
        perm_entropy, fisher_info = fisher_shannon(
            lbb_sample,
            dx=dx, taux=taux,
            probs=probs, tie_precision=tie_precision
        )

        if "miee" in metric_to_return:
            miee_value = _miee(perm_entropy, fisher_info)
        
        for metric in metric_to_return:
            if metric == "permutation entropy":
                value = perm_entropy
            elif metric == "fisher information":
                value = fisher_info
            elif metric == "miee":
                value = miee_value
            results[metric].append(value)
    
    return results