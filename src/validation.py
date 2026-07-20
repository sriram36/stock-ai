"""
Combinatorial Purged Cross-Validation wrapper (skfolio), adapted for
triple-barrier labels.

Save this file as: src/validation.py

Note on purging: skfolio's CombinatorialPurgedCV takes purged_size and
embargo_size as fixed observation counts, not label-end-time (t1) aware
purging based on each label's actual overlap (the full method described
in Lopez de Prado's book). Since every triple-barrier label from
labeling.py has a bounded duration <= vertical_days, setting
purged_size = vertical_days is a safe, documented approximation: it
purges AT LEAST as much as true t1-aware purging would in the worst
case, which is the conservative direction to be wrong in. If you ever
cite CPCV results from this pipeline, note this assumption -- it is not
the exact textbook implementation, it's a defensible approximation of it.

Install: pip install skfolio
"""

import numpy as np
from math import comb
from skfolio.model_selection import CombinatorialPurgedCV


def build_cpcv(
    vertical_days: int, n_folds: int = 10, n_test_folds: int = 8
) -> CombinatorialPurgedCV:
    """
    Construct a CombinatorialPurgedCV splitter sized for triple-barrier
    labels with the given max holding period.

    Parameters
    ----------
    vertical_days : int
        MUST match the vertical_days used in labeling.py's
        triple_barrier_labels() call that produced your labels. This
        sets both the purge and embargo window -- see module docstring.
    n_folds, n_test_folds : int
        Defaults (10, 8) match the ~45-backtest-path setup used in
        published CPCV research. Reduce for smaller datasets -- path
        count grows combinatorially, check with n_backtest_paths() below
        before committing to large values on limited data.
    """
    return CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=vertical_days,
        embargo_size=vertical_days,
    )


def n_backtest_paths(n_folds: int, n_test_folds: int) -> int:
    """
    Number of distinct backtest paths CPCV will generate. Check this
    before committing to a large n_folds -- e.g. (10, 8) already gives
    45 paths, each of which needs a full train+evaluate cycle.
    """
    return comb(n_folds, n_test_folds)


if __name__ == "__main__":
    cv = build_cpcv(vertical_days=10, n_folds=6, n_test_folds=2)
    X = np.random.randn(300, 5)

    n_splits = 0
    for train_idx, test_idx_groups in cv.split(X):
        # Note: skfolio yields test_idx as a LIST of arrays (one per test
        # fold in that combinatorial path), not a single flat array --
        # unlike standard sklearn CV. Handle accordingly when fitting.
        n_splits += 1
        if n_splits == 1:
            print(
                f"First split: train size={len(train_idx)}, "
                f"test groups in this path={len(test_idx_groups)}, "
                f"each test group size~={len(test_idx_groups[0])}"
            )

    print(f"\nTotal train/test splits generated: {n_splits}")
    print(f"Expected combinatorial backtest paths: {n_backtest_paths(6, 2)}")