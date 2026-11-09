"""
The tile selection rule of Figs. 3E-F and 4B-C (figures/_eigvec_selection.py),
on eigenvectors built so that the expected choice follows from the rule alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from figures._eigvec_selection import (
    fig3_tiles, fig4_tiles, positive_peak, ranked_pairs, select_pairs, signed_towards,
)

P = 64


def _basis(n: int = P) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))
    return q.T


def _mix(basis: np.ndarray, a: int, b: int, cos: float) -> np.ndarray:
    return cos * basis[a] + np.sqrt(1 - cos ** 2) * basis[b]


def test_greedy_pairs_use_each_eigenvector_once_best_first():
    b = _basis()
    inc = b[:3]
    res = np.stack([_mix(b, 0, 10, 0.9), _mix(b, 0, 1, 0.8), _mix(b, 2, 11, 0.5)])
    pairs = ranked_pairs(inc, res, [0, 1, 2])
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1), (2, 2)]
    assert [c for _, _, c in pairs] == sorted((c for _, _, c in pairs), reverse=True)


@pytest.mark.parametrize("metric, first", [("G", 2), ("J", 1)])
def test_pool_starts_at_rank_2_for_G_and_rank_1_for_J(metric, first):
    b = _basis()
    inc = b[:12]
    res = np.stack([_mix(b, k, 40 + k, 0.99 - 0.05 * k) for k in range(12)])
    ranks = [i + 1 for i, _, _ in select_pairs(inc, res, metric)]
    assert min(ranks) == first and max(ranks) == first + 9 and len(ranks) == 10
    assert ranks[0] == first                     # the best-aligned pair of the pool comes first


def test_fig3_tiles_are_inception_vectors_with_positive_peak():
    b = _basis()
    inc = -b[:12]
    res = b[:12]
    tiles, records = fig3_tiles(inc, res, "G", n=4)
    assert len(records) == 4 and all(r["abs_cos"] == pytest.approx(1.0) for r in records)
    for tile, rec in zip(tiles, records):
        assert rec["inception"] == rec["resnet"] and rec["inception"] >= 2
        np.testing.assert_allclose(tile, positive_peak(inc[rec["inception"] - 1]))
        assert tile[np.argmax(np.abs(tile))] > 0


def test_fig4_partners_agree_in_sign_and_onoff_is_used_once():
    b = _basis()
    inc = b[:12]
    res = -b[:12]
    onoff = np.stack([_mix(b, (k + 1) % 12, 40 + k, 0.7) for k in range(12)])
    tiles, records = fig4_tiles(inc, res, onoff, "J", n=3)
    assert len({r["onoff"] for r in records}) == 3
    for k in range(3):
        assert tiles["inception"][k] @ tiles["resnet50"][k] > 0
        assert tiles["inception"][k] @ tiles["onoff"][k] > 0
        assert records[k]["onoff"] <= 10


def test_fig4_without_onoff_leaves_that_row_out():
    b = _basis()
    tiles, records = fig4_tiles(b[:12], b[:12], None, "G", n=3)
    assert tiles["onoff"] is None and "onoff" not in records[0]


def test_signed_towards():
    v = np.array([1.0, 0.0])
    np.testing.assert_allclose(signed_towards(v, np.array([-2.0, 1.0])), -v)
