"""
Which eigenvectors the example-image tiles of Figs. 3E-F and 4B-C show.

The tiles are selected, not drawn in rank order, so that the same eigenvectors
can be compared across encoders. For each metric and area:

  1. Pool: ten eigenvectors per encoder.
  2. Pairs: all Inception/ResNet-50 pairs in the pool are ordered by |cosine|,
     best first, and taken greedily so that each eigenvector is used once.
  3. Fig. 3 draws the Inception eigenvectors of the first four pairs. Each is
     signed so that its largest-magnitude pixel is positive.
  4. Fig. 4 uses the first three pairs. The Inception row is as in Fig. 3, and
     the ResNet-50 row shows the partners, signed to agree with the Inception
     eigenvector. The ON-OFF row shows, for each Inception eigenvector in turn,
     the unused ON-OFF eigenvector among ranks 1-10 with the largest |cosine|,
     signed in the same way.

Ranks in the returned records are 1-based, as in the paper.
"""
from __future__ import annotations

import numpy as np

POOL_SIZE = 10
FIRST_RANK = {"G": 1, "J": 1}          # 1-based first rank of the pool
ONOFF_POOL_SIZE = 10
N_TILES_FIG3 = 4
N_TILES_FIG4 = 3


def unit_rows(vectors) -> np.ndarray:
    """[k, P] eigenvectors as a float array with unit-norm rows."""
    v = vectors.detach().cpu().numpy() if hasattr(vectors, "detach") else np.asarray(vectors)
    v = v.astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def positive_peak(v: np.ndarray) -> np.ndarray:
    """Sign `v` so that its largest-magnitude pixel is positive."""
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def signed_towards(v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Sign `v` so that its cosine with `reference` is positive."""
    return v * np.sign(v @ reference)


def ranked_pairs(inception: np.ndarray, resnet: np.ndarray, pool: list[int]) -> list[tuple]:
    """Greedy unique pairs (index in inception, index in resnet, |cosine|), best first."""
    overlap = np.abs(inception[pool] @ resnet[pool].T)
    used_i, used_j, pairs = set(), set(), []
    for flat in np.argsort(-overlap, axis=None):
        a, b = np.unravel_index(flat, overlap.shape)
        if a in used_i or b in used_j:
            continue
        used_i.add(a)
        used_j.add(b)
        pairs.append((pool[a], pool[b], float(overlap[a, b])))
    return pairs


def select_pairs(inception: np.ndarray, resnet: np.ndarray, metric: str) -> list[tuple]:
    """The ranked Inception/ResNet-50 pairs of one metric ("G" or "J") and area."""
    first = FIRST_RANK[metric] - 1
    last = min(first + POOL_SIZE, inception.shape[0], resnet.shape[0])
    return ranked_pairs(inception, resnet, list(range(first, last)))


def fig3_tiles(inception_vectors, resnet_vectors, metric: str, n: int = N_TILES_FIG3):
    """Inception tiles of Fig. 3E-F: ([n, P] vectors, records with 1-based ranks)."""
    inc, res = unit_rows(inception_vectors), unit_rows(resnet_vectors)
    pairs = select_pairs(inc, res, metric)[:n]
    tiles = np.stack([positive_peak(inc[i]) for i, _, _ in pairs])
    records = [{"inception": i + 1, "resnet": j + 1, "abs_cos": round(c, 3)} for i, j, c in pairs]
    return tiles, records


def fig4_tiles(inception_vectors, resnet_vectors, onoff_vectors, metric: str,
               n: int = N_TILES_FIG4):
    """
    Tiles of Fig. 4B-C: ({encoder: [n, P] vectors}, records with 1-based ranks).
    `onoff_vectors` may be None, which leaves the ON-OFF row out.
    """
    inc, res = unit_rows(inception_vectors), unit_rows(resnet_vectors)
    onoff = None if onoff_vectors is None else unit_rows(onoff_vectors)
    tiles = {"inception": [], "resnet50": [], "onoff": []}
    records, used = [], set()
    for i, j, c in select_pairs(inc, res, metric)[:n]:
        reference = positive_peak(inc[i])
        tiles["inception"].append(reference)
        tiles["resnet50"].append(signed_towards(res[j], reference))
        record = {"inception": i + 1, "resnet": j + 1, "inc_res_abs_cos": round(c, 3)}
        if onoff is not None:
            pool = [o for o in range(min(ONOFF_POOL_SIZE, onoff.shape[0])) if o not in used]
            o = max(pool, key=lambda q: abs(onoff[q] @ reference))
            used.add(o)
            tiles["onoff"].append(signed_towards(onoff[o], reference))
            record.update(onoff=o + 1, inc_onoff_abs_cos=round(float(abs(onoff[o] @ reference)), 3))
        records.append(record)
    return {e: (np.stack(t) if t else None) for e, t in tiles.items()}, records
