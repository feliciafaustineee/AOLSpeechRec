import numpy as np


def _dtw_py(a, b):
    """Jarak DTW antara dua urutan frame (T x D). Loop Python dengan float biasa agar cepat."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).tolist()
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        crow = cost[i - 1]
        for j in range(1, m + 1):
            best = prev[j]
            if prev[j - 1] < best:
                best = prev[j - 1]
            if cur[j - 1] < best:
                best = cur[j - 1]
            cur[j] = crow[j - 1] + best
        prev = cur
    return prev[m] / (n + m)


try:
    from numba import njit

    @njit(cache=True)
    def _dtw_nb(a, b):
        n, m = a.shape[0], b.shape[0]
        prev = np.full(m + 1, np.inf); prev[0] = 0.0
        for i in range(1, n + 1):
            cur = np.full(m + 1, np.inf)
            for j in range(1, m + 1):
                c = 0.0
                for d in range(a.shape[1]):
                    t = a[i-1, d] - b[j-1, d]
                    c += t * t
                c = np.sqrt(c)
                best = prev[j]
                if prev[j-1] < best: best = prev[j-1]
                if cur[j-1] < best: best = cur[j-1]
                cur[j] = c + best
            prev = cur
        return prev[m] / (n + m)

    def dtw_distance(a, b):
        return _dtw_nb(np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64))
except ImportError:
    dtw_distance = _dtw_py


def predict(train_feats, train_labels, test_feat):
    dists = [dtw_distance(test_feat, f) for f in train_feats]
    return train_labels[int(np.argmin(dists))], min(dists)


def wer(ref, hyp):
    """Word Error Rate: edit distance kata / jumlah kata referensi."""
    r, h = ref.lower().replace("?", "").split(), hyp.lower().replace("?", "").split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            sub = 0 if r[i-1] == h[j-1] else 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + sub)
    return d[len(r), len(h)] / max(1, len(r))
