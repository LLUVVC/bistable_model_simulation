"""Exact stationary distribution of the full well-mixed model (X, X2) from the 2D CME.

Reactions (counts; A = a*V, B = b*V are buffered):
  2X -> X2        l1p/V * X(X-1)
  X2 -> 2X        l1m * X2
  X2 + A -> X2+X  l2p * a * X2
  X2 + X -> X2+A  l2m/V * X2 * X
  B -> X          l3p * b * V
  X -> B          l3m * X
Truncated state space X in [0, Xmax], X2 in [0, Ymax]; moves leaving it are dropped.
Check that the returned P is ~0 at the boundary, otherwise enlarge Xmax / Ymax.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def solve_cme_2d(l, a, b, V, Xmax, Ymax):
    """
    Xmax= M_max (X), Ymax=N_max (X_2)
    
    The matrix is of the form idx-(target, source)
    """
    l1p, l1m, l2p, l2m, l3p, l3m = l
    nx, ny = Xmax + 1, Ymax + 1 
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    idx = lambda x, y: x * ny + y # idx corresponds to s.
    rows, cols, vals = [], [], []

    def add(rate, dx, dy):
        x2, y2 = X + dx, Y + dy ### new state
        ### find the nonzero entry (rate>0) that is valid (inside the domain)
        ok = (rate > 0) & (x2 >= 0) & (x2 <= Xmax) & (y2 >= 0) & (y2 <= Ymax)
        ### i: current state; j: the neighbouring target state; rate: rate values in nonzero entries
        ### i and j are flattend states ids (s)
        i, j, r = idx(X[ok], Y[ok]), idx(x2[ok], y2[ok]), rate[ok]
        ### from the column i {source, the cols.extend([i,_])} to row j {target, the rows.extend([j,_])},
        ### the rate value is +r: flowing from source to target
        ### from the column i {source, the cols.extend([_,i])} to row i {target, the rows.extend([_,i])},
        ### the rate value is -r: flowing out from the target
        rows.extend([j, i]); cols.extend([i, i]); vals.extend([r, -r])   # transposed generator Q^T

    add(l1p / V * X * (X - 1.0), -2, 1)
    add(l1m * Y * 1.0, 2, -1)
    add(l2p * a * Y * 1.0, 1, 0)
    add(l2m / V * Y * X * 1.0, -1, 0)
    add(np.full(X.shape, l3p * b * V), 1, 0)
    add(l3m * X * 1.0, -1, 0)

    n = nx * ny
    QT = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tolil()
    QT[0, :] = 1.0                      # replace one equation by normalization sum(P) = 1
    rhs = np.zeros(n); rhs[0] = 1.0
    P = spl.spsolve(QT.tocsc(), rhs).reshape(nx, ny)
    return P.sum(axis=1), P             # marginal P(X = x), joint P(X, X2)


if __name__ == "__main__":
    a, b, V = 10.0, 20.0, 8.0
    l = (37.5, 1500, 6, 1, 5.75, 25)    # K_eq = 0.025
    PX, P = solve_cme_2d(l, a, b, V, 420, 400)
    print("P(X<150) =", PX[:150].sum(), " boundary mass:", P[-1, :].sum(), P[:, -1].sum())
