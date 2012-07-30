import numpy as np
from numpy.linalg import qr
import scipy.linalg

class IncrementalQR(object):
    """Perform QR calculations on a matrix too large to fit in memory.

    Given a matrix X with x_cols columns, you feed in successive "strips" of
    X. At any time you can call .r() to get the R part of a QR decomposition
    of X.

    If y_cols is given, you also feed in strips of a matrix Y, and at any time
    you can call .qty() to get Q'Y, where Q is from the QR decomposition.

    Q itself is not calculated or stored.
    """
    # _calc_q is for use of the test suite only, don't use it:
    def __init__(self, x_cols, y_cols=None, _calc_q=False):
        self._x_cols = x_cols
        if y_cols is None:
            y_cols = 0
        self._y_cols = y_cols
        self._r = np.empty((0, self._x_cols + self._y_cols))
        if _calc_q:
            self._q = np.empty((0, x_cols))
        else:
            self._q = None

    def append(self, x, y=None):
        assert x.ndim == 2 and x.shape[1] == self._x_cols
        # Combining the x's and y's into a single matrix lets us entirely skip
        # forming the Q matrix explicitly, and that's a huge saving, more than
        # offsetting the cost of this copy. The trick is that if
        #   QR = [x y]
        # Then
        #   R = Q'[x y]
        # i.e., the last column of R is the same as Q'y. So we really just
        # compute the QR decomposition on [x y] and then split up the result
        # in the accessor functions.
        #
        # For even more efficiency, the caller could allocate x and y in a
        # single matrix and unpack again themselves, but this copy doesn't
        # seem to be the bottleneck in practice.
        if y is not None:
            if y.ndim == 1:
                y = y.reshape((-1, 1))
            assert y.shape[1] == self._y_cols
            x = np.hstack((x, y))
        if self._q is not None:
            q_x, r_x = qr(x)
        else:
            r_x = qr(x, mode="r")
        r_stacked = np.vstack((self._r, r_x))
        del r_x
        if self._q is not None:
            q_tmp, r_new = qr(r_stacked)
        else:
            r_new = qr(r_stacked, mode="r")
        del r_stacked
        self._r = r_new
        if self._q is not None:
            if self._q.size == 0:
                self._q = q_x
            else:
                # Form (_q   0)
                #      ( 0 q_x) * q_tmp
                qs = scipy.linalg.block_diag(self._q, q_x)
                self._q = np.dot(qs, q_tmp)

    def full_r(self):
        "Returns the R part of a QR decomposition of [X Y]."
        return self._r

    def r(self):
        "Returns the R part of a QR decomposition of X."
        if self._y_cols:
            return self._r[:self._x_cols, :self._x_cols]
        else:
            return self._r

    def qty(self):
        "Returns the value of Q'Y for X = QR."
        return self._r[:self._x_cols, self._x_cols:]

def test_incremental_qr():
    x = np.arange(1000 * 10).reshape(1000, 10)
    y = np.arange(1000 * 2).reshape(1000, 2)
    # One shot:
    inc = IncrementalQR(10, 2, _calc_q=True)
    inc.append(x, y)
    assert np.allclose(np.dot(inc._q.T, inc._q),
                       np.eye(inc._q.shape[1]))
    assert np.allclose(np.dot(inc._q, inc.full_r()), np.hstack([x, y]))
    assert np.allclose(inc.qty(), np.dot(inc._q.T[:10, :], y))
    # No y:
    inc = IncrementalQR(10, _calc_q=True)
    inc.append(x)
    assert np.allclose(np.dot(inc._q, inc.full_r()), x)
    assert np.allclose(np.dot(inc._q, inc.r()), x)
    # In 10 pieces:
    inc = IncrementalQR(10, 2, _calc_q=True)
    for i in xrange(10):
        s = slice(100 * i, 100 * i + 100)
        inc.append(x[s, :], y[s, :])
    assert np.allclose(np.dot(inc._q.T, inc._q),
                       np.eye(inc._q.shape[1]))
    assert np.allclose(np.dot(inc._q, inc.full_r()), np.hstack([x, y]))
    assert np.allclose(inc.qty(), np.dot(inc._q.T[:10, :], y))
    # No y:
    inc = IncrementalQR(10, _calc_q=True)
    for i in xrange(10):
        s = slice(100 * i, 100 * i + 100)
        inc.append(x[s, :])
    assert np.allclose(np.dot(inc._q.T, inc._q),
                       np.eye(inc._q.shape[1]))
    assert np.allclose(np.dot(inc._q, inc.full_r()), x)
    assert np.allclose(np.dot(inc._q, inc.r()), x)
    # With 1d y:
    inc = IncrementalQR(10, 1, _calc_q=True)
    for i in xrange(10):
        s = slice(100 * i, 100 * i + 100)
        inc.append(x[s, :], y[s, 0])
    y_col = y[:, 0].reshape(-1, 1)
    assert np.allclose(np.dot(inc._q.T, inc._q),
                       np.eye(inc._q.shape[1]))
    assert np.allclose(np.dot(inc._q, inc.full_r()), np.hstack([x, y_col]))
    assert np.allclose(inc.qty(), np.dot(inc._q.T[:10, :], y_col))
    
if __name__ == "__main__":
    import nose
    nose.runmodule()
