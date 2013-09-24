# This file is part of pyrerp
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# How to compute incremental std dev:
#   http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html

import numpy as np
from numpy.linalg import solve, inv
from scipy import stats, sparse

# These are copied with trivial syntactic modification from R: see
# stats:::Pillai, stats:::Wilks, etc. They each return a 4-tuple
#   (raw test value, approximate F, df1, df2)
# NB: this means that they are GPLed and I don't have the power to change
# that except by rewriting them from scratch.
_mv_tests = {}

def _pillai(eig, q, df_res):
    test = np.sum(eig * 1. / (1 + eig))
    p = len(eig)
    s = min(p, q)
    n = 0.5 * (df_res - p - 1)
    m = 0.5 * (abs(p - q) - 1)
    tmp1 = 2 * m + s + 1
    tmp2 = 2 * n + s + 1
    return (test, (tmp2 * 1. / tmp1 * test)/(s - test), s * tmp1, s * tmp2)
_mv_tests["Pillai"] = _pillai

# Test insensitivity to eigenvalue order:
_mv_test_vecs = [(np.array([10, 0.3]), 2, 8),
                 (np.array([0.3, 10]), 2, 8)]

def test__pillai():
    for tv in _mv_test_vecs:
        assert np.allclose(_pillai(*tv),
                           [1.13986013986014, 5.30081300813008, 4, 16])

def _wilks(eig, q, df_res):
    test = np.prod(1./(1 + eig))
    p = len(eig)
    tmp1 = df_res - 0.5 * (p - q + 1)
    tmp2 = (p * q - 2) * 1. / 4
    tmp3 = p ** 2 + q ** 2 - 5
    if tmp3 > 0:
        tmp3 = np.sqrt(((p * q) ** 2 - 4) * 1. /tmp3)
    else:
        tmp3 = 1
    return (test, ((test ** (-1./tmp3) - 1) * (tmp1 * tmp3 - 2 * tmp2)) * 1./p/q, 
            p * q, tmp1 * tmp3 - 2 * tmp2)
_mv_tests["Wilks"] = _wilks
    
def test__wilks():
    for tv in _mv_test_vecs:
        assert np.allclose(_wilks(*tv),
                           [0.0699300699300699, 9.7353692808323267, 4, 14])

def _hl(eig, q, df_res):
    test = np.sum(eig)
    p = len(eig)
    m = 0.5 * (abs(p - q) - 1)
    n = 0.5 * (df_res - p - 1)
    s = min(p, q)
    tmp1 = 2 * m + s + 1
    tmp2 = 2 * (s * n + 1)
    return (test, (tmp2 * test) * 1./s/s/tmp1, s * tmp1, tmp2)
_mv_tests["Hotelling-Lawley"] = _hl

def test__hl():
    for tv in _mv_test_vecs:
        assert np.allclose(_hl(*tv),
                           [10.30, 15.45, 4.00, 12.00])
def _roy(eig, q, df_res):
    p = len(eig)
    test = np.max(eig)
    tmp1 = max(p, q)
    tmp2 = df_res - tmp1 + q
    return (test, (tmp2 * test) * 1. /tmp1, tmp1, tmp2)
_mv_tests["Roy"] = _roy

def test__roy():
    for tv in _mv_test_vecs:
        assert np.allclose(_roy(*tv),
                           [10, 40, 2, 8])

class LSResult(object):
    def __init__(self, coef, vcov, rssp, rdf):
        self._coef = coef
        self._vcov = vcov
        self._rssp = rssp
        self._rdf = rdf

    def coef(self):
        """Returns a matrix of coefficients. Each column is the coefficients
        for one column of the outcome vector Y."""
        return self._coef

    def vcov(self):
        return self._vcov

    def rss(self):
        """Returns the residual sum-of-squares vector.

        Each entry is the residual sum of squares for the corresponding column
        of y.

        If this is the result of a weighted least squares fit, then it is
        the weighted sum of the residual squares."""
        return self._rssp.diagonal()

    def rssp(self):
        """Returns the residual sum-of-squares-and-products matrix.

        Each entry is the residual sum of products for the corresponding
        columns of y. The diagonal contains the residual sum of squares.

        If this is the result of a weighted least squares fit, then it is the
        weighted sum of the residual squares and products."""
        return self._rssp

    def rdf(self):
        "Returns the residual degrees of freedom."
        return self._rdf

    def scaled_vcov(self):
        """Returns the scaled variance-covariance matrix.

        This is a 3-dimensional array of shape (N, N, D), where D is the
        number of columns in y, and for each such column there is an NxN
        matrix of estimated variance-covariances of the coefficients fit to
        that column."""
        return (self.rss()[np.newaxis, np.newaxis, :]
                * 1. / self._rdf * self._vcov[:, :, np.newaxis])

    def se(self):
        """Returns the standard errors of the coefficient estimates.

        This is a matrix of the same shape as .coef()."""
        return np.sqrt(self.scaled_vcov().diagonal().T)

    def t_tests(self):
        """For each coefficient, performs a t-test where the null is that that
        coefficient has the true value of 0.

        Returns a tuple (t, p).

        Each is a matrix of the same shape as .coef(). 't' contains t values
        for each coefficient, and 'p' contains corresponding two-tailed
        p-values for the t-test with .rdf() degrees of freedom."""
        se = self.se()
        t = self._coef / se
        p = 2 * stats.distributions.t.sf(np.abs(t), self._rdf)
        return (t, p)

    def lht_by_dim(self, hypothesis_matrix, rhs=None):
        hypothesis_matrix = np.atleast_2d(hypothesis_matrix)
        y_dim = self._coef.shape[1]
        q = hypothesis_matrix.shape[0]
        if rhs is None:
            rhs = np.zeros((q, y_dim))
        # If a 1d vector is given, assume it was meant as a column:
        rhs = np.atleast_1d(rhs)
        if rhs.ndim == 1:
            rhs = rhs.reshape((-1, 1))
        trans_coef = np.dot(hypothesis_matrix, self._coef) - rhs
        vcov = self.scaled_vcov()
        F = np.empty(y_dim)
        for i in xrange(y_dim):
            ssh = np.dot(trans_coef[:, [i]].T,
                         solve(np.dot(np.dot(hypothesis_matrix,
                                             vcov[:, :, i]),
                                      hypothesis_matrix.T),
                               trans_coef[:, [i]]))
            F[i] = ssh * 1. / q
        p = stats.distributions.f.sf(F, q, self.rdf())
        return F, q, self.rdf(), p

    def lht_multivariate(self, hypothesis_matrix, rhs=None, subset=None):
        # Returns a dict with one entry per standard multivariate test, and
        # each entry is a tuple of length 5:
        #   (raw statistic, approximate F, df1, df2, p)
        # Optionally, this can be done on just a subset of the exogenous
        # variable dimensions. 'subset' can be any way of indexing some subset
        # of the dimensions; setting it to an integer produces a
        # unidimensional lht.
        hypothesis_matrix = np.atleast_2d(hypothesis_matrix)
        if subset is None:
            subset = slice(None)
        subset = np.atleast_1d(np.arange(self._rssp.shape[0])[subset])
        y_dim = len(subset)
        q = hypothesis_matrix.shape[0]
        if rhs is None:
            rhs = np.zeros((q, y_dim))
        # If a 1d vector is given, assume it was meant as a column:
        rhs = np.atleast_1d(rhs)
        if rhs.ndim == 1:
            rhs = rhs.reshape((-1, 1))
        # SSPH <- t(L %*% B - rhs) %*% solve(L %*% V %*% t(L)) %*% (L %*% B - rhs)        
        # where L is the hypothesis matrix, B is the coefs, rhs is the null
        # values for L*B, and V is the unscaled variance-covariance matrix.
        trans_coef = np.dot(hypothesis_matrix, self._coef[:, subset]) - rhs
        ssph = np.dot(trans_coef.T,
                      solve(np.dot(np.dot(hypothesis_matrix, self._vcov),
                                   hypothesis_matrix.T),
                            trans_coef))
        sspe = self._rssp[subset.reshape((-1, 1)), subset.reshape((1, -1))]
        eigs = np.linalg.eigvals(np.linalg.lstsq(sspe, ssph)[0]).real
        results = {}
        for name, fn in _mv_tests.iteritems():
            (stat, F, df1, df2) = fn(eigs, q, self._rdf)
            p = stats.distributions.f.sf(F, df1, df2)
            results[name] = (stat, F, df1, df2, p)
        return results

class QRIncrementalLS(object):
    """Perform least-squares regression with very large model matrices.
    
    Supports arbitrary numbers of predictors, and for a given model matrix can
    simultaneously solve multiple regression problems with differing outcome
    vectors.

    This uses .incremental_qr.IncrementalQR to form the QR
    decomposition. It is slower than XtXIncrementalLS, and cannot accept
    sparse matrices, but it may be more numerically stable."""

    def __init__(self):
        self._qr = None
        self._x_cols = None
        self._y_cols = None
        self._y_ssp = None
        self._x_rows = 0

    def append(self, x_strip, y_strip):
        from .incremental_qr import IncrementalQR
        assert x_strip.ndim == y_strip.ndim == 2
        if self._qr is None:
            self._x_cols = x_strip.shape[1]
            self._y_cols = y_strip.shape[1]
            self._qr = IncrementalQR(self._x_cols, self._y_cols)
            self._y_ssp = np.zeros((self._y_cols, self._y_cols))
        self._qr.append(x_strip, y_strip)
        self._y_ssp += np.dot(y_strip.T, y_strip)
        self._x_rows += x_strip.shape[0]
        
    def fit(self):
        assert self._qr is not None, "Must append at least 1 row!"
        r = self._qr.r()
        qty = self._qr.qty()
        assert qty.shape == (self._x_cols, self._y_cols)
        coef = solve(r, qty)
        vcov = inv(np.dot(r.T, r))
        # Q'y is a projection of y onto the subspace of R^nrows that is
        # spanned by the X vectors. In other words, Q'y is (an orthonormal
        # projection of) that part of y which can be explained by
        # X. Therefore, the explained SS is (Q'y ** 2).sum() (if you want to
        # do ordered-entry anova type stuff, then these have the values you
        # need), and the residual sum of squares is the difference between the
        # total sum of squares and the explained sum of squares:
        explained_ssp = np.dot(qty.T, qty)
        rssp = self._y_ssp - explained_ssp
        return LSResult(coef, vcov, rssp, self._x_rows - self._x_cols)

class GroupWeightedLSResult(LSResult):
    def __init__(self, coef, vcov, rss, rdf,
                 group_weights, group_rssp, group_df):
        LSResult.__init__(self, coef, vcov, rss, rdf)
        self._group_weights = group_weights
        self._group_rssp = group_rssp
        self._group_df = group_df

    def group_weights(self):
        """Returns the group weights used to generate this fit."""
        return self._group_weights

    def group_rssp(self):
        """Returns a dictionary of the sum-of-squares-and-products matrices
        for each group. Keys are group names, values are matrices.

        These matrices are *unweighted*."""
        return self._group_rssp

    def group_df(self):
        """Returns a dictionary of the degrees of freedom for each group. Keys
        are group names, values are degrees of freedom.

        Note that these are not residual degrees of freedom, but rather a
        simple count of how many rows were given for each group."""
        return self._group_df

class _XtXAccumulator(object):
    def __init__(self, x_cols, y_cols):
        self.xtx = np.zeros((x_cols, x_cols))
        self.xty = np.zeros((x_cols, y_cols))
        self.y_ssp = np.zeros((y_cols, y_cols))
        self.rows = 0

    @classmethod
    def append_top_half(cls, x_strip, y_strip):
        assert x_strip.ndim == y_strip.ndim == 2
        assert x_strip.shape[0] == y_strip.shape[0]
        if sparse.issparse(x_strip):
            xtx = x_strip.T * x_strip
        else:
            xtx = np.dot(x_strip.T, x_strip)
        if sparse.issparse(x_strip) or sparse.issparse(y_strip):
            xty = x_strip.T * y_strip
        else:
            xty = np.dot(x_strip.T, y_strip)
        if sparse.issparse(y_strip):
            y_ssp = y_strip.T * y_strip
        else:
            y_ssp = np.dot(y_strip.T, y_strip)
        return (x_strip.shape[0], xtx, xty, y_ssp)

    def append_bottom_half(self, top_half_rv):
        (rows, xtx, xty, y_ssp) = top_half_rv
        self.rows += rows
        # If you add a dense array to a sparse matrix, what you get out is a
        # dense np.matrix, and we just want to deal with np.ndarray's.
        print "offending code:", self.xtx, xtx
        print type(self.xtx)
        print type(xtx)
        self.xtx += xtx
        if isinstance(self.xtx, np.matrix):
            self.xtx = np.asarray(self.xtx)
        self.xty += xty
        if isinstance(self.xty, np.matrix):
            self.xty = np.asarray(self.xty)
        self.y_ssp += y_ssp
        if isinstance(self.y_ssp, np.matrix):
            self.y_ssp = np.asarray(self.y_ssp)
            
class XtXGroupWeightedIncrementalLS(object):
    """Perform weighted least-squares regression with very large model
    matrices.
    
    Supports arbitrary numbers of predictors, and for a given model matrix can
    simultaneously solve multiple regression problems with differing outcome
    vectors.

    For each set of data points you pass in, you also specify which "group" it
    belongs too. When fitting your data, you must specify a weight for each
    group of data (e.g., to handle heteroskedasticity). You may call
    fit() repeatedly with different weights; this is much faster than
    recreating your model matrix from scratch.

    Memory usage is roughly (x_cols + y_cols)^2 doubles PER GROUP.

    This works by the direct method (forming X'X and solving it). It is quite
    fast, and can handle sparse matrices (in the sense of scipy.sparse). It
    may be less numerically stable than QR-based methods."""

    def __init__(self):
        self._x_cols = None
        self._y_cols = None
        self._accumulators = {}

    @classmethod
    def append_top_half(cls, group, x_strip, y_strip):
        """The stateless part of append(), split out to ease parallel
        processing. You can run many append_top_half's in different processes
        in parallel, and then queue them into append_bottom_half."""
        return (group,
                x_strip.shape[1], y_strip.shape[1],
                _XtXAccumulator.append_top_half(x_strip, y_strip))

    def append_bottom_half(self, top_half_rv):
        (group, x_cols, y_cols, accumulator_top_half_rv) = top_half_rv
        self._x_cols = x_cols
        self._y_cols = y_cols
        if not group in self._accumulators:
            self._accumulators[group] = _XtXAccumulator(x_cols, y_cols)
        self._accumulators[group].append_bottom_half(accumulator_top_half_rv)

    def append(self, group, x_strip, y_strip):
        self.append_bottom_half(self.append_top_half(group, x_strip, y_strip))

    def groups(self):
        return self._accumulators.keys()

    def fit_unweighted(self):
        ones = dict([(g, 1) for g in self._accumulators.keys()])
        return self.fit(ones)

    def fgls(self, maxiter=100):
        fit = self.fit_unweighted()
        group_df = fit.group_df()
        old_group_weights = [1] * len(self._accumulators)
        for i in xrange(maxiter):
            print "iter %s" % (i,)
            group_rssp = fit.group_rssp()
            group_weights = {}
            for group, rssp in group_rssp.iteritems():
                # Assume that -- if we are multivariate -- the
                # heteroskedasticity parameters are the same for each
                # dimension:
                rss = rssp.diagonal().mean()
                group_weights[group] = 1. / (rss * 1. / group_df[group])
            fit = self.fit(group_weights)
            # XX stupid convergence criterion:
            print group_weights
            if np.allclose(old_group_weights, group_weights.values()):
                break
            old_group_weights = group_weights.values()
        else:
            raise Exception, fit
        return fit

    def fit(self, group_weights):
        assert self._x_cols is not None, "Need at least 1 row!"
        xtwx = np.zeros((self._x_cols, self._x_cols))
        xtwy = np.zeros((self._x_cols, self._y_cols))
        for group, accumulator in self._accumulators.iteritems():
            xtwx += group_weights[group] * accumulator.xtx
            xtwy += group_weights[group] * accumulator.xty
        coef = solve(xtwx, xtwy)
        vcov = inv(xtwx)

        df = 0
        group_df = {}
        group_rssp = {}
        rssp = np.zeros((self._y_cols, self._y_cols))
        for group, accumulator in self._accumulators.iteritems():
            df += accumulator.rows
            group_df[group] = accumulator.rows
            # Residual sum of squares and products matrix is
            #  (Y - XB)'(Y - XB)
            #    = Y'Y - B'X'Y - (B'X'Y')' + B'X'XB
            this_btxty = np.dot(coef.T, accumulator.xty)
            this_rssp = (accumulator.y_ssp
                         - this_btxty
                         - this_btxty.T
                         + np.dot(np.dot(coef.T, accumulator.xtx), coef))
            group_rssp[group] = this_rssp
            rssp += group_weights[group] * this_rssp
        rdf = df - self._x_cols

        return GroupWeightedLSResult(coef, vcov, rssp, rdf,
                                     dict(group_weights), group_rssp, group_df)

class XtXIncrementalLS(object):
    """Perform least-squares regression with very large model matrices.
    
    Supports arbitrary numbers of predictors, and for a given model matrix can
    simultaneously solve multiple regression problems with differing outcome
    vectors.

    This is faster than QRIncrementalLS, and can accept sparse matrices (using
    scipy.sparse), but it may be less numerically stable. It is a thin wrapper
    around XtXGroupWeightedIncrementalLS."""

    _test_sparse = None

    def __init__(self):
        self._gwils = XtXGroupWeightedIncrementalLS()

    @classmethod
    def append_top_half(cls, x_strip, y_strip):
        return XtXGroupWeightedIncrementalLS.append_top_half("foo",
                                                             x_strip, y_strip)

    def append_bottom_half(self, append_top_half_rv):
        return self._gwils.append_bottom_half(append_top_half_rv)

    def append(self, *args):
        return self.append_bottom_half(self.append_top_half(*args))

    def fit(self):
        return self._gwils.fit({"foo": 1})

def _incremental_ls_tst(class_):
    x = np.arange(10)
    r1 = np.array([-1.34802662, 0.88780193, 0.97355492, 1.09878012,
                   -1.24346173, 0.03237138, 1.70651768, -0.70375099,
                   0.21029281, 0.80667505])
    r2 = np.array([0.48670125, -1.82877490, -0.32244478, -1.00960602,
                   0.54804895, -0.24075048, 0.43178080, -1.14938703,
                   -0.07269548, -1.75175427])
    y1 = 1 + 2 * x + r1
    y2 = 3 + 7 * x + r2
    X = np.hstack([np.ones((10, 1)), x.reshape((-1, 1))])
    Y = np.hstack([y1.reshape((-1, 1)), y2.reshape((-1, 1))])
    # True values calculated with R:
    def check(fit):
        assert np.allclose(fit.coef(),
                           np.array([[0.9867409, 2.739645],
                                     [2.0567410, 6.948770]]))
        assert fit.rdf() == 8
        assert np.allclose(fit.rss(),
                           [9.558538, 7.01811])
        assert np.allclose(fit.rssp(),
                           np.array([[9.558538163832505, -2.846727419732163],
                                     [-2.846727419732163, 7.018110344812262]]))
        y1_svcov = np.array([[0.41275506, -0.065171851],
                             [-0.06517185, 0.014482634]])
        y2_svcov = np.array([[0.30305476, -0.047850752],
                             [-0.047850752, 0.010633501]])
        svcov = np.concatenate([y1_svcov[..., np.newaxis],
                                y2_svcov[..., np.newaxis]],
                               axis=2)
        assert np.allclose(fit.scaled_vcov(), svcov)
        assert np.allclose(fit.se(),
                           np.array([[0.6424602, 0.5505041],
                                     [0.1203438, 0.1031189]]))
        (t, p) = fit.t_tests()
        assert np.allclose(t,
                           np.array([[1.535879, 4.976612],
                                     [17.090542, 67.386024]]))
        assert np.allclose(p,
                           np.array([[1.631190e-01, 1.084082e-03], 
                                     [1.396098e-07, 2.617624e-12]]))
        F, df1, df2, p = fit.lht_by_dim([1, 0])
        assert np.allclose(F, [2.358923445752836, 24.76666631211173])
        assert df1 == 1
        assert df2 == 8
        assert np.allclose(p, [0.1631190208872526, 0.001084082203556753])
        F, df1, df2, p = fit.lht_by_dim(np.eye(2))
        assert np.allclose(F, [585.023031528357, 8862.63814885004])
        assert df1 == 2
        assert df2 == 8
        assert np.allclose(p, [2.12672266281485e-09, 4.1419421127122e-14])
        F, df1, df2, p = fit.lht_by_dim([0, 1], rhs=2)
        assert np.allclose(F, [0.2223036807600276, 2303.129418461575])
        assert df1 == 1
        assert df2 == 8
        assert np.allclose(p, [0.6498810865566884, 3.931182195134797e-11])
        for rhs in ([3, 2],
                    [[3, 3], [2, 2]]):
            F, df1, df2, p = fit.lht_by_dim(np.eye(2), rhs=[3, 2])
            assert np.allclose(F, [13.04324941825649, 3912.42755464922])
            assert df1 == 2
            assert df2 == 8
            assert np.allclose(p, [0.003034103325876033, 1.088126971915703e-12])

        mv = fit.lht_multivariate(np.eye(2))
        assert np.allclose(mv["Pillai"],
                           (1.0062957480131434, 4.0506850846259770,
                            4, 16, 0.01866246907197198))
        assert np.allclose(mv["Wilks"],
                           (0.0003166157715902, 193.1988279124278733,
                            4, 14, 4.4477e-12))
        assert np.allclose(mv["Hotelling-Lawley"],
                           (3136.5178539783428278, 4704.7767809675142416,
                            4, 12, 4.684822e-19))
        assert np.allclose(mv["Roy"],
                           (3136.5111954638095995, 12546.0447818552383978,
                            2, 8, 1.0320e-14))
        
        for rhs in ([3, 2],
                    [[3, 3], [2, 2]]):
            mv = fit.lht_multivariate([[3, 1], [0, 2]], rhs)
            assert np.allclose(mv["Pillai"],
                               (1.0000722571514054, 4.0005780989830289,
                                4, 16, 0.0195210910404859))
            assert np.allclose(mv["Wilks"],
                               (0.0004862616193171, 155.2205092909892130,
                                4, 14, 1.989176787533242e-11))
            assert np.allclose(mv["Hotelling-Lawley"],
                               (2054.3575308552108254, 3081.5362962828162381,
                                4, 12, 5.920032e-18))
            assert np.allclose(mv["Roy"],
                               (2054.3569717521481834, 8217.4278870085927338,
                                2, 8, 5.6033913856139e-14))

        # Check multivariate tests on a single variable (via 'subset') give
        # the same answer as univariate tests:
        for subset in (0, [0], np.asarray([True, False])):
            print subset
            mv = fit.lht_multivariate([1, 0], subset=subset)
            F, df1, df2, p = fit.lht_by_dim([1, 0])
            for (mv_raw, mv_F, mv_df1, mv_df2, mv_p) in mv.itervalues():
                assert np.allclose(mv_F, F[0])
                assert np.allclose(mv_df1, df1)
                assert np.allclose(mv_df2, df2)
                assert np.allclose(mv_p, p[0])

    def do_fit(iterable):
        ls = class_()
        for x, y in iterable:
            ls.append(x, y)
        fit = ls.fit()
        check(fit)
    for test_sparse in ([], ["X"], ["Y"], ["X", "Y"]):
        print "test_sparse", test_sparse
        if test_sparse and not hasattr(class_, "_test_sparse"):
            continue
        if "X" in test_sparse:
            tX = sparse.csr_matrix(X)
        else:
            tX = X
        if "Y" in test_sparse:
            tY = sparse.csr_matrix(Y)
        else:
            tY = Y
        # One-shot:
        do_fit([(tX, tY)])
        # Two-shot incremental:
        do_fit([(tX[:5, :], tY[:5, :]),
                (tX[5:, :], tY[5:, :])])
        # Generator:
        def gen():
            for i in xrange(10):
                xrow = tX[i, :]
                # Weird fiddling because this code needs to work for both
                # dense and sparse matrices:
                if xrow.ndim == 1:
                    xrow = xrow.reshape((1, -1))
                yrow = tY[i, :]
                if yrow.ndim == 1:
                    yrow = yrow.reshape((1, -1))
                yield xrow, yrow
        do_fit(gen())

def test_incremental_ls():
    for class_ in (XtXIncrementalLS, QRIncrementalLS):
        _incremental_ls_tst(class_)

def _group_weighted_incremental_ls_tst(class_):
    x = np.arange(10)
    r1 = np.array([-1.34802662, 0.88780193, 0.97355492, 1.09878012,
                   -1.24346173, 0.03237138, 1.70651768, -0.70375099,
                   0.21029281, 0.80667505])
    r2 = np.array([0.48670125, -1.82877490, -0.32244478, -1.00960602,
                   0.54804895, -0.24075048, 0.43178080, -1.14938703,
                   -0.07269548, -1.75175427])
    y1 = 1 + 2 * x + r1
    y2 = 3 + 7 * x + r2
    X = np.hstack([np.ones((10, 1)), x.reshape((-1, 1))])
    Y = np.hstack([y1.reshape((-1, 1)), y2.reshape((-1, 1))])

    groups = np.array(["a"] * 5 + ["b"] * 5)
    Xa = X[groups == "a", :]
    Xb = X[groups == "b", :]
    Ya = Y[groups == "a", :]
    Yb = Y[groups == "b", :]

    # True values calculated with R:
    def check(ls):
        fit11 = ls.fit({"a": 1, "b": 1})
        resid = Y - np.dot(X, fit11.coef())
        assert np.allclose(fit11.coef(),
                           np.array([[0.9867409, 2.739645],
                                     [2.0567410, 6.948770]]))
        assert fit11.rdf() == 8
        assert np.allclose(fit11.rss(),
                           [9.558538, 7.01811])
        assert np.allclose(fit11.rssp(),
                           np.array([[9.558538163832505, -2.846727419732163],
                                     [-2.846727419732163, 7.018110344812262]]))
        y1_svcov = np.array([[0.41275506, -0.065171851],
                             [-0.06517185, 0.014482634]])
        y2_svcov = np.array([[0.30305476, -0.047850752],
                             [-0.047850752, 0.010633501]])
        svcov = np.concatenate([y1_svcov[..., np.newaxis],
                                y2_svcov[..., np.newaxis]],
                               axis=2)
        assert np.allclose(fit11.scaled_vcov(), svcov)
        assert np.allclose(fit11.se(),
                           np.array([[0.6424602, 0.5505041],
                                     [0.1203438, 0.1031189]]))
        (t, p) = fit11.t_tests()
        assert np.allclose(t,
                           np.array([[1.535879, 4.976612],
                                     [17.090542, 67.386024]]))
        assert np.allclose(p,
                           np.array([[1.631190e-01, 1.084082e-03], 
                                     [1.396098e-07, 2.617624e-12]]))
        # Not from R:
        assert fit11.group_df() == {"a": 5, "b": 5}
        assert np.allclose(fit11.group_rssp()["a"],
                           np.array([[6.267541851659734, -4.28037434739278],
                                     [-4.280374347392778, 4.24310595227175]]))
        assert np.allclose(fit11.group_rssp()["b"],
                           np.array([[3.290996312172771, 1.433646927660614],
                                     [1.433646927660614, 2.775004392540514]]))

        fit12 = ls.fit({"a": 1, "b": 2})
        assert np.allclose(fit12.coef(),
                           np.array([[1.009419062117645, 2.877818731529399],
                                     [2.054144681852942, 6.926762882588236]]))
        assert fit12.rdf() == 8
        # Really we care about rssp and vcov being accurate, but R doesn't
        # provide them, it does provide standard error, and standard error
        # involves all of the things we *do* care about, so we can just check
        # it:
        assert np.allclose(fit12.se(),
                           np.array([[0.7142305799847996, 0.6216165223656146],
                                     [0.1190384299974666, 0.1036027537276024]]))
        assert fit12.group_df() == {"a": 5, "b": 5}
        assert np.allclose(fit12.group_rssp()["a"],
                           np.array([[6.27300555451370, -4.253108798401872],
                                     [-4.25310879840187, 4.415039078639084]]))
        assert np.allclose(fit12.group_rssp()["b"],
                           np.array([[3.287297566115147, 1.415398625005633],
                                     [1.415398625005633, 2.658347656920942]]))

    # Big chunks:
    ls_big = class_()
    ls_big.append("a", Xa, Ya)
    ls_big.append("b", Xb, Yb)
    check(ls_big)

    # Row at a time, interleaved:
    ls_row = class_()
    for i in xrange(5):
        ls_row.append("a", Xa[[i], :], Ya[[i], :])
        ls_row.append("b", Xb[[i], :], Yb[[i], :])
    check(ls_row)

def test_group_weighted_ls():
    for class_ in (XtXGroupWeightedIncrementalLS,):
        _group_weighted_incremental_ls_tst(class_)

if __name__ == "__main__":
    import nose
    nose.runmodule()
