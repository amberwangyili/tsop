import numpy as np

from tsop.basic import (at_nan2zero, at_signlog, at_signsqrt, at_zero2nan,
                        cs_rank, cs_remove_middle, cs_scale, cs_winsor,
                        cs_zscore, ts_corr_binary, ts_delay, ts_diff, ts_fill,
                        ts_max, ts_mean, ts_mean_exp, ts_median, ts_min)
from tsop.basic import ts_rank as ts_rank_window
from tsop.basic import ts_skew, ts_std, ts_std_normalized, ts_sum, ts_zscore


def assert_allclose_exact(actual, expected, tol=1e-4):
    """Compare arrays exactly on NaN positions and within tol on finite values."""
    assert (
        actual.shape == expected.shape
    ), f"shape mismatch: {actual.shape} vs {expected.shape}"
    # NaN-mask must match
    nan_actual = np.isnan(actual)
    nan_expected = np.isnan(expected)
    assert np.array_equal(
        nan_actual, nan_expected
    ), f"NaN positions differ:\n got {nan_actual}\n expected {nan_expected}"
    # compare finite entries
    mask = ~nan_expected
    np.testing.assert_allclose(actual[mask], expected[mask], atol=tol, rtol=0)


class TestCrossSectional:
    def test_cs_zscore_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = cs_zscore(inp)
        exp = np.array(
            [
                [-1.2247, 1.3363, np.nan, -1.0000, np.nan],
                [0.0000, -1.0690, np.nan, 1.0000, np.nan],
                [1.2247, -0.2673, np.nan, np.nan, np.nan],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_cs_winsor_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = cs_winsor(inp, 0.05, False)
        exp = np.array(
            [
                [3, 1, np.nan, 3, np.nan],
                [3, 1, 4, 5, -2],
                [3, 1, np.nan, np.nan, 1],
            ]
        )
        assert_allclose_exact(out, exp)

    def test_cs_scale_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = cs_scale(inp)
        exp = np.array(
            [
                [1.0000, 2.0000, np.nan, 1.0000, np.nan],
                [1.5000, 1.0000, 1.5000, 2.0000, 1.0000],
                [2.0000, 1.3333, np.nan, np.nan, 2.0000],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_cs_remove_middle_example(self):
        inp = np.array(
            [[0.81], [0.91], [0.13], [0.91], [0.63], [0.10], [0.28]], dtype=float
        )
        out = cs_remove_middle(inp, 0.5)
        exp = np.array([[np.nan], [0.91], [np.nan], [
                       0.91], [np.nan], [0.10], [np.nan]])
        assert_allclose_exact(out, exp)

    def test_cs_rank_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = cs_rank(inp)
        exp = np.array(
            [
                [1.0, 2.0, np.nan, 1.0, np.nan],
                [1.5, 1.0, 1.5, 2.0, 1.0],
                [2.0, 1.5, np.nan, np.nan, 2.0],
            ]
        )
        assert_allclose_exact(out, exp)


class TestArrayTransforms:
    def test_at_nan2zero_example(self):
        inp = np.array([[2, 3, np.nan, 3, np.inf], [
                       3, 0, 4, 5, -2]], dtype=float)
        out = at_nan2zero(inp)
        exp = np.array([[2, 3, 0, 3, 0], [3, 0, 4, 5, -2]], dtype=float)
        assert_allclose_exact(out, exp)

    def test_at_signlog_example(self):
        inp = np.array([[2, 3, np.nan, 3, np.inf], [
                       3, 0, 4, 5, -2]], dtype=float)
        out = at_signlog(inp)
        exp = np.array(
            [
                [1.0986, 1.3863, np.nan, 1.3863, np.nan],
                [1.3863, 0.0000, 1.6094, 1.7918, -1.0986],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_at_signsqrt_example(self):
        inp = np.array([[2, 3, np.nan, 3, np.inf], [
                       3, 0, 4, 5, -2]], dtype=float)
        out = at_signsqrt(inp)
        exp = np.array(
            [
                [1.7321, 2.0000, np.nan, 2.0000, np.nan],
                [2.0000, 0.0000, 2.2361, 2.4495, -1.7321],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_at_zero2nan_example(self):
        inp = np.array([[0.0, 2, 0.0], [3, 0.0, 4]], dtype=float)
        out = at_zero2nan(inp)
        exp = np.array([[np.nan, 2.0, np.nan], [3.0, np.nan, 4.0]])
        assert_allclose_exact(out, exp)


class TestTimeSeries:
    def test_ts_std_normalized_example(self):
        inp = np.array([[1, 3, 2, 3, 4], [3, 0, np.nan, 5, -2]], dtype=float)
        out = ts_std_normalized(inp, 3)
        exp = np.array(
            [
                [np.nan, np.nan, 2.0000, 5.1962, 4.0000],
                [np.nan, np.nan, np.nan, 1.4142, -0.4041],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_zscore_example(self):
        inp = np.array(
            [
                [0.6557, 0.8491, 0.6787, 0.7431, 0.6555],
                [0.0357, 0.9340, 0.7577, 0.3922, 0.1712],
            ],
            dtype=float,
        )
        out = ts_zscore(inp, 3)
        exp = np.array(
            [
                [np.nan, np.nan, np.nan, 0.1322, -0.8782],
                [np.nan, np.nan, np.nan, -0.3448, -1.1361],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_delay_example(self):
        inp = np.array([[1, 3, 2, 3, 4], [3, 0, 4, 5, -2]], dtype=float)
        out = ts_delay(inp, 2)
        exp = np.array([[np.nan, np.nan, 1, 3, 2], [np.nan, np.nan, 3, 0, 4]])
        assert_allclose_exact(out, exp)

    def test_ts_median_example(self):
        inp = np.array([[1, 2, 3, 4], [4, 5, 6, np.nan]], dtype=float)
        out = ts_median(inp, 2)
        exp = np.array(
            [[np.nan, 1.5000, 2.5000, 3.5000], [np.nan, 4.5000, 5.5000, np.nan]]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_min_example(self):
        inp = np.array([[1, 2, 3, 4], [4, 5, 6, np.nan]], dtype=float)
        out = ts_min(inp, 3)
        exp = np.array([[1, 1, 1, 2], [4, 4, 4, np.nan]])
        assert_allclose_exact(out, exp)

    def test_ts_max_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = ts_max(inp, 2)
        exp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 3, 4, 5, 5], [4, 4, np.nan, np.nan, 1]]
        )
        assert_allclose_exact(out, exp)

    def test_ts_mean_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = ts_mean(inp, 2)
        exp = np.array(
            [
                [np.nan, 2.5000, np.nan, 3.0000, np.nan],
                [np.nan, 1.5000, 2.0000, 4.5000, 1.5000],
                [np.nan, 2.5000, np.nan, np.nan, 1.0000],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_fill_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = ts_fill(inp)
        exp = np.array(
            [[2, 3, 3, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, 1, 1, 1]])
        assert_allclose_exact(out, exp)

    def test_ts_mean_exp_example(self):
        inp = np.array(
            [[2, 3, np.nan, 3, np.inf], [3, 0, 4, 5, -2], [4, 1, np.nan, np.nan, 1]],
            dtype=float,
        )
        out = ts_mean_exp(inp, 3, 0.2)
        exp = np.array(
            [
                [2.0000, 2.5556, np.nan, 3.0000, np.nan],
                [3.0000, 1.3333, 2.4262, 3.3607, 1.8689],
                [4.0000, 2.3333, np.nan, np.nan, 1.0000],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_diff_example(self):
        inp = np.array([[1, 3, 2, 3, 4], [3, 0, 4, 5, -2]], dtype=float)
        out = ts_diff(inp, 1)
        exp = np.array(
            [[np.nan, 2, -1, 1, 1], [np.nan, -3, 4, 1, -7]], dtype=float)
        assert_allclose_exact(out, exp)

    def test_ts_corr_binary_example(self):
        A = np.array([[1, 3, 2, 3, 4], [3, 0, 4, 5, -2]], dtype=float)
        B = np.array([[2, 3, 3, 4, 3], [2, 1, 5, 4, -2]], dtype=float)
        out = ts_corr_binary(A, B, 3)
        exp = np.array(
            [
                [np.nan, np.nan, 0.8660, 0.5000, 0],
                [np.nan, np.nan, 0.8462, 0.9078, 0.9651],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_sum_example(self):
        inp = np.array([[1, 3, 2, 3, 4], [3, 0, np.nan, 5, -2]], dtype=float)
        out = ts_sum(inp, 3)
        exp = np.array([[1, 4, 6, 8, 9], [3, 3, 3, 5, 3]])
        assert_allclose_exact(out, exp)

    def test_ts_std_example(self):
        inp = np.array([[1, 3, 2, 3, 4], [3, 0, np.nan, 5, -2]], dtype=float)
        out = ts_std(inp, 3)
        exp = np.array(
            [
                [np.nan, np.nan, 1.0000, 0.5774, 1.0000],
                [np.nan, np.nan, np.nan, 3.5355, 4.9497],
            ]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_skew_example(self):
        inp = np.array([[1, 2, 2, 3], [2, 3, 3, 2]], dtype=float)
        out = ts_skew(inp, 3)
        exp = np.array(
            [[np.nan, np.nan, -0.7071, 0.7071], [np.nan, np.nan, -0.7071, -0.7071]]
        )
        assert_allclose_exact(np.round(out, 4), exp)

    def test_ts_rank_example(self):
        inp = np.array([[1, 2, 3, 4], [4, 5, 6, np.nan]], dtype=float)
        out = ts_rank_window(inp, 3)
        exp = np.array([[1.500, 2.000, 2.000, 2.000],
                       [1.500, 2.000, 2.000, np.nan]])
        assert_allclose_exact(np.round(out, 3), exp)
