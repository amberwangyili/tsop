#basic.pyx

"""
tsop.basic
==========

一组面向二维金融时间序列的 C++/Cython 算子封装。

包含：
- 横截面 z-score（cs_zscore）
- 列式 Winsorization（cs_winsor）
- 线性缩放（cs_scale）
- 时间序列滚动算子（ts_mean, ts_std, ts_diff, ts_corr_binary 等）
- ……

所有接口在此模块下均有完整的 NumPy 风格 docstring，可直接被 Sphinx autodoc 提取。
"""

#distutils : language = c++
#cython : language_level = 3
import numpy as np

cimport numpy as np

np.import_array()

cdef extern from "basic.h" namespace "tsop":
    void ts_rank_c(const double* A, double* V, int n, int d, int days)
    void ts_skew_c(const double* A, double* V, int n, int d, int days)
    void ts_std_c(const double* A, double* V, int n, int d, int days)
    void ts_sum_c(const double* A, double* V, int n, int d, int days)
    void ts_corr_binary_c(const double* A, const double* B, double* V, int n, int d, int days)
    void ts_diff_c(const double* A, double* V, size_t n_rows, size_t n_cols, int order)
    void ts_mean_exp_c(const double* A, double* V, int n, int d, int days, double exp_factor)
    void ts_fill_c(const double* A, double* V, int n, int d)
    void ts_min_c(const double* A, double* V, int n, int d, int days)
    void ts_max_c(const double* A, double* V, int n, int d, int days)
    void ts_median_c(const double* A, double* V, int n, int d, int days)
    void ts_mean_c(const double* A, double* V, int n, int d, int days)
    void ts_delay_c(const double* A, double* V, int n, int d, int days)
    void ts_zscore_c(const double* A, double* V, int n, int d, int days)
    void ts_std_normalized_c(const double* A, double* V, int n, int d, int days)
    void at_nan2zero_c(const double* A, double* V, int n, int d)
    void at_zero2nan_c(const double* A, double* V, int n, int d)
    void at_signlog_c(const double* A, double* V, int n, int d)
    void at_signsqrt_c(const double* A, double* V, int n, int d)
    void cs_rank_c(const double* A, double* V, int n, int d)
    void cs_remove_middle_c(const double* A, double* V, int n, int d, double filter_percentile)
    void cs_scale_c(const double* A, double* V, int n, int d)
    void cs_winsor_c(const double* A, double* V, int n, int d, double filter_percentile, bint remove_extreme)
    void cs_zscore_c(const double* A, double* V, int n, int d)

import numpy as np

cimport numpy as np
from cython cimport numeric


def cs_zscore(np.ndarray[numeric, ndim=2] A):
    """
    Computes column-wise z-scores for stock values.

    Standardizes the data in array `A` by subtracting the mean and dividing by the standard deviation for each column (day).

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d) containing the z-scores.

    Notes
    -----
    - **信号组合的重缩放**： 当组合不同变量（如收益率和成交量）且其数量级差异巨大时，`cs_zscore` 确保两者在统一缩放下平等贡献，同时保留各变量内部的相对大小。
    
    - **Alpha 示例**：
    
        - **动量 Alpha**：对每只股票的 5 天收益率进行横截面标准化，识别表现优于平均水平的股票：

            .. math::

                \mathrm{alpha} = \mathbb{1}(\mathtt{cs\_zscore}(\mathrm{return\_5}) > 1.0)

        - **成交量调整的动量 Alpha**：结合 z-score 标准化后的收益率和成交量，识别具有高动量和流动性的股票：

            .. math::

                \mathrm{alpha} = \mathtt{cs\_zscore}(\mathrm{return\_5}) \cdot \mathtt{cs\_zscore}(\mathrm{volume})

        - **市值 Alpha**：对 PE 比率进行 z-score 标准化，筛选估值过高的股票：

            .. math::

                \mathrm{alpha} = \mathbb{1}(\mathtt{cs\_zscore}(\mathrm{PE}) < -0.5)

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf],
                          [3, 0, 4, 5, -2],
                          [4, 1, np.nan, np.nan, 1]])
    >>> cs_zscore(input)
    array([[-1.2247, 1.3363, np.nan, -1.0000, np.nan], 
           [0, -1.0690, np.nan, 1.0000, np.nan],
           [1.2247, -0.2673, np.nan, np.nan, np.nan]])
    """


    # Ensure input array is C-contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    # Call the C++ function using the data pointers
    cs_zscore_c(&A_c[0, 0], &V[0, 0], n, d)
    return V



def cs_winsor(np.ndarray[numeric, ndim=2] A, double filter_percentile, remove_extreme):
    """
    Applies column-wise Winsorization to stock values.

    Limits extreme data points in each column (day) by either replacing them
    with the nearest threshold values (Winsorization) or removing them (setting to NaN),
    based on the specified percentile.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    filter_percentile : float
        The percentile for filtering (e.g., 0.05 for 5%).
    remove_extreme : bool
        If True, extreme values are set to NaN; if False, they are replaced with threshold values.

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the Winsorized values are stored.

    Notes
    -----
    - **异常值处理**：
        
        - 异常值会为回归、分类或 alpha 生成模型引入噪声和不稳定性。通过减少这些值的影响，通过 `cs_winsor`，可以有效地处理异常值，将它们限制在一定阈值范围内或直接移除。
       
        - 在基于动量的策略中，异常值（极端收益率）通常包含有用信息，应进行 Winsorization，而非直接移除，因为它们可能指示强劲的价格趋势。
       
        - 收益率或基本面比率中的异常值可能会扭曲用于 alpha 信号的排序或平均值。通过 `cs_winsor`，可以确保横截面度量更稳定可靠，从而提升股票选择效果。

    - **Alpha 示例**：

        - **动量 Alpha**：在 Winsorization 后根据最近 5 天收益率对股票进行排序，减少极端值的影响：

            .. math::

                \mathrm{alpha} = \mathtt{cs\_rank}(\mathtt{cs\_winsor}(\mathrm{return\_5}, 0.05, \mathtt{remove\_extreme=False}))

        - **均值回归 Alpha**：筛选出极端 3 天收益率（例如最高和最低 1%）以聚焦于可能均值回归的股票：

            .. math::

                \mathrm{alpha} = \mathbb{1}(\mathrm{return\_3}\  \mathrm{\ is\ in\ winsorized\ range})

        - **估值 Alpha**：对估值指标（如 PE 或 PB 比率）进行 Winsorization，以消除异常值（例如由于一次性事件导致的异常高或低值）。清洗后的数据可以提高信号质量：
     
            .. math::

                \mathrm{alpha} = \mathtt{cs\_rank}(\mathtt{cs\_winsor}(\mathrm{PE}, 0.01, \mathtt{remove\_extreme=True}))

        - **流动性 Alpha**：在A股市场中，流动性较差的股票可能会扭曲分析结果，对每日成交量数据进行 Winsorization，以移除或限制极端值：

            .. math::

                \mathrm{alpha} = \mathtt{cs\_rank}(\mathtt{cs\_winsor}(\mathrm{volume}, 0.05, \mathtt{remove\_extreme=False}))

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                          [3, 0, 4, 5, -2], 
                          [4, 1, np.nan, np.nan, 1]])
    >>> cs_winsor(input, 0.05, False)
    array([[3, 1, np.nan, 3, np.nan], 
           [3, 1, 4, 5, -2], 
           [3, 1, np.nan, np.nan, 1]])
    """

    # Ensure input array is C-contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    # Call the C++ function using the data pointers
    cs_winsor_c(&A_c[0, 0], &V[0, 0], n, d, filter_percentile, remove_extreme)
    return V



def cs_scale(np.ndarray[numeric, ndim=2] A):
    """
    Scales stock values between 1 and 2 for each day.

    Linearly scales the values in each column (day) so that the minimum value maps to 1 and the maximum to 2.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the scaled values are stored.

    Notes
    -----
    - **适用场景**：  

    - **不同数量级数据的融合**：例如将收益率（Return）和成交量（Volume）两个数量级相差较大的数据合并处理时，可对两者分别缩放，确保其相对大小不会失真。

    - **相对距离的保留**：与排序方法（如cs_rank）不同，**cs_scale** 不会破坏原始数据的相对距离，对价格波动率、动量等绝对差异重要的场景尤为有用。

    - **因子实例**：  

        - **动量-成交量因子** : 将10日动量和日均成交量分别用 **cs_scale** 归一化， 该因子能捕捉动量高且成交活跃的股票。  

        - **风险调整收益因子**  : 将收益率和波动率分别使用 **cs_scale** 归一化，该因子捕捉高收益、低风险的股票。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                          [3, 0, 4, 5, -2], 
                          [4, 1, np.nan, np.nan, 1]])
    >>> cs_scale(input)
    array([[1.0000, 2.0000, np.nan, 1.0000, np.nan],
           [1.5000, 1.0000, 1.5000, 2.0000, 1.0000],
           [2.0000, 1.3333, np.nan, np.nan, 2.0000]])
    """
    # Ensure input array is C-contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    # Call the C++ function using the data pointers
    cs_scale_c(&A_c[0, 0], &V[0, 0], n, d)

    return V


def cs_remove_middle(np.ndarray[numeric, ndim=2] A, double filter_percentile):
    """
    Removes middle percentile values in each column (day).

    Sets to NaN the values within the specified middle percentile range,
    effectively removing the middle values while keeping the extremes.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    filter_percentile : float
        The percentile range to remove (e.g., 0.2 for middle 20%).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the filtered values are stored.

    Notes
    -----
    - 有时, alpha 中的极端值比中间值具有更强的预测能力，因此，仅基于极端值进行交易可以带来更高的回报，但代价是更高的换手率和更高的利润与亏损（PnL）波动性。例如，filter_percentile = 0.80 意味着我们将移除中间 80% 的数据，仅保留前后10%的尾部数据。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[0.81], [0.91], [0.13], [0.91], [0.63], [0.10], [0.28]])
    >>> cs_remove_middle(input, 0.5)
    array([[ nan],
           [0.91],
           [ nan],
           [0.91],
           [ nan],
           [0.1 ],
           [ nan]])
    """

#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

#Call the C++ function using the data pointers
    cs_remove_middle_c(&A_c[0, 0], &V[0, 0], n, d, filter_percentile)
    return V

def cs_rank(np.ndarray[numeric, ndim=2] A):
    """
    Assigns ranks to stock values each day, scaling the ranks uniformly between 1 and 2.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the scaled ranks are stored.

    Notes
    -----
    - **标准化的横截面排序**：`cs_rank` 在股票横截面排序中尤其有效。通过将排名缩放到 1 和 2 之间，它避免了数值不稳定性，并确保不同交易日的排名具有可比性。这对于检测表现指标（如收益率、波动率）的相对强弱非常有用。

    - **尾部信号提取**：

       - 能够隔离特定比例的股票（例如，前 10% 或后 10%），使该函数非常适合构建针对极端表现者的 alpha。例如：

         - :math:`\mathrm{alpha} = \mathbb{1}(\mathtt{cs\_rank}(\mathrm{return}) > 1.9)` 识别动量领先者。

         - :math:`\mathrm{alpha} = \mathbb{1}(\mathtt{cs\_rank}(\mathrm{volatility}) < 1.1)` 捕捉低波动率股票。

    - **Alpha实例**   

        - **动量 Alpha**： 基于过去 5 天收益率对股票进行排名构建

            .. math:: 

                \mathrm{alpha} = \mathbb{1}(\mathtt{cs\_rank}(\mathtt{return\_5}) > 1.8)

        - **波动率 Alpha**：利用历史价格波动率的横截面排名来寻找低风险股票, 偏向于价格波动较稳定的股票。

            .. math::

                \mathrm{alpha} = \mathbb{1}(\mathtt{cs\_rank}(\mathrm{volatility}) < 1.2)

        - **流动性 Alpha**：基于过去 10 天日均交易量对股票进行排名, 在构建其他信号之前用于过滤流动性不足的股票。
        
            .. math::

                \mathrm{alpha} = \mathtt{cs\_rank}(\mathtt{volume\_10})

        - **反转 Alpha**：通过对 3 日收益率进行排名，定位近期表现较差的股票：

            .. math::

                \mathrm{alpha} = \mathbb{1}(\mathtt{cs\_rank}(\mathtt{return\_3}) < 1.1)


    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf],
    ...                   [3, 0, 4, 5, -2],
    ...                   [4, 1, np.nan, np.nan, 1]])
    >>> cs_rank(input)
    array([[1.0, 2.0,   nan, 1.0,   nan],
           [1.5, 1.0, 1.5, 2.0, 1.0],
           [2.0, 1.5,   nan,  nan, 2.0]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

#Call the C++ function using the data pointers
    cs_rank_c(&A_c[0, 0], &V[0, 0], n, d)

    return V

def at_nan2zero(np.ndarray[numeric, ndim=2] A):
    """
    Converts NaN and infinite values to zero.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where NaN and infinite values are replaced with zero.

    Notes
    -----
    - 在组合多个 alpha 信号时，NaN 值的存在可能会传播并污染原本可靠的信号。使用 at_nan2zero 可以确保在聚合过程中仅使用有效数据，从而维护组合 alpha 的完整性。当处理稀疏变量（计算 NaN 的平均值会得到 NaN）时，也有助于减缓其影响。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                         [3, 0, 4, 5, -2]])
    >>> at_nan2zero(input)
    array([[2, 3, 0, 3, 0],
           [3, 0, 4, 5, -2]])
    """
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    at_nan2zero_c(&A_c[0, 0], &V[0, 0], n, d)
    return V

def at_signlog(np.ndarray[numeric, ndim=2] A):
    """
    Applies a sign-preserving logarithmic transformation.

    Computes `sign(A) * log(1 + |A|)` for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the transformed values are stored.

    Notes
    -----
    - 许多机器学习模型在数据经过对数变换以减少偏态时表现更佳, 而原始收益/现金流/净利润可能差异很大。应用 at_signlog 可以按比例缩放, 而标准对数无法处理负值或零。at_signlog 函数能够在应用对数缩放的同时保留数据的符号，使数据更易于后续处理。 

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                         [3, 0, 4, 5, -2]])
    >>> at_signlog(input)
    array([[1.0986, 1.3863, np.nan, 1.3863, np.nan],
          [1.3863, 0, 1.6094, 1.7918, -1.0986]])
    """
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    at_signlog_c(&A_c[0, 0], &V[0, 0], n, d)
    return V

def at_signsqrt(np.ndarray[numeric, ndim=2] A):
    """
    Applies a sign-preserving square root transformation.

    Computes `sign(A) * sqrt(|A| + 1)` for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the transformed values are stored.

    Notes
    -----
    平方根变换减少了大值和小值之间的差距，但其平滑效果比对数变换弱，适合数据中较少的极端值。对数变换会显著压缩极大值，同时扩大极小值的相对差距。适合数据分布跨度极大或者具有指数增长特性数据的情况。另外, 平方根变换对小值变化更敏感, 对数变换对大值变化更敏感, 因此, 如果数据主要为正值、跨度较小或更关注小值的变化，则优先考虑平方根变换, 如果数据跨度极大、分布偏态严重或具有指数特性，则对数变换更适合。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                         [3, 0, 4, 5, -2]])
    >>> at_signsqrt(input)
    array([[1.7321, 2.0000, np.nan, 2.0000,  np.nan],
           [2.0000,      0, 2.2361, 2.4495, -1.7321])
    """
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    at_signsqrt_c(&A_c[0, 0], &V[0, 0], n, d)
    return V
    

def at_zero2nan(np.ndarray[numeric, ndim=2] A):
    """
    Converts zero values to NaN.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where zero values are replaced with NaN.

    Notes
    -----
    - 在信号处理中，零值可能代表无效或丢失的数据。通过将零值替换为 NaN，可以避免在求和或取平均时将这些无效值误算为有意义的贡献。这对于后续的聚合、加权平均或其他数学操作非常重要。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[0.0, 2, 0.0],
    ...                   [3, 0.0, 4]])
    >>> at_zero2nan(input)
    array([[nan, 2.0, nan],
           [3.0, nan, 4.0]])
    """
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    at_zero2nan_c(&A_c[0, 0], &V[0, 0], n, d)
    return V


def ts_std_normalized(np.ndarray[numeric, ndim=2] A, int days):
    """
    Normalizes each value by the standard deviation over a time window.

    Divides each element by the standard deviation computed over the specified window size.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days) for computing the standard deviation.

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the normalized values are stored.

    Notes
    -----
    - `ts_std_normalized` 通过标准化标准差，它允许在不同股票或不同时期之间比较波动水平，而不受其绝对价格水平的影响。归一化值可作为动量指标。 归一化值的突然飙升可能表明突破，因为价格相对于近期波动率发生显著变化。 较高的归一化值可能表示在波动率调整后强劲的上升动量。使用归一化值设置动态阈值，适应变化的市场波动率，用于交易的进入和退出。 

    - **注意点**

        1. **涨跌停板效应**：在计算滚动标准差时，需要考虑涨跌停板对价格波动的限制，可能需要对停板数据进行特殊处理。

        2. **交易活跃度**：部分股票可能存在流动性不足的问题，导致价格跳跃和波动率异常，这需要在策略中进行调整。

        3. **数据质量**：确保使用高质量的数据，包括处理停牌、复牌和除权除息等事件对价格和波动率的影响。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 3, 2, 3, 4], 
                          [3, 0, np.nan, 5, -2]])
    >>> ts_std_normalized(input, 3)
    array([[np.nan, np.nan, 2.0000, 5.1962, 4.0000],
           [np.nan, np.nan, np.nan, 1.4142, -0.4041]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A
    
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_std_normalized_c(A_data, V_data, n, d, days)

    return V

def ts_zscore(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes rolling z-scores over a specified window size.

    Calculates the z-score for each element using the mean and standard deviation over a rolling window.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days) for computing mean and standard deviation.

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the z-scores are stored.

    Notes
    -----
    `ts_zscore` 函数有助于识别股票价格或其他指标何时显著偏离其近期平均值。 通过考虑指定窗口内的标准差，`ts_zscore` 对数据进行波动率归一化，确保信号不会被波动率的变化所过度影响。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[0.6557, 0.8491, 0.6787, 0.7431, 0.6555], 
                          [0.0357, 0.9340, 0.7577, 0.3922, 0.1712]])
    >>> ts_zscore(input, 3)
    array([[np.nan, np.nan, np.nan, 0.1322, -0.8782],
           [np.nan, np.nan, np.nan, -0.3448, -1.1361]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A
    
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_zscore_c(A_data, V_data, n, d, days)

    return V

def ts_delay(np.ndarray[numeric, ndim=2] A, int days):
    """
    Shifts the time series data backward by a specified number of days.

    Delays the data by the specified number of days, inserting NaN at the beginning.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Number of days to delay.

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the delayed data is stored.

    Notes
    -----
    - `ts_delay` 是时间序列分析中的基础算子， 通过将数据向后移动，可以创建滞后变量， 滞后特征用于捕捉过去事件对当前或未来事件的影响。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 3, 2, 3, 4], 
                          [3, 0, 4, 5, -2]])
    >>> ts_delay(input, 2)
    array([[np.nan, np.nan, 1, 3, 2],
           [np.nan, np.nan, 3, 0, 4]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A
    
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_delay_c(A_data, V_data, n, d, days)

    return V

def ts_median(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling median over a specified window size.

    Calculates the median value within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the median values are stored.

    Notes
    -----

    - 中位数可作为均值回归策略的基准。通过将当前价格与中位数比较，交易者可以识别超买或超卖状态。由于中位数不受异常值影响，在存在缺失值或异常的数据集中，中位数可以作为更好的度量

    - **中位数绝对偏差（MAD） Alpha**：使用 MAD 作为稳健的变异性度量, 有助于识别未被标准差捕捉到的异常值或波动性变化。

        .. math::

            \mathrm{MAD} = \mathtt{ts\_median}( | \mathrm{price} - \mathtt{ts\_median}(\mathrm{price}, n) |, n )


    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 2, 3, 4], [4, 5, 6, np.nan]])
    >>> ts_median(input, 2)
    array([[np.nan, 1.5000, 2.5000, 3.5000],
           [np.nan, 4.5000, 5.5000, np.nan]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_median_c(A_data, V_data, n, d, days)
    return V

def ts_min(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling minimum over a specified window size.

    Calculates the minimum value within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the minimum values are stored.

    Notes
    -----
    `ts_min` 函数可用于计算指定窗口期内的最小价格，从而识别支撑位。交易者常常利用近期的最低价来判断可能出现买盘兴趣的支撑水平 ，特别是对于波动较大的中小盘股票, 而对于估值较低的蓝筹股，当价格达到近期新低时，可能是逢低买入的机会。`ts_min` 也有助于设置止损位。如果价格跌破此最低点，可能预示着下行趋势，及时退出可限制损失。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 2, 3, 4], 
                          [4, 5, 6, np.nan]])
    >>> ts_min(input, 3)
    array([[1, 1, 1, 2],
           [4, 4, 4, np.nan]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_min_c(A_data, V_data, n, d, days)
    return V

def ts_max(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling maximum over a specified window size.

    Calculates the maximum value within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the maximum values are stored.

    Notes
    -----
    `ts_max` 函数可用于计算指定窗口期内的最高价格，从而识别阻力位。交易者常利用近期的最高价来判断可能出现卖压增加的阻力水平。在行情比较好的时候, 若某只股票突破近期高点，也可能吸引更多的买盘，推动价格进一步上涨。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                          [3, 0, 4, 5, -2], 
                          [4, 1, np.nan, np.nan, 1]])
    >>> ts_max(input, 2)
    array([[2, 3, np.nan, 3, np.inf], 
           [3, 3, 4, 5, 5], 
           [4, 4, np.nan, np.nan, 1]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_max_c(A_data, V_data, n, d, days)
    return V

def ts_mean(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling mean over a specified window size.

    Calculates the average value within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the mean values are stored.

    Notes
    -----
    `ts_mean` 函数广泛用于平滑短期波动，突出时间序列数据的长期趋势。 通过使用移动平均，交易信号的反应速度被减缓，导致交易频率降低。这可以减少交易成本、滑点以及交易对市场价格的影响，特别有利于交易成本较高或流动性较低的市场。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                          [3, 0, 4, 5, -2], 
                          [4, 1, np.nan, np.nan, 1]])
    >>> ts_mean(input, 2)
    array([[np.nan, 2.5000, np.nan, 3.0000, np.nan], 
           [np.nan, 1.5000, 2.0000, 4.5000, 1.5000],
           [np.nan, 2.5000, np.nan, np.nan, 1.0000]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_mean_c(A_data, V_data, n, d, days)
    return V


def ts_fill(np.ndarray[numeric, ndim=2] A):
    """
    Fills NaN values with the last valid observation.

    Propagates the last valid (non-NaN) value forward to fill NaNs.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the filled values are stored.

    Notes
    -----
    `ts_fill` 函数对于处理包含缺失值或不规则间隔报告的时间序列数据非常有价值。比如, A股上市公司每季度发布财务报告，利用 `ts_fill` 可以在财报发布之间保持 EPS 数据的连续性，及时捕捉盈利变化，`ts_fill` 有助于通过填充缺失值来对齐数据集，方便多因子模型的构建。 在中国市场，技术指标如均线、MACD 等需要连续的数据，`ts_fill` 可以确保这些指标的准确性。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                          [3, 0, 4, 5, -2], 
                          [4, 1, np.nan, np.nan, 1]])
    >>> ts_fill(input)
    array([[2, 3, 3, 3, np.inf], 
           [3, 0, 4, 5, -2], 
           [4, 1, 1, 1, 1]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A
    
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_fill_c(A_data, V_data, n, d)
    return V

def ts_mean_exp(np.ndarray[numeric, ndim=2] A, int days, double exp_factor):
    """
    Computes the exponentially weighted moving average over a specified window size.

    Calculates the weighted mean where weights decrease exponentially.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).
    exp_factor : float
        Exponential decay factor (0 < exp_factor < 1).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the weighted means are stored.

    Notes
    -----
    与简单移动平均（SMA）相比，EWMA 减少了滞后，因为它对最近的数据点赋予更大权重。这使交易者能够更快地对市场变化作出反应。通过调整 `exp_factor`，交易者可以控制EWMA的敏感度。较小的 `exp_factor`（接近0）会使平均值更平滑，对近期变化反应较慢；通常需要约 :math:`2/{
    \mathrm { exp\_factor }
}` 天，变化才会消散。可以根据市场波动调整 `exp_factor`。在高波动时期，较小的 `exp_factor` 可帮助平滑过度的噪声；在稳定时期，较大的 `exp_factor` 可使平均值更具响应性。

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[2, 3, np.nan, 3, np.inf], 
                          [3, 0, 4, 5, -2], 
                          [4, 1, np.nan, np.nan, 1]])
    >>> ts_mean_exp(input, 3, 0.2)
    array([[2.0000, 2.5556, np.nan, 3.0000, np.nan], 
           [3.0000, 1.3333, 2.4262, 3.3607, 1.8689], 
           [4.0000, 2.3333, np.nan, np.nan, 1.0000]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef size_t n = A_c.shape[0]
    cdef size_t d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] V = np.empty((n, d), dtype=np.float64)

#Obtain pointers to data
    cdef const double* A_ptr = &A_c[0, 0]
    cdef double* V_ptr = &V[0, 0]

#Call the C++ function
    ts_mean_exp_c(A_ptr, V_ptr, n, d, days, exp_factor)
    return V

def ts_diff(np.ndarray[numeric, ndim=2] A, int order):
    """
    Computes the n-th order difference along each row.

    Calculates the difference between successive elements, repeated `order` times.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n_rows, n_cols).
    order : int
        The order of difference (e.g., 1 for first difference).

    Returns
    -------
    V : ndarray
        Output array of shape (n_rows, n_cols), where the differences are stored.

    Notes
    -----
    - **平滑和调整信号**：`ts_diff` 可用于捕捉指标的“变化率”。例如，计算一阶差分可以突出每日的指标变化，而高阶差分则可以揭示数据的加速或减速趋势。这对于分析动量（momentum）或反转（reversal）非常有用。
        
        1. **动量 Alpha**：使用 5 天累计收益构建的动量 Alpha, 它同时捕捉近期价格变化和加速度：

            .. math::

                \mathrm{momentum} = \mathtt{ts\_diff}(\mathrm{price}, 1) + \mathtt{ts\_diff}(\mathrm{price}, 2)
     
        2. **反转 Alpha**：利用差分操作发现价格的从负收益到正收益的快速切换：

            .. math::

                \mathrm{reversal} = \mathbb{1}(\mathtt{ts\_diff}(\mathrm{return}, 1) < 0 \, \land \, \mathtt{ts\_diff}(\mathrm{return}, 2) > 0)

    - **放缓或加速 Alphas**：通过将 `ts_diff` 与延迟算子结合，可以调节 alphas 对于价格或交易量变化的反应速度。 这能够平滑 alpha 的响应，同时仍然包含变化率。 例如： 

            .. math::

                \mathrm{alpha} = \mathtt{ts\_delay}(\mathrm{metric}, 1) + \mathtt{ts\_diff}(\mathrm{metric}, 1)

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 3, 2, 3, 4], 
                          [3, 0, 4, 5, -2]])
    >>> ts_diff(input, 1)
    array([[np.nan, 2, -1, 1, 1],
           [np.nan, -3, 4, 1, -7]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef size_t n_rows = A_c.shape[0]
    cdef size_t n_cols = A_c.shape[1]

    cdef np.ndarray[np.double_t, ndim=2] V = np.empty((n_rows, n_cols), dtype=np.float64)

#Obtain pointers to data
    cdef const double* A_ptr = &A_c[0, 0]
    cdef double* V_ptr = &V[0, 0]

#Call the C++ function
    ts_diff_c(A_ptr, V_ptr, n_rows, n_cols, order)
    return V

def ts_corr_binary(np.ndarray[numeric, ndim=2] A, np.ndarray[numeric, ndim=2] B, int days):
    """
    Compute the rolling correlation between two 2D arrays over a specified window.

    Parameters
    ----------
    A : ndarray
        The first input array of shape (n, d).
    B : ndarray
        The second input array of shape (n, d).
    days : int
        The window size for calculating the rolling correlation.

    Returns
    -------
    V : ndarray
        An array of shape (n, d) containing the rolling correlation coefficients.

    Notes
    -----
    - **beta估计**： `ts_corr_binary` 用于滚动贝塔计算，衡量股票收益相对于市场收益的敏感性。通过设置 `A` 为股票收益，`B` 为市场指数收益（如CSI300），可以分析股票在指定窗口内对大盘波动的响应程度。 例子1: 在牛市期间，滚动beta较高的股票可能是动量策略的首选。 例子2: 将股票收益与 VIX 指数收益在 10 天窗口内进行相关性分析，识别对波动率变化敏感的股票

    - **前导-滞后关系**： 此函数有助于识别两个变量之间的前导和滞后关系。例如，将 `A` 设置为股票收益，`B` 设置为行业指数收益， 通过计算股票收益与行业指数收益之间的相关性，识别与行业趋势背离的股票

    - **跨资产相关性**： 如果在某个时间偏移量（滞后或前移）下，滚动相关性达到最大值，说明这个偏移量反映了前导变量与滞后变量的时间差。 比较债券收益率变化（`A`）与股票指数收益（`B`）的时间关系，观察债券收益率是否领先于股票市场表现。 

    - **alpha 信号稳定性**： 时间相关性可以作为 alpha 信号的稳定性指标。例如，股票近期收益与其历史模式的高相关性可能表明信号的延续性，是延续策略的潜在候选。

    - **市场环境分析**：相关性指标有助于评估市场环境的变化。例如，个股与指数之间的相关性上升通常表明风险偏好下降，有助于市场时机策略。

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 3, 2, 3, 4], [3, 0, 4, 5, -2]])
    >>> B = np.array([[2, 3, 3, 4, 3], [2, 1, 5, 4, -2]])
    >>> ts_corr_binary(A, B, days=3)
    array([[np.nan, np.nan, 0.8660, 0.5000, 0],
           [np.nan, np.nan, 0.8462, 0.9078, 0.9651]])
    """
#Ensure input arrays are NumPy arrays of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    if not B.flags['C_CONTIGUOUS']:
        B = np.ascontiguousarray(B)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] B_c
    if B.dtype != np.float64:
        B_temp = np.array(B, dtype=np.float64, copy=True)
        B_c = B_temp
    else:
        B_c = B

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]

    cdef np.ndarray[np.double_t, ndim=2] V = np.empty((n, d), dtype=np.float64)

#Obtain pointers to data
    cdef const double* A_ptr = &A_c[0, 0]
    cdef const double* B_ptr = &B_c[0, 0]
    cdef double* V_ptr = &V[0, 0]

#Call the C++ function
    ts_corr_binary_c(A_ptr, B_ptr, V_ptr, n, d, days)

    return V

def ts_sum(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling sum over a specified window size.

    Calculates the sum of values within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the sums are stored.

    Notes
    -----

    - **趋势捕捉的滚动聚合**：`ts_sum` 非常适合滚动聚合操作，用于捕捉特定周期内的趋势。

        - 例如，对 5 天收益率求和可以揭示短期表现。

            .. math::

                \mathrm{alpha} = \mathtt{ts\_sum}(\mathrm{return}, 5)

        - 通过对近期负收益率求和检测短期反转：

            .. math::

                \mathrm{alpha} = \mathtt{ts\_sum}(\mathrm{return}[\mathrm{return} < 0], 3)

    - **信号增强**： 通过对时间窗口求和，可以放大微弱但持续的信号。这特别适用于检测具有持续模式的股票（例如价格稳定上涨或成交量逐步累积）。 
        
        - 对过去 10 天的交易量求和以识别持续被大量资金关注的股票：

            .. math::

                \mathrm{alpha} = \mathtt{ts\_sum}(\mathrm{volume}, 10)

        - 对绝对收益率在一个窗口内求和，突出近期波动性显著的股票：

            .. math::

                \mathrm{alpha} = \mathtt{ts\_sum}(|\mathrm{return}|, 5)

    - **组合性**：该操作符可以与其他操作符结合使用以构建复杂的 alpha。

        - 将累计收益与滞后差分相结合，生成更平滑的信号：

            .. math::

                \mathrm{alpha} = \mathtt{ts\_sum}(\mathrm{return}, 5) + \mathtt{ts\_diff}(\mathrm{return}, 1)


    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 3, 2, 3, 4], 
                          [3, 0, np.nan, 5, -2]])
    >>> ts_sum(input, 3)
    array([[1, 4, 6, 8, 9],
           [3, 3, 3, 5, 3]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_sum_c(A_data, V_data, n, d, days)
    return V



def ts_std(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling standard deviation over a specified window size.

    Calculates the standard deviation within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the standard deviations are stored.

    Notes
    -----

    - `ts_std` 是测量金融时间序列数据波动率的基础工具。 标准差是风险的替代指标。收益率标准差较高的股票通常表示风险较大，而较低的值表明更稳定的特性。 在投资组合管理中，`ts_std` 可用于比较不同资产的风险特征，并调整配置以平衡风险。 在主板市场，低波动率股票通常代表更稳定的蓝筹股，适合长期投资者。

        - **基于波动率的筛选Alpha**： 投资于过去20天波动率最低的股票，因为它们可能提供更稳定的回报。

            .. math::

                \mathrm{alpha} = \mathbb{1}[\mathrm{cs\_rank}(-\mathtt{ts\_std}(\mathrm{return}, 20)> θ)]

        - **风险调整收益 Alpha**： 偏好具有强劲动量但经波动率调整后的股票, 类似于过去10天的夏普比率, 偏好高收益且低波动率的股票。

            .. math::

                \mathrm{alpha} = {\mathtt{ts\_mean}(\mathrm{return}, 10)}/{\mathtt{ts\_std}(\mathrm{return}, 10)}

    - 将 `ts_std` 与其他时间序列算子（如 `ts_mean`）结合，可以帮助识别趋势或反转。 低波动率时期可能会先于重大价格变动。通过监测不同窗口期的标准差，交易者可以识别潜在的突破机会。 相反，高波动率可能预示着市场不确定性或潜在的反转。 

        - **波动率突破 Alpha**： 识别近期波动率显著超过历史波动率的股票，表明潜在的突破。 比值显著大于1表示波动率正在增加。

            .. math::

                \mathrm{volatility\_breakout} = {\mathtt{ts\_std}(\mathrm{return}, 5)}/{\mathtt{ts\_std}(\mathrm{return}, 20)}

        - **波动率差异检测Alpha**：识别跨资产之间波动率的差异，以利用相对错误定价。 波动率相对于市场指数更高的股票可能对消息反应过度。

            .. math::

                \mathrm{alpha} = \mathtt{ts\_std}(\mathrm{return}_{\mathrm{stock}}, N) - \mathtt{ts\_std}(\mathrm{return}_{\mathrm{index}}, N)

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 3, 2, 3, 4], 
                          [3, 0, np.nan, 5, -2]])
    >>> ts_std(input, 3)
    array([[np.nan, np.nan, 1.0000, 0.5774, 1.0000],
           [np.nan, np.nan, np.nan, 3.5355, 4.9497]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A

    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_std_c(A_data, V_data, n, d, days)
    return V


def ts_skew(np.ndarray[numeric, ndim=2] A, int days):
    """
    Computes the rolling skewness over a specified window size.

    Calculates the skewness within a rolling window for each element.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the skewness values are stored.

    Notes
    -----
    - `ts_skew` 函数计算指定窗口内指标的偏度，提供了关于分布不对称性的洞察。正偏度表示分布具有较长的右尾, 负偏度表示分布具有较长的左尾, 将偏度纳入动量策略，可增强对持续正趋势的检测。

        - **基于偏度的动量 Alpha**：对过去 `n` 天内具有正偏度的资产做多，捕捉上涨趋势明显的股票，特别是在牛市或政策利好时期。

            .. math::

                \mathrm{alpha} = \mathbb{1}( \mathtt{ts\_skew}(\mathrm{return}, n) > k )

        - **偏度调整的仓位大小**：根据负偏度的大小反比调整仓位大小, 在波动性大的市场中，如创业板股票，动态调整仓位以控制风险。

            .. math::

                \mathrm{position} = {1}/({1 + |\mathtt{ts\_skew}(\mathrm{return}, n)|})


    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 2, 2, 3], 
                          [2, 3, 3, 2]])
    >>> ts_skew(input, 3)
    array([[np.nan, np.nan, -0.7071, 0.7071],
           [np.nan, np.nan, -0.7071, -0.7071]])
    """
#Ensure input array is C - contiguous and of type np.float64
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_skew_c(A_data, V_data, n, d, days)
    return V

        
def ts_rank(np.ndarray[numeric, ndim=2] A, int days):
    """
    Assigns ranks over a rolling window for each element.

    Calculates the rank of the current value within a rolling window, scaling the ranks between 1 and 2.

    Parameters
    ----------
    A : ndarray
        Input array of shape (n, d).
    days : int
        Window size (number of days).

    Returns
    -------
    V : ndarray
        Output array of shape (n, d), where the ranks are stored.

    Notes
    -----
    - **捕捉相对历史位置**： `ts_rank` 函数评估当前值在指定窗口内相对于自身历史值的位置。`ts_rank` 实现了跨时间的数据归一化，减少了极端值的影响，使不同时间段更具可比性。这在处理非平稳时间序列数据时尤其有用。在A股市场中, `ts_rank` 在以下场景中存在优势:  

        1. 新上市的股票由于历史数据较短，`ts_rank` 有助于在有限的数据窗口内评估其相对表现。

        2. 由于涨跌停限制，价格可能连续多日无法交易。使用 `ts_rank` 可以在价格变化受限的情况下，通过其他指标（如成交量、换手率）的排名来辅助决策。

        3. 小市值股票可能更容易出现极端波动。通过时间序列排名，可以更好地捕捉这些波动带来的交易机会。

    - **收益率/波动率/成交量/价格因子**：对过去 N 天的收益率进行排名以捕捉动量 / 利用波动率的时间序列排名来检测突破期 / 识别相对于历史数据出现异常交易量的股票 / 当当前价格相对于其历史排名较低时，检测潜在的反转

    Examples
    --------
    >>> import numpy as np
    >>> input = np.array([[1, 2, 3, 4], 
                          [4, 5, 6, np.nan]])
    >>> ts_rank(input, 3)
    array([[1.500, 2.000, 2.000, 2.000],
           [1.500, 2.000, 2.000, np.nan]])
    """
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_c
    if A.dtype != np.float64:
        A_temp = np.array(A, dtype=np.float64, copy=True)
        A_c = A_temp
    else:
        A_c = A
    
    cdef int n = A_c.shape[0]
    cdef int d = A_c.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = np.empty((n, d), dtype=np.float64)

    cdef double* A_data = &A_c[0, 0]
    cdef double* V_data = &V[0, 0]

#Call the C++ function using the data pointers
    ts_rank_c(A_data, V_data, n, d, days)

    return V