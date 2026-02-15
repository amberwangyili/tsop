#ifndef TSOP_BASIC_H
#define TSOP_BASIC_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tsop {

/**
 * @brief Computes column-wise z-scores for stock values.
 *
 * The function calculates the z-score for each value in each column (day),
 * standardizing the data by subtracting the mean and dividing by the standard deviation.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the z-scores will be stored.
 * @param n Number of rows (stocks)
 * @param d Number of columns (days)
 *
 * @note
 * - **Column-wise Processing**: Iterates over each column, representing each day.
 * - **Value Collection**: Collects finite and non-NaN values for each column. Infinite values cause
 * the entire column to be set to NaN.
 * - **Mean and Standard Deviation**: Computes the mean and standard deviation of the collected
 * values.
 * - **Z-Score Calculation**: Calculates the z-score for each valid value. Positions corresponding
 * to invalid inputs (NaN or infinite) are set to NaN in the output.
 * - **Insufficient Data**: If there are less than two valid data points, the output for that column
 * is set to NaN.
 * - **Zero Standard Deviation**: If the standard deviation is zero, the output for that column is
 * set to NaN.
 *
 * @example
 * ```cpp
 * double input[3][5] = {
 *   {2, 3, NAN, 3, INFINITY},
 *   {3, 0, 4, 5, -2},
 *   {4, 1, NAN, NAN, 1}
 * };
 * double output[3][5];
 * cs_zscore_c(&input[0][0], &output[0][0], 3, 5);
 * // Expected output (approximate values):
 * // output = {
 * //   { -1.0,  1.2247, NAN, -1.0, NAN },
 * //   {  0.0, -1.2247, 0.0, 1.0,  -1.0 },
 * //   {  1.0,  0.0,    NAN, NAN, 1.0 }
 * // };
 * ```
 */
template<typename T>
inline void cs_zscore_c(const T* A, T* V, int n, int d) {
    for (int j = 0; j < d; ++j) {  // Iterate over columns (days)
        // Collect valid values in column j
        std::vector<double> values;
        bool has_infinite = false;
        for (int i = 0; i < n; ++i) {
            double val = A[i * d + j];
            if (std::isfinite(val) || std::isnan(val)) {
                if (!std::isnan(val) && std::isfinite(val)) {
                    values.push_back(val);
                }
            } else {
                has_infinite = true;
                V[i * d + j] = NAN;
            }
        }
        // If there are infinite values, set all outputs in this column to NaN
        if (has_infinite || values.empty()) {
            for (int i = 0; i < n; ++i) {
                V[i * d + j] = NAN;
            }
            continue;
        }
        // Compute mean
        double sum = 0.0;
        for (double val : values) {
            sum += val;
        }
        double mean = sum / values.size();

        // Compute standard deviation using N (values.size())
        double sq_sum = 0.0;
        for (double val : values) {
            double diff = val - mean;
            sq_sum += diff * diff;
        }
        double stddev = std::sqrt(sq_sum / values.size());

        // Handle case where stddev is zero
        if (stddev == 0.0) {
            // Set z-scores to NaN when standard deviation is zero
            for (int i = 0; i < n; ++i) {
                V[i * d + j] = NAN;
            }
            continue;
        }

        // Compute z-scores for valid values
        for (int i = 0; i < n; ++i) {
            double val = A[i * d + j];
            if (!std::isnan(val) && std::isfinite(val)) {
                V[i * d + j] = (val - mean) / stddev;
            } else {
                V[i * d + j] = NAN;
            }
        }
    }
}

/**
 * @brief Applies column-wise Winsorization to stock values.
 *
 * The function limits extreme data points in each column (day) by either replacing them
 * with the nearest threshold values (Winsorization) or removing them (setting to NaN),
 * based on the specified percentile.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the Winsorized values will be stored.
 * @param n Number of rows (stocks)
 * @param d Number of columns (days)
 * @param filter_percentile The percentile for filtering (e.g., 0.05 for 5%)
 * @param remove_extreme If true, extreme values are set to NaN; if false, they are replaced with
 * threshold values.
 *
 * @note
 * - **Column-wise Processing**: Iterates over each column, representing each day.
 * - **Value Collection**: Collects finite and non-NaN values for each column.
 * - **Sorting**: Sorts the collected values in ascending order.
 * - **Threshold Calculation**: Calculates lower and upper thresholds based on the specified
 * filter_percentile.
 * - **Winsorization**: Values beyond the thresholds are either replaced with the threshold values
 * or set to NaN, depending on `remove_extreme`.
 * - **Invalid Inputs**: Positions corresponding to invalid inputs (NaN or infinite) are set to NaN
 * in the output.
 *
 * @example
 * ```cpp
 * double input[3][5] = {
 *   {2, 3, NAN, 3, INFINITY},
 *   {3, 0, 4, 5, -2},
 *   {4, 1, NAN, NAN, 1}
 * };
 * double output[3][5];
 * double filter_percentile = 0.05; // 5% percentile
 * bool remove_extreme = false;
 * cs_winsor_c(&input[0][0], &output[0][0], 3, 5, filter_percentile, remove_extreme);
 * // Expected output:
 * // output = {
 * //   {2.0, 2.5, NAN, 3.0, NAN},
 * //   {3.0, 0.0, 4.0, 5.0, -2.0},
 * //   {3.5, 1.0, NAN, NAN, 1.0}
 * // };
 * ```
 */
template<typename T>
inline void cs_winsor_c(const T* A, T* V, int n, int d, double filter_percentile,
                        bool remove_extreme) {
    for (int j = 0; j < d; ++j) {  // Iterate over columns (days)
        // Collect valid values and their indices
        std::vector<std::pair<double, int>> values;  // Pair of (value, row index)
        for (int i = 0; i < n; ++i) {
            double val = A[i * d + j];
            if (!std::isnan(val) && std::isfinite(val)) {
                values.emplace_back(val, i);
                // Initialize output to input value
                V[i * d + j] = val;
            } else {
                // Set output to NaN for invalid inputs
                V[i * d + j] = NAN;
            }
        }

        int n_values = values.size();
        if (n_values == 0) {
            // All values are invalid for this day
            continue;
        }

        // Sort the values
        std::sort(values.begin(), values.end(),
                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                      return a.first < b.first;
                  });

        // Calculate thresholds
        double lower_p = filter_percentile / 2.0;
        double upper_p = 1.0 - lower_p;

        int lower_idx = static_cast<int>(std::ceil(lower_p * (n_values - 1)));
        int upper_idx = static_cast<int>(std::floor(upper_p * (n_values - 1)));

        // Ensure indices are within bounds
        lower_idx = std::min(std::max(lower_idx, 0), n_values - 1);
        upper_idx = std::min(std::max(upper_idx, 0), n_values - 1);

        // If lower_idx > upper_idx, skip Winsorization for this column
        if (lower_idx > upper_idx) {
            continue;
        }

        double lower_threshold = values[lower_idx].first;
        double upper_threshold = values[upper_idx].first;

        // Apply Winsorization or remove extremes
        for (const auto& p : values) {
            double val = p.first;
            int i = p.second;
            if (val < lower_threshold) {
                if (remove_extreme) {
                    V[i * d + j] = NAN;
                } else {
                    V[i * d + j] = lower_threshold;
                }
            } else if (val > upper_threshold) {
                if (remove_extreme) {
                    V[i * d + j] = NAN;
                } else {
                    V[i * d + j] = upper_threshold;
                }
            }
            // else, value remains unchanged
        }

        // Infinite values are already set to NaN during initialization
    }
}

/**
 * @brief Scales stock values between 1 and 2 for each day.
 *
 * The function linearly scales the values in each column (day) so that the minimum value maps to 1
 * and the maximum to 2.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the scaled values will be stored.
 * @param n Number of rows (stocks)
 * @param d Number of columns (days)
 *
 * @note
 * - **Column-wise Processing**: Iterates over each column, representing each day.
 * - **Value Collection**: Collects valid (finite and non-NaN) values.
 * - **Scaling**: Scales the values linearly between 1 and 2 using the formula:
 *   \f[
 *   V_{ij} = 1.0 + \frac{A_{ij} - \min A_j}{\max A_j - \min A_j}
 *   \f]
 *   If all values are identical, they are assigned a value of 1.5.
 * - **Invalid Inputs**: Positions corresponding to invalid inputs are set to NaN in the output.
 *
 * @example
 * ```cpp
 * double input[3][3] = {
 *   {2, 3, NAN},
 *   {3, 0, 4},
 *   {4, 1, NAN}
 * };
 * double output[3][3];
 * cs_scale_c(&input[0][0], &output[0][0], 3, 3);
 * // Expected output:
 * // output = {
 * //   {1.0, 2.0, NAN},
 * //   {1.5, 1.0, 1.0},
 * //   {2.0, 1.3333, NAN}
 * // };
 * ```
 */
template<typename T>
inline void cs_scale_c(const T* A, T* V, int n, int d) {
    for (int j = 0; j < d; ++j) {  // Iterate over columns (days)
        // Collect the valid (finite and not NaN) values and their indices
        std::vector<std::pair<double, int>> values;  // Pair of (value, row index)
        for (int i = 0; i < n; ++i) {
            double val = A[i * d + j];
            if (!std::isnan(val) && std::isfinite(val)) {
                values.emplace_back(val, i);
            } else {
                // Set output to NaN for invalid inputs
                V[i * d + j] = NAN;
            }
        }

        if (values.empty()) {
            // If all values are NaN or infinite, continue to next day
            continue;
        }

        // Find min and max among valid values
        double min_val = values[0].first;
        double max_val = values[0].first;
        for (const auto& p : values) {
            if (p.first < min_val) min_val = p.first;
            if (p.first > max_val) max_val = p.first;
        }

        double range = max_val - min_val;

        // For the case where max == min, set scaled value to 1.5
        if (range == 0.0) {
            for (const auto& p : values) {
                int i = p.second;
                V[i * d + j] = 1.5;
            }
        } else {
            // Compute scaled values
            for (const auto& p : values) {
                double val = p.first;
                int i = p.second;
                V[i * d + j] = 1.0 + (val - min_val) / range;
            }
        }
    }
}

/**
 * @brief Removes middle percentile values in each column (day).
 *
 * The function sets to NaN the values within the specified middle percentile range,
 * effectively removing the middle values while keeping the extremes.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the filtered values will be stored.
 * @param n Number of rows (stocks)
 * @param d Number of columns (days)
 * @param filter_percentile The percentile range to remove (e.g., 0.2 for middle 20%)
 *
 * @note
 * - **Column-wise Processing**: Iterates over each column, representing each day.
 * - **Value Collection**: Collects valid (finite and non-NaN) values.
 * - **Threshold Calculation**: Calculates lower and upper thresholds to define the middle
 * percentile.
 * - **Filtering**: Sets values within the middle percentile range to NaN.
 * - **Invalid Inputs**: Positions corresponding to invalid inputs are set to NaN in the output.
 *
 * @example
 * ```cpp
 * double input[5][1] = {
 *   {1},
 *   {2},
 *   {3},
 *   {4},
 *   {5}
 * };
 * double output[5][1];
 * double filter_percentile = 0.4; // Remove middle 40%
 * cs_remove_middle_c(&input[0][0], &output[0][0], 5, 1, filter_percentile);
 * // Expected output:
 * // output = {
 * //   {1},
 * //   {2},
 * //   {NAN},
 * //   {4},
 * //   {5}
 * // };
 * ```
 */
template<typename T>
inline void cs_remove_middle_c(const T* A, T* V, int n, int d, double filter_percentile) {
    for (int j = 0; j < d; ++j) {  // Iterate over columns (days)
        // Collect the non-NaN and finite values for this column
        std::vector<std::pair<double, int>> values;  // Pair of (value, row index)
        for (int i = 0; i < n; ++i) {
            double val = A[i * d + j];
            if (!std::isnan(val) && std::isfinite(val)) {
                values.emplace_back(val, i);
            } else {
                // Initialize the output as NaN for invalid inputs
                V[i * d + j] = NAN;
            }
        }

        int n_values = values.size();
        if (n_values == 0) {
            // If all values are NaN or infinite, continue to next day
            continue;
        }

        // Sort the values in ascending order
        std::sort(values.begin(), values.end(),
                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                      return a.first < b.first;
                  });

        // Calculate the percentiles
        double lower_p = (1.0 - filter_percentile) / 2.0;
        double upper_p = 1.0 - lower_p;

        // Calculate the indices corresponding to the percentiles
        int lower_idx = static_cast<int>(lower_p * (n_values - 1));
        int upper_idx = static_cast<int>(upper_p * (n_values - 1));

        double lower_threshold = values[lower_idx].first;
        double upper_threshold = values[upper_idx].first;

        // For each value, decide whether to keep it or set to NaN
        for (int i = 0; i < n; ++i) {
            // If the value was invalid, we already set V[i * d + j] = NAN
            if (std::isnan(A[i * d + j]) || !std::isfinite(A[i * d + j])) {
                continue;
            }

            double val = A[i * d + j];

            if (val < lower_threshold || val > upper_threshold) {
                // Keep the value
                V[i * d + j] = val;
            } else {
                // Remove the middle values
                V[i * d + j] = NAN;
            }
        }
    }
}

/**
 * @brief Assigns ranks to stock values each day, scaling the ranks uniformly between 1 and 2.
 *
 * The function iterates over each column (representing each day) and assigns ranks to the stock
 * values. Infinite values (np.inf, -np.inf) are treated as invalid and excluded.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the scaled ranks will be stored.
 * @param n Number of rows (stocks)
 * @param d Number of columns (days)
 *
 * @note
 * - **Column-wise Processing**: The function iterates over each column (j), representing each day.
 * - **Value Collection**: For each column, it collects non-NaN and finite values along with their
 * row indices. Infinite values are treated as invalid and excluded.
 * - **Sorting and Ranking**: The collected values are sorted in ascending order. Ranks are
 * assigned, handling ties by assigning the same rank to identical values.
 * - **Scaling Ranks**: Ranks are scaled uniformly between 1 and 2. The scaling formula is:
 *   \f[
 *   \text{scaled\_rank} = 1.0 + \frac{(\text{rank} - 1) \times (2.0
 * - 1.0)}{(\text{max\_assigned\_rank} - 1)}
 *   \f]
 *   If there's only one unique rank (i.e., all values are identical), the rank is set to 1.5.
 * - **Assigning Ranks to Output**: The scaled ranks are assigned to the corresponding positions in
 * the output array V. Positions corresponding to NaN or infinite input values are set to NaN in the
 * output.
 *
 * @example
 * ```cpp
 * double input[3][5] = {
 *   {2, 3, NAN, 3, INFINITY},
 *   {3, 0, 4, 5, -2},
 *   {4, 1, NAN, NAN, 1}
 * };
 * double output[3][5];
 * cs_rank_c(&input[0][0], &output[0][0], 3, 5);
 * // Expected output:
 * // output = {
 * //   {1.0000, 2.0000, NAN, 1.0000, NAN},
 * //   {1.5000, 1.0000, 1.5000, 2.0000, 1.0000},
 * //   {2.0000, 1.5000, NAN, NAN, 2.0000}
 * // };
 * ```
 */
template<typename T>
inline void cs_rank_c(const T* A, T* V, int n, int d) {
    for (int j = 0; j < d; ++j) {  // Iterate over columns (days)
        // Collect the non-NaN and finite values for this column
        std::vector<std::pair<double, int>> values;  // Pair of (value, row index)
        for (int i = 0; i < n; ++i) {
            double val = A[i * d + j];
            if (!std::isnan(val) && std::isfinite(val)) {
                values.emplace_back(val, i);
            }
        }

        if (values.empty()) {
            // If all values are NaN or infinite, set output to NaN for all rows
            for (int i = 0; i < n; ++i) {
                V[i * d + j] = NAN;
            }
            continue;
        }

        // Sort the values in ascending order
        std::sort(values.begin(), values.end(),
                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                      return a.first < b.first;
                  });

        // Assign ranks, handling ties
        size_t num_values = values.size();
        std::vector<double> ranks(num_values);
        size_t rank = 1;
        ranks[0] = rank;

        for (size_t k = 1; k < num_values; ++k) {
            if (values[k].first != values[k - 1].first) {
                rank = k + 1;
            }
            ranks[k] = rank;
        }

        // Get the maximum assigned rank
        double max_assigned_rank = ranks.back();

        // Map the ranks to the scale between 1 and 2
        std::vector<double> scaled_ranks(num_values);
        for (size_t k = 0; k < num_values; ++k) {
            if (max_assigned_rank > 1) {
                scaled_ranks[k] = 1.0 + (ranks[k] - 1) * (2.0 - 1.0) / (max_assigned_rank - 1);
            } else {
                scaled_ranks[k] = 1.5;  // Only one unique rank
            }
        }

        // Initialize the entire column to NaN
        for (int i = 0; i < n; ++i) {
            V[i * d + j] = NAN;
        }

        // Assign the scaled ranks to the appropriate rows
        for (size_t k = 0; k < num_values; ++k) {
            int row = values[k].second;
            V[row * d + j] = scaled_ranks[k];
        }
    }
}

/**
 * @brief Converts NaN values to zero.
 *
 * The function replaces all NaN or infinite values in the input array with zero.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the result will be stored.
 * @param n Number of rows
 * @param d Number of columns
 *
 * @note
 * - **Value Replacement**: NaN and infinite values are replaced with zero.
 *
 * @example
 * ```cpp
 * double input[2][3] = {
 *   {NAN, 2, INFINITY},
 *   {3, NAN, 4}
 * };
 * double output[2][3];
 * at_nan2zero_c(&input[0][0], &output[0][0], 2, 3);
 * // Expected output:
 * // output = {
 * //   {0.0, 2.0, 0.0},
 * //   {3.0, 0.0, 4.0}
 * // };
 * ```
 */
template<typename T>
inline void at_nan2zero_c(const T* A, T* V, int n, int d) {
    for (int i = 0; i < n * d; ++i) {
        V[i] = std::isfinite(A[i]) ? A[i] : 0.0;
    }
}

/**
 * @brief Converts zero values to NaN.
 *
 * The function replaces all zero values in the input array with NaN.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the result will be stored.
 * @param n Number of rows
 * @param d Number of columns
 *
 * @note
 * - **Value Replacement**: Zero values are replaced with NaN.
 *
 * @example
 * ```cpp
 * double input[2][3] = {
 *   {0.0, 2, 0.0},
 *   {3, 0.0, 4}
 * };
 * double output[2][3];
 * at_zero2nan_c(&input[0][0], &output[0][0], 2, 3);
 * // Expected output:
 * // output = {
 * //   {NAN, 2.0, NAN},
 * //   {3.0, NAN, 4.0}
 * // };
 * ```
 */
template<typename T>
inline void at_zero2nan_c(const T* A, T* V, int n, int d) {
    for (int i = 0; i < n * d; ++i) {
        V[i] = (A[i] == 0.0) ? NAN : A[i];
    }
}

/**
 * @brief Applies sign-preserving logarithmic transformation.
 *
 * The function computes \f$ \text{sign}(A_{ij}) \times \log(1 + |A_{ij}|) \f$ for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the transformed values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 *
 * @note
 * - **Transformation**: Applies sign-preserving logarithmic transformation.
 * - **Invalid Inputs**: NaN and infinite values result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[2][2] = {
 *   {1.0, -2.0},
 *   {0.0, 3.0}
 * };
 * double output[2][2];
 * at_signlog_c(&input[0][0], &output[0][0], 2, 2);
 * // Expected output:
 * // output = {
 * //   {0.6931, -1.0986},
 * //   {0.0,    1.3863}
 * // };
 * ```
 */
template<typename T>
inline void at_signlog_c(const T* A, T* V, int n, int d) {
    for (int i = 0; i < n * d; ++i) {
        if (std::isnan(A[i]) || std::isinf(A[i])) {
            V[i] = NAN;
        } else {
            V[i] = std::copysign(std::log1p(std::abs(A[i])), A[i]);
        }
    }
}

/**
 * @brief Applies sign-preserving square root transformation.
 *
 * The function computes \f$ \text{sign}(A_{ij}) \times \sqrt{|A_{ij}| + 1} \f$ for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the transformed values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 *
 * @note
 * - **Transformation**: Applies sign-preserving square root transformation.
 * - **Invalid Inputs**: NaN and infinite values result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[2][2] = {
 *   {3.0, -8.0},
 *   {0.0, 15.0}
 * };
 * double output[2][2];
 * at_signsqrt_c(&input[0][0], &output[0][0], 2, 2);
 * // Expected output:
 * // output = {
 * //   {2.0, -3.0},
 * //   {1.0, 4.0}
 * // };
 * ```
 */
template<typename T>
inline void at_signsqrt_c(const T* A, T* V, int n, int d) {
    for (int i = 0; i < n * d; ++i) {
        if (std::isnan(A[i]) || std::isinf(A[i])) {
            V[i] = NAN;
        } else {
            if (A[i] == 0) {
                V[i] = 0;
            } else {
                V[i] = std::copysign(std::sqrt(std::abs(A[i]) + 1), A[i]);
            }
        }
    }
}

/**
 * @brief Normalizes each value by the standard deviation over a time window.
 *
 * The function divides each element by the standard deviation computed over the specified window
 * size.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the normalized values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days) for computing the standard deviation
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Normalization**: Divides the current value by the standard deviation of the window.
 * - **Invalid Inputs**: NaN values in the current position result in NaN in the output.
 * - **Insufficient Data**: If the window has less than two valid data points, the output is set to
 * NaN.
 *
 * @example
 * ```cpp
 * double input[2][5] = {
 *   {1, 2, 3, 4, 5},
 *   {2, 4, 6, 8, 10}
 * };
 * double output[2][5];
 * int days = 3;
 * ts_std_normalized_c(&input[0][0], &output[0][0], 2, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 1.2247, 1.4142, 1.5275},
 * //   {NAN, NAN, 1.2247, 1.4142, 1.5275}
 * // };
 * ```
 */
template<typename T>
inline void ts_std_normalized_c(const T* A, T* V, std::size_t n, std::size_t d,
                                int days) {
    // Incremental: window [max(0,j-days+1), j], output NaN if total count <= 2
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::size_t i = 0; i < n; ++i) {
        const double* row = A + i * d;
        double* v_row = V + i * d;
        double sum = 0.0, sum_sq = 0.0;
        int valid = 0, total = 0;

        for (std::size_t j = 0; j < d; ++j) {
            double val_in = row[j];
            total++;
            if (std::isfinite(val_in)) { sum += val_in; sum_sq += val_in * val_in; valid++; }
            // Remove leaving element
            if (static_cast<int>(j) >= days) {
                double val_out = row[j - days];
                total--;
                if (std::isfinite(val_out)) { sum -= val_out; sum_sq -= val_out * val_out; valid--; }
            }

            double cur = row[j];
            if (std::isnan(cur) || total <= 2 || valid <= 1) { v_row[j] = NAN; continue; }
            double var = (sum_sq - sum * sum / valid) / (valid - 1);
            v_row[j] = (var <= 0.0) ? NAN : cur / std::sqrt(var);
        }
    }
}

/**
 * @brief Computes rolling z-scores over a specified window size.
 *
 * The function calculates the z-score for each element using the mean and standard deviation over a
 * rolling window.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the z-scores will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days) for computing mean and standard deviation
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Z-Score Calculation**: For each position, computes \f$ z = \frac{A_{ij} - \mu}{\sigma} \f$.
 * - **Invalid Inputs**: If insufficient data is available or standard deviation is zero, the output
 * is set to NaN.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_zscore_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 0.0, 0.0, 0.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_zscore_c(const T* A, T* V, std::size_t n, std::size_t d,
                        std::size_t days) {
    // Incremental: window is [j-days, j] inclusive (days+1 elements), output NaN for j < days
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; ++i) {
        const double* row = A + i * d;
        double* v_row = V + i * d;
        double sum = 0.0, sum_sq = 0.0;
        int count = 0;

        for (size_t j = 0; j < d; ++j) {
            // Add entering element
            double val_in = row[j];
            if (!std::isnan(val_in)) { sum += val_in; sum_sq += val_in * val_in; count++; }
            // Remove leaving element (window start was j-days-1 in previous step)
            if (j >= days + 1) {
                double val_out = row[j - days - 1];
                if (!std::isnan(val_out)) { sum -= val_out; sum_sq -= val_out * val_out; count--; }
            }
            if (j < days) { v_row[j] = NAN; continue; }
            double cur = row[j];
            if (std::isnan(cur) || count < 2) { v_row[j] = NAN; continue; }
            double mean = sum / count;
            double var = (sum_sq - sum * sum / count) / (count - 1);
            v_row[j] = (var <= 0.0) ? NAN : (cur - mean) / std::sqrt(var);
        }
    }
}

/**
 * @brief Shifts the time series data backward by a specified number of days.
 *
 * The function delays the data by the specified number of days, inserting NaN at the beginning.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the delayed data will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Number of days to delay
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Shifting**: Moves data backward by `days`, filling the beginning with NaN.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 2;
 * ts_delay_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 1.0, 2.0, 3.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_delay_c(const T* A, T* V, int n, int d, int days) {
    for (int i = 0; i < n; ++i) {
        const double* A_row = A + i * d;
        double* V_row = V + i * d;

        for (int t = 0; t < d; ++t) {
            if (t < days) {
                V_row[t] = NAN;
            } else {
                V_row[t] = A_row[t - days];
            }
        }
    }
}

/**
 * @brief Computes the rolling minimum over a specified window size.
 *
 * The function calculates the minimum value within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the minimum values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: Positions with NaN or insufficient data result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {5, 4, 3, 2, 1}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_min_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {5.0, 4.0, 3.0, 2.0, 1.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_min_c(const T* A, T* V, int n, int d, int days) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {      // Rows
        for (int t = 0; t < d; ++t) {  // Columns
            if (!std::isfinite(A[i * d + t])) {
                V[i * d + t] = A[i * d + t];  // Preserve NaN and infinities
                continue;
            }

            int window_start = std::max(0, t - days + 1);
            double min_value = INFINITY;
            bool has_valid = false;

            for (int k = window_start; k <= t; ++k) {
                double val = A[i * d + k];
                if (std::isnan(val)) {
                    continue;  // Skip NaN values
                }
                if (val < min_value) {
                    min_value = val;
                    has_valid = true;
                }
            }

            if (has_valid) {
                V[i * d + t] = min_value;
            } else {
                V[i * d + t] = NAN;
            }
        }
    }
}

/**
 * @brief Computes the rolling median over a specified window size.
 *
 * The function calculates the median value within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the median values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: Positions with NaN or insufficient data result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 3, 5, 7, 9}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_median_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 3.0, 5.0, 7.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_median_c(const T* A, T* V, int n, int d, int days) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {      // Rows
        for (int t = 0; t < d; ++t) {  // Columns
            if (std::isnan(A[i * d + t])) {
                V[i * d + t] = A[i * d + t];  // Preserve NaN
                continue;
            }
            int window_start = std::max(0, t - days + 1);
            std::vector<double> window_values;
            int valid_count = 0;
            for (int k = window_start; k <= t; ++k) {
                double val = A[i * d + k];
                valid_count++;
                if (std::isfinite(val)) {
                    window_values.push_back(val);
                }
            }
            if (valid_count == days && !window_values.empty()) {
                std::sort(window_values.begin(), window_values.end());
                int n_vals = window_values.size();
                double median;
                if (n_vals % 2 == 0) {
                    median = (window_values[n_vals / 2 - 1] + window_values[n_vals / 2]) / 2.0;
                } else {
                    median = window_values[n_vals / 2];
                }
                V[i * d + t] = median;
            } else {
                V[i * d + t] = NAN;
            }
        }
    }
}

/**
 * @brief Computes the rolling mean over a specified window size.
 *
 * The function calculates the average value within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the mean values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: Positions with NaN or insufficient data result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_mean_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 2.0, 3.0, 4.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_mean_c(const T* A, T* V, int n, int d, int days) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {
        const double* row = A + i * d;
        double* v_row = V + i * d;
        double sum = 0.0;
        int valid = 0;

        for (int t = 0; t < d; ++t) {
            // Add entering element
            double val_in = row[t];
            if (std::isfinite(val_in)) { sum += val_in; valid++; }
            // Remove leaving element
            if (t >= days) {
                double val_out = row[t - days];
                if (std::isfinite(val_out)) { sum -= val_out; valid--; }
            }
            // Output: require full window and current element finite
            if (t < days - 1 || !std::isfinite(val_in) || valid == 0) {
                v_row[t] = NAN;
            } else {
                v_row[t] = sum / valid;
            }
        }
    }
}

/**
 * @brief Computes the rolling maximum over a specified window size.
 *
 * The function calculates the maximum value within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the maximum values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: Positions with NaN or insufficient data result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 3, 5, 7, 9}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_max_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 5.0, 7.0, 9.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_max_c(const T* A, T* V, int n, int d, int days) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {      // Rows
        for (int t = 0; t < d; ++t) {  // Columns
            if (!std::isfinite(A[i * d + t])) {
                V[i * d + t] = A[i * d + t];  // Preserve NaN and infinities
                continue;
            }

            int window_start = std::max(0, t - days + 1);
            double max_value = -INFINITY;
            bool has_valid = false;

            for (int k = window_start; k <= t; ++k) {
                double val = A[i * d + k];
                if (std::isnan(val)) {
                    continue;  // Skip NaN values
                }
                if (val > max_value) {
                    max_value = val;
                    has_valid = true;
                }
            }

            if (has_valid) {
                V[i * d + t] = max_value;
            } else {
                V[i * d + t] = NAN;
            }
        }
    }
}

/**
 * @brief Fills NaN values with the last valid observation.
 *
 * The function propagates the last valid (non-NaN) value forward to fill NaNs.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the filled values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 *
 * @note
 * - **Forward Fill**: NaN values are replaced with the last valid value.
 * - **Initial NaNs**: If initial values are NaN, they remain NaN until a valid value appears.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {NAN, 2.0, NAN, NAN, 5.0}
 * };
 * double output[1][5];
 * ts_fill_c(&input[0][0], &output[0][0], 1, 5);
 * // Expected output:
 * // output = {
 * //   {NAN, 2.0, 2.0, 2.0, 5.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_fill_c(const T* A, T* V, int n, int d) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {
        const double* A_row = A + i * d;
        double* V_row = V + i * d;

        double last_value = NAN;
        for (int t = 0; t < d; ++t) {
            double current_value = A_row[t];
            if (std::isnan(current_value)) {
                V_row[t] = last_value;
            } else {
                V_row[t] = current_value;
                last_value = current_value;
            }
        }
    }
}

/**
 * @brief Computes the exponentially weighted moving average over a specified window size.
 *
 * The function calculates the weighted mean where weights decrease exponentially.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the weighted means will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 * @param exp_factor Exponential decay factor (0 < exp_factor < 1)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Exponential Weights**: Weights decrease exponentially with time.
 * - **Invalid Inputs**: Positions with NaN or insufficient data result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * double exp_factor = 0.5;
 * ts_mean_exp_c(&input[0][0], &output[0][0], 1, 5, days, exp_factor);
 * // Expected output (approximate values):
 * // output = {
 * //   {NAN, NAN, 2.6667, 3.6667, 4.6667}
 * // };
 * ```
 */
template<typename T>
inline void ts_mean_exp_c(const T* A, T* V, int n, int d, int days, double exp_factor) {
    // Precompute weight table once: w[k] = (1-alpha)^k, for k = 0..days-1
    double alpha = exp_factor;
    double decay = 1.0 - alpha;
    std::vector<double> w_table(days);
    w_table[0] = 1.0;
    for (int k = 1; k < days; ++k) w_table[k] = w_table[k-1] * decay;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {
        const double* row = A + i * d;
        double* v_row = V + i * d;

        for (int t = 0; t < d; ++t) {
            if (!std::isfinite(row[t])) { v_row[t] = NAN; continue; }

            int ws = std::max(0, t - days + 1);
            double wsum = 0.0, vsum = 0.0;
            // Most recent element gets weight w_table[0]=1, oldest gets w_table[t-ws]
            for (int k = ws; k <= t; ++k) {
                double val = row[k];
                if (std::isfinite(val)) {
                    double w = w_table[t - k];
                    wsum += w;
                    vsum += w * val;
                }
            }
            v_row[t] = (wsum > 0.0) ? vsum / wsum : NAN;
        }
    }
}

/**
 * @brief Computes the n-th order difference along each row.
 *
 * The function calculates the difference between successive elements, repeated `order` times.
 *
 * @param A Input array, shape (n_rows, n_cols)
 * @param V Output array, shape (n_rows, n_cols), where the differences will be stored.
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param order The order of difference (e.g., 1 for first difference)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row.
 * - **Order of Difference**: Higher-order differences are computed recursively.
 * - **Invalid Inputs**: Positions with NaN result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int order = 1;
 * ts_diff_c(&input[0][0], &output[0][0], 1, 5, order);
 * // Expected output:
 * // output = {
 * //   {NAN, 1.0, 1.0, 1.0, 1.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_diff_c(const T* A, T* V, size_t n_rows, size_t n_cols, int order) {
    // Base case: order == 1
    if (order == 1) {
        for (size_t i = 0; i < n_rows; ++i) {
            V[i * n_cols + 0] = NAN;  // First element is NaN
            for (size_t j = 1; j < n_cols; ++j) {
                double current = A[i * n_cols + j];
                double previous = A[i * n_cols + j - 1];

                if (std::isnan(current) || std::isnan(previous)) {
                    V[i * n_cols + j] = NAN;
                } else {
                    V[i * n_cols + j] = current - previous;
                }
            }
        }
    } else if (order > 1) {
        // Allocate temporary array to hold intermediate results
        T* temp = new T[n_rows * n_cols];

        // Compute the first difference
        ts_diff_c(A, temp, n_rows, n_cols, 1);

        // Compute higher-order differences recursively
        ts_diff_c(temp, V, n_rows, n_cols, order - 1);

        delete[] temp;
    } else {
        // If order <= 0, copy A to V
        std::copy(A, A + static_cast<std::ptrdiff_t>(n_rows * n_cols), V);
    }
}

/**
 * @brief Computes the rolling Pearson correlation coefficient between two time series.
 *
 * The function calculates the correlation coefficient between corresponding values in
 * two input arrays (\f$ A \f$ and \f$ B \f$) over a rolling window of a specified size.
 *
 * @param A Input array 1, shape (n, d)
 * @param B Input array 2, shape (n, d)
 * @param V Output array, shape (n, d), where the correlation coefficients will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days) for computing the correlation
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row (stock).
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Correlation Calculation**:
 *   - Computes the mean of the windowed values for both arrays.
 *   - Computes covariance and variances for the windowed values.
 *   - Calculates the correlation coefficient as:
 *     \f[
 *     \text{corr} = \frac{\text{cov}(A, B)}{\sqrt{\text{var}(A) \cdot \text{var}(B)}}
 *     \f]
 * - **Invalid Inputs**: If insufficient data is available or the variance of either array is zero,
 *   the output is set to NaN or 0, respectively.
 *
 * @example
 * ```cpp
 * double inputA[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double inputB[1][5] = {
 *   {5, 4, 3, 2, 1}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_corr_binary_c(&inputA[0][0], &inputB[0][0], &output[0][0], 1, 5, days);
 * // Expected output (approximate values):
 * // output = {
 * //   {NAN, NAN, -1.0, -1.0, -1.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_corr_binary_c(const T* A, const T* B, T* V, int n, int d, int days) {
    // Incremental correlation using running sums: sumA, sumB, sumA2, sumB2, sumAB
    // Window [t-days+1, t], requires full window
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {
        const double* rowA = A + i * d;
        const double* rowB = B + i * d;
        double* v_row = V + i * d;
        double sA = 0, sB = 0, sA2 = 0, sB2 = 0, sAB = 0;
        int cnt = 0;

        for (int t = 0; t < d; ++t) {
            // Add entering pair
            double a = rowA[t], b = rowB[t];
            if (!std::isnan(a) && !std::isnan(b)) {
                sA += a; sB += b; sA2 += a*a; sB2 += b*b; sAB += a*b; cnt++;
            }
            // Remove leaving pair
            if (t >= days) {
                double ao = rowA[t - days], bo = rowB[t - days];
                if (!std::isnan(ao) && !std::isnan(bo)) {
                    sA -= ao; sB -= bo; sA2 -= ao*ao; sB2 -= bo*bo; sAB -= ao*bo; cnt--;
                }
            }
            if (t - days + 1 < 0 || cnt < 2) { v_row[t] = NAN; continue; }
            // corr = (n*sumAB - sumA*sumB) / sqrt((n*sumA2 - sumA^2)*(n*sumB2 - sumB^2))
            double num = cnt * sAB - sA * sB;
            double dA = cnt * sA2 - sA * sA;
            double dB = cnt * sB2 - sB * sB;
            if (dA <= 0.0 || dB <= 0.0) { v_row[t] = 0.0; }
            else { v_row[t] = num / std::sqrt(dA * dB); }
        }
    }
}

/**
 * @brief Computes the rolling sum over a specified window size.
 *
 * The function calculates the sum of values within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the sums will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row.
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: NaN values are treated as zero in the sum.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_sum_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {6.0, 9.0, 12.0, 15.0, 12.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_sum_c(const T* A, T* V, int n, int d, int days) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {  // Iterate over each row
        // Initialize cumulative sum array for the current row
        std::vector<double> cumulative_sum(d + 1, 0.0);

        for (int j = 0; j < d; ++j) {
            double val = A[i * d + j];
            if (std::isnan(val)) {
                val = 0.0;  // Convert NaN to zero
            }
            cumulative_sum[j + 1] = cumulative_sum[j] + val;
        }

        for (int j = 0; j < d; ++j) {
            int window_start = std::max(0, j - days + 1);
            V[i * d + j] = cumulative_sum[j + 1] - cumulative_sum[window_start];
        }
    }
}

/**
 * @brief Computes the rolling standard deviation over a specified window size.
 *
 * The function calculates the standard deviation within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the standard deviations will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row.
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: Positions with NaN or insufficient data result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_std_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output (approximate values):
 * // output = {
 * //   {NAN, NAN, 1.0, 1.0, 1.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_std_c(const T* A, T* V, size_t n, size_t d, int days) {
    // Incremental: window [j-days+1, j], requires full window
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; ++i) {
        const double* row = A + i * d;
        double* v_row = V + i * d;
        double sum = 0.0, sum_sq = 0.0;
        int count = 0;

        for (size_t j = 0; j < d; ++j) {
            double val_in = row[j];
            if (!std::isnan(val_in)) { sum += val_in; sum_sq += val_in * val_in; count++; }
            if (static_cast<int>(j) >= days) {
                double val_out = row[j - days];
                if (!std::isnan(val_out)) { sum -= val_out; sum_sq -= val_out * val_out; count--; }
            }
            int ws = static_cast<int>(j) - days + 1;
            if (ws < 0 || count <= 1 || std::isnan(row[j])) { v_row[j] = NAN; continue; }
            double var = (sum_sq - sum * sum / count) / (count - 1);
            v_row[j] = std::sqrt(std::max(0.0, var));
        }
    }
}

/**
 * @brief Computes the rolling skewness over a specified window size.
 *
 * The function calculates the skewness within a rolling window for each element.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the skewness values will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row.
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Invalid Inputs**: Positions with insufficient data or zero standard deviation result in NaN.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {1, 2, 3, 4, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_skew_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output (approximate values):
 * // output = {
 * //   {NAN, NAN, 0.0, 0.0, 0.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_skew_c(const T* A, T* V, int n, int d, int days) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; ++i) {
        const double* A_row = A + i * d;
        double* V_row = V + i * d;

        for (int t = 0; t < d; ++t) {
            int window_start = std::max(0, t - days + 1);
            int window_end = t;

            std::vector<double> window_values;
            for (int k = window_start; k <= window_end; ++k) {
                double val = A_row[k];
                if (!std::isnan(val)) {
                    window_values.push_back(val);
                }
            }

            int count = window_values.size();
            if (count < 3) {
                V_row[t] = NAN;
                continue;
            }

            // Compute mean
            double sum = 0.0;
            for (double val : window_values) {
                sum += val;
            }
            double mean = sum / count;

            // Compute standard deviation
            double sq_diff_sum = 0.0;
            for (double val : window_values) {
                double diff = val - mean;
                sq_diff_sum += diff * diff;
            }
            double variance = sq_diff_sum / count;
            double std_dev = std::sqrt(variance);

            if (std_dev == 0.0) {
                V_row[t] = NAN;  // Skewness is undefined when standard deviation is zero
                continue;
            }

            // Compute skewness
            double cube_diff_sum = 0.0;
            for (double val : window_values) {
                double diff = val - mean;
                cube_diff_sum += diff * diff * diff;
            }
            double skewness = (cube_diff_sum / count) / (std_dev * std_dev * std_dev);

            V_row[t] = skewness;
        }
    }
}

/**
 * @brief Assigns ranks over a rolling window for each element.
 *
 * The function calculates the rank of the current value within a rolling window, scaling the ranks
 * between 1 and 2.
 *
 * @param A Input array, shape (n, d)
 * @param V Output array, shape (n, d), where the ranks will be stored.
 * @param n Number of rows
 * @param d Number of columns
 * @param days Window size (number of days)
 *
 * @note
 * - **Row-wise Processing**: Iterates over each row.
 * - **Windowing**: Uses a rolling window of the specified size.
 * - **Ranking**: Assigns ranks, handling ties and scaling between 1 and 2.
 * - **Invalid Inputs**: Positions with NaN or empty window result in NaN in the output.
 *
 * @example
 * ```cpp
 * double input[1][5] = {
 *   {3, 1, 4, 1, 5}
 * };
 * double output[1][5];
 * int days = 3;
 * ts_rank_c(&input[0][0], &output[0][0], 1, 5, days);
 * // Expected output:
 * // output = {
 * //   {NAN, NAN, 1.5, 1.0, 2.0}
 * // };
 * ```
 */
template<typename T>
inline void ts_rank_c(const T* A, T* V, int n, int d, int days) {
    // Sorted contiguous arrays replace std::map for cache-friendly sliding-window rank.
    // Two parallel arrays: vals[] (sorted unique values), cnts[] (occurrence counts).
    // Insert/remove via std::lower_bound + std::memmove; rank = insert position.
    #ifdef _OPENMP
    #pragma omp parallel
    {
    #endif
        // Per-thread buffers, allocated once outside the row loop
        std::vector<double> vals_buf(days);
        std::vector<int>    cnts_buf(days);

    #ifdef _OPENMP
        #pragma omp for schedule(static)
    #endif
        for (int i = 0; i < n; ++i) {
            const T* A_row = A + i * d;
            T*       V_row = V + i * d;
            double*  vp = vals_buf.data();
            int*     cp = cnts_buf.data();
            int nu = 0;     // number of unique values in window

            for (int t = 0; t < d; ++t) {
                // --- Remove leaving element first ---
                if (t >= days) {
                    double val_out = static_cast<double>(A_row[t - days]);
                    if (!std::isnan(val_out)) {
                        double* pos = std::lower_bound(vp, vp + nu, val_out);
                        int idx = static_cast<int>(pos - vp);
                        if (--cp[idx] == 0) {
                            int tail = nu - idx - 1;
                            if (tail > 0) {
                                std::memmove(vp + idx, vp + idx + 1, tail * sizeof(double));
                                std::memmove(cp + idx, cp + idx + 1, tail * sizeof(int));
                            }
                            nu--;
                        }
                    }
                }

                // --- Insert entering element and compute rank ---
                double val_in = static_cast<double>(A_row[t]);
                if (std::isnan(val_in)) { V_row[t] = NAN; continue; }

                double* pos = std::lower_bound(vp, vp + nu, val_in);
                int idx = static_cast<int>(pos - vp);

                if (idx < nu && vp[idx] == val_in) {
                    cp[idx]++;
                } else {
                    int tail = nu - idx;
                    if (tail > 0) {
                        std::memmove(vp + idx + 1, vp + idx, tail * sizeof(double));
                        std::memmove(cp + idx + 1, cp + idx, tail * sizeof(int));
                    }
                    vp[idx] = val_in;
                    cp[idx] = 1;
                    nu++;
                }

                if (nu == 1) { V_row[t] = 1.5; continue; }
                // idx = count of unique values strictly less than val_in
                V_row[t] = 1.0 + static_cast<double>(idx) / (nu - 1);
            }
        }
    #ifdef _OPENMP
    }
    #endif
}
}  // namespace tsop
#endif  // TSOP_BASIC_H
