// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a lazy expression for advanced (NumPy-style) array indexing.
 */

#pragma once

#include "./concepts.hpp"
#include "./layout/for_each.hpp"
#include "./layout/range.hpp"
#include "./stdutil/array.hpp"
#include "./traits.hpp"

#include <array>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda {

  // Forward declarations for detail namespace items used by expr_indexed.
  namespace detail {

    // Check if a type is a range or range::all_t (not ellipsis).
    template <typename T>
    concept IsRangeOrRangeAll = std::same_as<std::remove_cvref_t<T>, range> || std::same_as<std::remove_cvref_t<T>, range::all_t>;

    // Check if a type is valid for slicing (range, range::all_t, ellipsis, or convertible to long).
    template <typename T>
    concept IsSliceArg = IsRangeOrRangeAll<T> || std::same_as<std::remove_cvref_t<T>, ellipsis> || std::convertible_to<T, long>;

    // Check if at least one argument is a range or range::all_t (not all long).
    template <typename... Ts>
    constexpr bool has_range_or_rangeall = (IsRangeOrRangeAll<Ts> || ...) || (std::same_as<std::remove_cvref_t<Ts>, ellipsis> || ...);

    // Extract a subset of indices from a container using a range.
    template <typename Container>
    std::vector<long> extract_indices(Container const &c, range const &rg) {
      auto len  = static_cast<long>(c.size());
      auto last = (rg.last() == -1 && rg.step() > 0) ? len : rg.last();
      range actual_range(rg.first(), last, rg.step());
      std::vector<long> result;
      result.reserve(actual_range.size());
      for (long i : actual_range) { result.push_back(c[static_cast<size_t>(i)]); }
      return result;
    }

  } // namespace detail

  /**
   * @addtogroup av_utils
   * @{
   */

  /**
   * @brief Lazy expression for advanced array indexing with arbitrary index containers.
   *
   * @details This expression type enables NumPy-style advanced indexing where arrays can be
   * indexed with containers of arbitrary integers (e.g., `std::vector<long>`). Unlike regular
   * slicing which returns a view, advanced indexing returns a lazy expression because the
   * selected elements may not be contiguous in memory.
   *
   * The expression stores:
   * - A view of the source array (with any range-based slicing already applied)
   * - Index containers converted to `std::vector<long>`
   * - Information about which dimensions use index containers
   *
   * Example usage:
   * @code{.cpp}
   * nda::array<double, 2> arr(10, 20);
   * std::vector<long> indices = {3, 7, 1};
   * auto expr = arr(indices, nda::range::all);  // Returns expr_indexed
   * nda::array<double, 2> result = expr;        // Materializes the expression
   * @endcode
   *
   * @tparam View Type of the internal view (a `basic_array_view`).
   * @tparam IndexContainers Types of the index containers (typically `std::vector<long>`).
   */
  template <typename View, typename... IndexContainers>
  struct expr_indexed {
    static_assert(sizeof...(IndexContainers) > 0, "expr_indexed requires at least one index container");

    /// Internal view of the source array.
    View v;

    /// Tuple of index containers.
    std::tuple<IndexContainers...> idx;

    /// Array indicating which dimensions of the view are indexed by containers.
    std::array<int, sizeof...(IndexContainers)> indexed_dims;

    /// Number of index containers.
    static constexpr int n_idx_containers = sizeof...(IndexContainers);

    /// Rank of the expression (same as underlying view).
    static constexpr int rank = get_rank<View>;

    /// Value type of the underlying view.
    using value_type = typename View::value_type;

    private:
    // Apply a function to the nth index container at runtime.
    template <typename F>
    [[nodiscard]] auto apply_to_nth(int n, F &&f) const {
      return [&]<size_t... Is>(std::index_sequence<Is...>) {
        using R = std::invoke_result_t<F, std::tuple_element_t<0, decltype(idx)> const &>;
        R result{};
        (void)((static_cast<int>(Is) == n ? (result = f(std::get<Is>(idx)), true) : false) || ...);
        return result;
      }(std::make_index_sequence<n_idx_containers>{});
    }

    // Find which index container (if any) handles dimension d. Returns -1 if none.
    [[nodiscard]] constexpr int find_container_for_dim(int d) const {
      for (int i = 0; i < n_idx_containers; ++i) {
        if (indexed_dims[i] == d) return i;
      }
      return -1;
    }

    // Resolve an index for a given dimension: map through container or pass through.
    [[nodiscard]] long resolve_index(int dim, long input_idx) const {
      int c = find_container_for_dim(dim);
      if (c < 0) return input_idx;
      return apply_to_nth(c, [i = input_idx](auto const &vec) { return static_cast<long>(vec[static_cast<size_t>(i)]); });
    }

    // Element access implementation.
    template <typename Self, size_t... Is, typename... Args>
    [[nodiscard]] static decltype(auto) call_impl(Self &&self, std::index_sequence<Is...>, Args... args) {
      std::array<long, rank> a = {static_cast<long>(args)...};
      return std::forward<Self>(self).v(self.resolve_index(static_cast<int>(Is), a[Is])...);
    }

    public:
    /**
     * @brief Get the shape of the expression.
     * @return `std::array<long, rank>` specifying the shape.
     */
    [[nodiscard]] std::array<long, rank> shape() const {
      std::array<long, rank> result = v.shape();
      for (int i = 0; i < n_idx_containers; ++i) {
        result[indexed_dims[i]] = apply_to_nth(i, [](auto const &c) { return static_cast<long>(c.size()); });
      }
      return result;
    }

    /**
     * @brief Get the total size of the expression.
     * @return Number of elements.
     */
    [[nodiscard]] long size() const { return stdutil::product(shape()); }

    /**
     * @brief Function call operator for element access.
     *
     * @details Accesses the element at the given indices, mapping through the index containers
     * for dimensions that use advanced indexing.
     *
     * @tparam Args Argument types (must all be convertible to long).
     * @param args Multi-dimensional indices.
     * @return Reference to the element at the resolved position.
     */
    template <typename... Args>
      requires(sizeof...(Args) == rank && (std::convertible_to<Args, long> && ...))
    [[nodiscard]] decltype(auto) operator()(Args const &...args) {
      return call_impl(*this, std::make_index_sequence<rank>{}, static_cast<long>(args)...);
    }

    /// Const overload for element access.
    template <typename... Args>
      requires(sizeof...(Args) == rank && (std::convertible_to<Args, long> && ...))
    [[nodiscard]] decltype(auto) operator()(Args const &...args) const {
      return call_impl(*this, std::make_index_sequence<rank>{}, static_cast<long>(args)...);
    }

    /**
     * @brief Assignment operator from an array or expression.
     * @tparam RHS Type satisfying ArrayOfRank<rank>.
     * @param rhs Source array/expression to assign from.
     * @return Reference to this expression.
     */
    template <ArrayOfRank<rank> RHS>
    expr_indexed &operator=(RHS const &rhs) {
      EXPECTS(shape() == rhs.shape());
      nda::for_each(shape(), [this, &rhs](auto const &...args) { (*this)(args...) = rhs(args...); });
      return *this;
    }

    /**
     * @brief Assignment operator from a scalar value.
     * @tparam RHS Scalar type.
     * @param rhs Scalar value to assign to all elements.
     * @return Reference to this expression.
     */
    template <typename RHS>
      requires(is_scalar_for_v<RHS, expr_indexed>)
    expr_indexed &operator=(RHS const &rhs) {
      nda::for_each(shape(), [this, &rhs](auto const &...args) { (*this)(args...) = rhs; });
      return *this;
    }

    /**
     * @brief Function call operator for slicing.
     *
     * @details Slices the expression using ranges, producing a new expr_indexed.
     * For indexed dimensions:
     * - range/range::all: Creates a new index container with the selected subset
     * - long: Removes the dimension (extracts single element from container)
     *
     * For non-indexed dimensions:
     * - Delegates to the underlying view's slicing
     *
     * @tparam Args Argument types (ranges, range::all, or long).
     * @param args Slicing arguments.
     * @return A new expr_indexed with sliced view and containers.
     */
    template <typename... Args>
      requires(sizeof...(Args) == rank && detail::has_range_or_rangeall<Args...> && (detail::IsSliceArg<Args> && ...))
    [[nodiscard]] auto operator()(Args const &...args) const {
      auto args_tuple = std::make_tuple(args...);

      // Build dimension mapping (old -> new) based on which args are not long
      constexpr std::array<bool, rank> dim_kept = {(!std::convertible_to<Args, long>)...};
      std::array<int, rank> old_to_new{};
      int new_idx = 0;
      for (int d = 0; d < rank; ++d) { old_to_new[d] = dim_kept[d] ? new_idx++ : -1; }

      // Build the sliced underlying view
      auto sliced_view = build_sliced_view(args_tuple);

      // Process each index container to build new containers
      auto new_containers_tuple = [&]<size_t... Cs>(std::index_sequence<Cs...>) {
        return std::make_tuple(slice_single_container<Cs>(args_tuple)...);
      }(std::make_index_sequence<n_idx_containers>{});

      // Map indexed dimensions to new indices
      std::array<int, n_idx_containers> new_indexed_dims{};
      for (int c = 0; c < n_idx_containers; ++c) {
        new_indexed_dims[c] = old_to_new[indexed_dims[c]];
      }

      return expr_indexed<decltype(sliced_view), IndexContainers...>{std::move(sliced_view), std::move(new_containers_tuple),
                                                                     new_indexed_dims};
    }

    private:

    // Build sliced view: for indexed dims pass full range, for non-indexed dims pass the slice arg.
    template <typename ArgTuple>
    [[nodiscard]] auto build_sliced_view(ArgTuple const &args_tuple) const {
      return [&]<size_t... Dims>(std::index_sequence<Dims...>) {
        // Convert all arguments to range for uniform handling.
        auto to_range = [&]<size_t D>() -> range {
          using ArgT            = std::remove_cvref_t<std::tuple_element_t<D, ArgTuple>>;
          auto const &arg       = std::get<D>(args_tuple);
          auto const full_range = range(0, v.shape()[D]);
          // Indexed dimensions and range::all_t both map to the full range.
          if constexpr (std::is_same_v<ArgT, range::all_t>) {
            return full_range;
          } else if constexpr (std::is_same_v<ArgT, range>) {
            return find_container_for_dim(static_cast<int>(D)) >= 0 ? full_range : arg;
          } else {
            return find_container_for_dim(static_cast<int>(D)) >= 0 ? full_range : range{static_cast<long>(arg), static_cast<long>(arg) + 1};
          }
        };
        return v(to_range.template operator()<Dims>()...);
      }(std::make_index_sequence<rank>{});
    }

    // Slice container C by dispatching to the correct dimension at runtime.
    template <size_t C, typename ArgTuple>
    [[nodiscard]] std::vector<long> slice_single_container(ArgTuple const &args_tuple) const {
      auto const &container = std::get<C>(idx);
      return [&]<size_t... Dims>(std::index_sequence<Dims...>) {
        auto slice_at_dim = [&]<size_t D>() -> std::vector<long> {
          using ArgT = std::remove_cvref_t<std::tuple_element_t<D, ArgTuple>>;
          if constexpr (std::is_same_v<ArgT, range>) {
            return detail::extract_indices(container, std::get<D>(args_tuple));
          } else {
            return {container.begin(), container.end()};
          }
        };
        std::vector<long> result;
        (void)((static_cast<int>(Dims) == indexed_dims[C] ? (result = slice_at_dim.template operator()<Dims>(), true) : false) || ...);
        return result;
      }(std::make_index_sequence<rank>{});
    }
  };

  // Deduction guide for expr_indexed.
  template <typename View, typename... IndexContainers>
  expr_indexed(View, std::tuple<IndexContainers...>, std::array<int, sizeof...(IndexContainers)>) -> expr_indexed<View, IndexContainers...>;

  /**
   * @brief Get the algebra of an expr_indexed (inherits from the view).
   * @tparam View View type.
   * @tparam IndexContainers Index container types.
   */
  template <typename View, typename... IndexContainers>
  inline constexpr char get_algebra<expr_indexed<View, IndexContainers...>> = get_algebra<View>;

  /**
   * @brief Get the rank of an expr_indexed.
   * @tparam View View type.
   * @tparam IndexContainers Index container types.
   */
  template <typename View, typename... IndexContainers>
  inline constexpr int get_rank<expr_indexed<View, IndexContainers...>> = get_rank<View>;

  /** @} */

  namespace detail {

    // Check if any argument is an IndexContainer.
    template <typename... Ts>
    constexpr bool has_index_container = (IndexContainer<std::remove_cvref_t<Ts>> || ...);

    // Compute which argument positions contain IndexContainers.
    template <typename... Ts>
    constexpr auto compute_indexed_dims() {
      constexpr int n_idx                            = (static_cast<int>(IndexContainer<std::remove_cvref_t<Ts>>) + ...);
      constexpr std::array<bool, sizeof...(Ts)> mask = {IndexContainer<std::remove_cvref_t<Ts>>...};
      std::array<int, n_idx> result{};
      int idx = 0;
      for (int i = 0; i < static_cast<int>(sizeof...(Ts)); ++i) {
        if (mask[i]) result[idx++] = i;
      }
      return result;
    }

    // Replace index containers with range::all for view construction.
    template <typename Arg>
    auto make_view_arg_for_indexed(Arg const &arg) {
      if constexpr (IndexContainer<std::remove_cvref_t<Arg>>) {
        return range::all;
      } else {
        return arg;
      }
    }

    // Convert index container to std::vector<long>.
    template <IndexContainer C>
    std::vector<long> to_index_vector(C const &c) {
      std::vector<long> result(std::ranges::size(c));
      std::ranges::transform(c, result.begin(), [](auto x) { return static_cast<long>(x); });
      return result;
    }

    // Extract index containers from arguments and convert to tuple of vectors.
    template <typename... Args>
    auto filter_and_convert(Args const &...args) {
      auto maybe_convert = []<typename T>(T const &arg) {
        if constexpr (IndexContainer<std::remove_cvref_t<T>>)
          return std::make_tuple(to_index_vector(arg));
        else
          return std::tuple<>{};
      };
      return std::tuple_cat(maybe_convert(args)...);
    }

  } // namespace detail

} // namespace nda
