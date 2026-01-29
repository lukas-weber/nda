// Copyright (c) 2018--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Includes the itertools header and provides some additional utilities.
 */

#pragma once

#include "../traits.hpp"

#include <itertools/itertools.hpp>

#include <ostream>
#include <type_traits>

namespace nda {

  /**
   * @addtogroup layout_utils
   * @{
   */

  /// Using declaration for itertools::range.
  using itertools::range;

  /**
   * @brief Mimics Python's `...` syntax.
   *
   * @details While itertools's `range::all_t` mimics Python's `:`, `ellipsis` mimics Python's `...`. It is repeated as
   * much as necessary to match the number of dimensions of an array/view when used to access elements/slices.
   */
  struct ellipsis : range::all_t {};

  /**
   * @brief Write `nda::range::all_t` to a std::ostream as `_`.
   *
   * @param os Output stream.
   * @return Reference to the output stream.
   */
  inline std::ostream &operator<<(std::ostream &os, range::all_t) noexcept { return os << "_"; }

  /**
   * @brief Write nda::ellipsis to a std::ostream as `___`.
   *
   * @param os Output stream.
   * @return Reference to the output stream.
   */
  inline std::ostream &operator<<(std::ostream &os, ellipsis) noexcept { return os << "___"; }

  /// Constexpr variable that is true if the parameter pack `Args` contains an nda::ellipsis.
  template <typename... Args>
  constexpr bool ellipsis_is_present = is_any_of<ellipsis, std::remove_cvref_t<Args>...>;

  /**
   * @brief A compile-time range with static first, last, and step values.
   *
   * @details This range type allows slicing operations to preserve static extent information in the resulting view.
   * The extent is computed at compile-time as `ceil((Last - First) / Step)` for positive steps.
   *
   * @tparam First The starting index (inclusive).
   * @tparam Last The ending index (exclusive, like Python).
   * @tparam Step The step size (default 1, can be negative for reverse iteration).
   */
  template <long First, long Last, long Step = 1>
  struct static_range {
    static_assert(Step != 0, "Error in nda::static_range: Step cannot be zero");
    static_assert((Step > 0 && Last >= First) || (Step < 0 && Last <= First),
                  "Error in nda::static_range: Invalid range bounds for given step");

    // Compute extent at class level for static_assert validation.
    static constexpr long _extent = (Step > 0) ? (Last - First + Step - 1) / Step : (First - Last - Step - 1) / (-Step);
    static_assert(_extent <= 255, "Error in nda::static_range: Extent exceeds maximum of 255");

    /// First index of the range.
    static constexpr long first_value = First;

    /// Last index (exclusive) of the range.
    static constexpr long last_value = Last;

    /// Step of the range.
    static constexpr long step_value = Step;

    /// Compute the size/extent at compile time.
    static constexpr long extent() noexcept { return _extent; }

    /// Get first index (for interface compatibility with runtime range).
    [[nodiscard]] constexpr long first() const noexcept { return First; }

    /// Get last index (for interface compatibility with runtime range).
    [[nodiscard]] constexpr long last() const noexcept { return Last; }

    /// Get step (for interface compatibility with runtime range).
    [[nodiscard]] constexpr long step() const noexcept { return Step; }

    /// Get size (for interface compatibility with runtime range).
    [[nodiscard]] constexpr long size() const noexcept { return _extent; }
  };

  /// Constexpr variable that is true if type T is an nda::static_range.
  template <typename T>
  constexpr bool is_static_range_v = false;

  /// @cond
  template <long First, long Last, long Step>
  constexpr bool is_static_range_v<static_range<First, Last, Step>> = true;
  /// @endcond

  /**
   * @brief Write nda::static_range to a std::ostream.
   *
   * @tparam First First index.
   * @tparam Last Last index.
   * @tparam Step Step size.
   * @param os Output stream.
   * @return Reference to the output stream.
   */
  template <long First, long Last, long Step>
  inline std::ostream &operator<<(std::ostream &os, static_range<First, Last, Step>) noexcept {
    return os << "static_range<" << First << ", " << Last << ", " << Step << ">";
  }

  /**
   * @brief Constexpr variable that is true if the type `T` is either an `nda::range`, an `nda::range::all_t`, an
   * nda::ellipsis, or an nda::static_range.
   */
  template <typename T>
  constexpr bool is_range_or_ellipsis = is_any_of<std::remove_cvref_t<T>, range, range::all_t, ellipsis> || is_static_range_v<std::remove_cvref_t<T>>;

  /** @} */

} // namespace nda
