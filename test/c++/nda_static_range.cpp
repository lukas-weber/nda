// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

// Test compile-time extent computation.
TEST(NDA, StaticRangeExtentComputation) {
  static_assert(nda::static_range<0, 10>::extent() == 10);
  static_assert(nda::static_range<2, 5>::extent() == 3);
  static_assert(nda::static_range<0, 10, 2>::extent() == 5);
  static_assert(nda::static_range<0, 10, 3>::extent() == 4);  // ceil(10/3) = 4
  static_assert(nda::static_range<1, 10, 3>::extent() == 3);  // (10-1+3-1)/3 = 3
  static_assert(nda::static_range<0, 0>::extent() == 0);      // empty range
  static_assert(nda::static_range<5, 5>::extent() == 0);      // empty range

  // Negative step
  static_assert(nda::static_range<9, 0, -1>::extent() == 9);
  static_assert(nda::static_range<10, 0, -2>::extent() == 5);

  // Member accessors
  static_assert(nda::static_range<2, 7, 2>::first_value == 2);
  static_assert(nda::static_range<2, 7, 2>::last_value == 7);
  static_assert(nda::static_range<2, 7, 2>::step_value == 2);

  EXPECT_TRUE(true); // Compile-time test passed
}

// Test basic static_range usage with dynamic array.
TEST(NDA, StaticRangeBasicUsage) {
  nda::array<int, 2> A(10, 10);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) A(i, j) = i * 10 + j;

  // Slice with static_range
  auto v = A(nda::static_range<2, 5>{}, nda::range::all);

  // Check shape
  EXPECT_EQ(v.shape()[0], 3); // 5 - 2 = 3
  EXPECT_EQ(v.shape()[1], 10);

  // Check values
  EXPECT_EQ(v(0, 0), 20);
  EXPECT_EQ(v(2, 9), 49);
}

// Test static extent preservation with stack_array.
TEST(NDA, StaticRangePreservesStaticExtent) {
  nda::stack_array<int, 10, 10> A;
  A = 1;

  auto v = A(nda::static_range<0, 5>{}, nda::range::all);

  // Verify that static extent is preserved
  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 5, "Static extent from static_range should be 5");
  static_assert(layout_t::static_extents[1] == 10, "Static extent from range::all should be preserved");

  EXPECT_EQ(v.shape()[0], 5);
  EXPECT_EQ(v.shape()[1], 10);
}

// Test static_range with step.
TEST(NDA, StaticRangeWithStep) {
  nda::array<int, 1> A(10);
  for (int i = 0; i < 10; ++i) A(i) = i;

  auto v = A(nda::static_range<0, 10, 2>{});

  EXPECT_EQ(v.shape()[0], 5); // (10 - 0 + 2 - 1) / 2 = 5
  EXPECT_EQ(v(0), 0);
  EXPECT_EQ(v(1), 2);
  EXPECT_EQ(v(2), 4);
  EXPECT_EQ(v(3), 6);
  EXPECT_EQ(v(4), 8);

  // Verify static extent is preserved
  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 5, "Static extent should be 5");
}

// Test static_range with negative step.
TEST(NDA, StaticRangeNegativeStep) {
  nda::array<int, 1> A(10);
  for (int i = 0; i < 10; ++i) A(i) = i;

  auto v = A(nda::static_range<9, 0, -1>{});

  EXPECT_EQ(v.shape()[0], 9);
  EXPECT_EQ(v(0), 9);
  EXPECT_EQ(v(1), 8);
  EXPECT_EQ(v(8), 1);

  // Verify static extent is preserved
  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 9, "Static extent should be 9");
}

// Test combined slicing with static_range, long, and range::all.
TEST(NDA, StaticRangeCombinedSlicing) {
  nda::array<int, 3> A(10, 10, 10);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      for (int k = 0; k < 10; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  // Mix static_range with long index and range::all
  auto v = A(nda::static_range<1, 4>{}, 5, nda::range::all);

  EXPECT_EQ(v.extent(0), 3);
  EXPECT_EQ(v.extent(1), 10);
  EXPECT_EQ(v(0, 0), 150); // A(1, 5, 0)
  EXPECT_EQ(v(2, 9), 359); // A(3, 5, 9)

  // Verify static extents
  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 3, "Static extent from static_range should be 3");
  static_assert(layout_t::static_extents[1] == 0, "Extent from dynamic array should be 0");
}

// Test static_range with stack_array preserving all static extents.
TEST(NDA, StaticRangeStackArrayFullStatic) {
  nda::stack_array<int, 10, 8, 6> A;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 6; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  // All static slicing
  auto v = A(nda::static_range<2, 7>{}, nda::range::all, nda::static_range<1, 4>{});

  // Verify all static extents are preserved
  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 5, "Static extent from static_range<2,7> should be 5");
  static_assert(layout_t::static_extents[1] == 8, "Static extent from range::all should be preserved");
  static_assert(layout_t::static_extents[2] == 3, "Static extent from static_range<1,4> should be 3");

  EXPECT_EQ(v.shape()[0], 5);
  EXPECT_EQ(v.shape()[1], 8);
  EXPECT_EQ(v.shape()[2], 3);
}

// Test dynamic range loses static extent.
TEST(NDA, DynamicRangeLosesStaticExtent) {
  nda::stack_array<int, 10, 10> A;
  A = 1;

  // Using dynamic range should lose static extent
  auto v = A(nda::range(0, 5), nda::range::all);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 0, "Dynamic range should result in dynamic extent");
  static_assert(layout_t::static_extents[1] == 10, "Static extent from range::all should be preserved");

  EXPECT_EQ(v.shape()[0], 5);
  EXPECT_EQ(v.shape()[1], 10);
}

// Test layout properties with static_range step 1 on non-fastest dimension.
// Note: static_range does NOT preserve contiguity even with step=1, because it changes
// the extent while keeping the original stride. Only range::all preserves contiguity.
TEST(NDA, StaticRangeLayoutPropertiesStep1) {
  nda::array<int, 2> A(10, 10);
  A = 1;

  // static_range on the first dimension breaks contiguity
  auto v = A(nda::static_range<2, 7>{}, nda::range::all);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;

  // Should NOT be contiguous because static_range changes the extent while keeping stride
  static_assert(!nda::has_contiguous(layout_t::layout_prop), "static_range breaks contiguity");
}

// Test layout properties with static_range step != 1.
TEST(NDA, StaticRangeLayoutPropertiesStepN) {
  nda::array<int, 2> A(10, 10);
  A = 1;

  // static_range with step != 1 loses contiguity
  auto v = A(nda::static_range<0, 10, 2>{}, nda::range::all);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;

  // Should lose contiguous property due to striding
  static_assert(!nda::has_contiguous(layout_t::layout_prop), "Layout should not be contiguous with step != 1");
}

// Test bounds checking for static_range.
TEST(NDA, StaticRangeBoundsChecking) {
  nda::array<int, 1> A(5);
  A = 1;

  // Valid static_range - should not throw
  EXPECT_NO_THROW(A(nda::static_range<0, 5>{}));
  EXPECT_NO_THROW(A(nda::static_range<2, 5>{}));

  // Out of bounds static_range - should throw
  EXPECT_THROW(A(nda::static_range<0, 6>{}), std::runtime_error);
  EXPECT_THROW(A(nda::static_range<3, 7>{}), std::runtime_error);
}

// Test instance methods of static_range.
TEST(NDA, StaticRangeMethods) {
  nda::static_range<2, 10, 3> r;

  EXPECT_EQ(r.first(), 2);
  EXPECT_EQ(r.last(), 10);
  EXPECT_EQ(r.step(), 3);
  EXPECT_EQ(r.size(), 3); // (10-2+3-1)/3 = 3
}

// Test static_range with ellipsis.
TEST(NDA, StaticRangeWithEllipsis) {
  nda::array<int, 3> A(10, 8, 6);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 6; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  auto v = A(nda::static_range<2, 5>{}, nda::ellipsis{});

  EXPECT_EQ(v.shape()[0], 3);
  EXPECT_EQ(v.shape()[1], 8);
  EXPECT_EQ(v.shape()[2], 6);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;
  static_assert(layout_t::static_extents[0] == 3, "Static extent from static_range should be 3");
  // Ellipsis on dynamic array results in dynamic extents
  static_assert(layout_t::static_extents[1] == 0, "Ellipsis on dynamic should be 0");
  static_assert(layout_t::static_extents[2] == 0, "Ellipsis on dynamic should be 0");
}

// Test static_range output stream operator.
TEST(NDA, StaticRangeOutputStream) {
  std::ostringstream oss;
  oss << nda::static_range<2, 10, 3>{};
  EXPECT_EQ(oss.str(), "static_range<2, 10, 3>");
}

// ======================================================================================
// Tests for bug fix: static_range slices use correct (original) strides
// ======================================================================================

// Test: Bug fix - static_range slice uses original strides, not recomputed.
// This was a bug where static_range slices were incorrectly marked as contiguous,
// causing strides to be recomputed from extents instead of using original strides.
TEST(NDA, StaticRangeSliceUsesOriginalStrides) {
  nda::array<int, 2> A(3, 3);
  for (int i = 0; i < 9; ++i) A.data()[i] = i;

  auto v = A(nda::static_range<1, 3>{}, nda::static_range<1, 3>{});

  // View should access elements at original strides (3, 1), not recomputed (2, 1)
  EXPECT_EQ(v(0, 0), 4);  // A(1,1) = 1*3 + 1 = 4
  EXPECT_EQ(v(1, 0), 7);  // A(2,1) = 2*3 + 1 = 7  (NOT 6 if strides were 2,1)
  EXPECT_EQ(v(0, 1), 5);  // A(1,2) = 1*3 + 2 = 5
  EXPECT_EQ(v(1, 1), 8);  // A(2,2) = 2*3 + 2 = 8
}

// Test: Non-contiguous slice at idx_map level is fully static with correct strides.
TEST(NDA, StaticRangeNonContiguousFullyStaticIdxMap) {
  using idxm_t = nda::idx_map<2, nda::encode(std::array{10, 10}), nda::C_stride_order<2>, nda::layout_prop_e::contiguous>;
  static_assert(idxm_t::is_fully_static);

  idxm_t idxm{};
  auto [offset, slice] = idxm.slice(nda::static_range<0, 10, 2>{}, nda::range::all);

  using slice_t = decltype(slice);
  static_assert(slice_t::is_fully_static);              // Should still be fully static
  static_assert(slice_t::static_strides[0] == 20);      // 10 * 2
  static_assert(slice_t::static_strides[1] == 1);       // Unchanged
  static_assert(!nda::has_contiguous(slice_t::layout_prop));  // Not contiguous
}

// Test: Rank-3 idx_map slicing with mixed arguments.
TEST(NDA, StaticRangeRank3IdxMapSlicing) {
  using idxm_t = nda::idx_map<3, nda::encode(std::array{10, 8, 6}), nda::C_stride_order<3>, nda::layout_prop_e::contiguous>;
  static_assert(idxm_t::is_fully_static);
  static_assert(idxm_t::static_strides[0] == 48);
  static_assert(idxm_t::static_strides[1] == 6);
  static_assert(idxm_t::static_strides[2] == 1);

  idxm_t idxm{};

  // Test 1: static_range, range::all, static_range - all step=1
  auto [offset1, slice1] = idxm.slice(nda::static_range<2, 7>{}, nda::range::all, nda::static_range<1, 4>{});
  using slice1_t = decltype(slice1);
  static_assert(slice1_t::is_fully_static);
  static_assert(slice1_t::static_extents[0] == 5);
  static_assert(slice1_t::static_extents[1] == 8);
  static_assert(slice1_t::static_extents[2] == 3);
  static_assert(slice1_t::static_strides[0] == 48);
  static_assert(slice1_t::static_strides[1] == 6);
  static_assert(slice1_t::static_strides[2] == 1);

  // Test 2: range::all, static_range with step 2, range::all
  auto [offset2, slice2] = idxm.slice(nda::range::all, nda::static_range<0, 8, 2>{}, nda::range::all);
  using slice2_t = decltype(slice2);
  static_assert(slice2_t::is_fully_static);
  static_assert(slice2_t::static_extents[0] == 10);
  static_assert(slice2_t::static_extents[1] == 4);
  static_assert(slice2_t::static_extents[2] == 6);
  static_assert(slice2_t::static_strides[0] == 48);
  static_assert(slice2_t::static_strides[1] == 12);  // 6 * 2
  static_assert(slice2_t::static_strides[2] == 1);
}

// Test: Rank-4 idx_map slicing.
TEST(NDA, StaticRangeRank4IdxMapSlicing) {
  using idxm_t = nda::idx_map<4, nda::encode(std::array{4, 5, 6, 7}), nda::C_stride_order<4>, nda::layout_prop_e::contiguous>;
  static_assert(idxm_t::is_fully_static);
  // Original strides: (210, 42, 7, 1)
  static_assert(idxm_t::static_strides[0] == 210);
  static_assert(idxm_t::static_strides[1] == 42);
  static_assert(idxm_t::static_strides[2] == 7);
  static_assert(idxm_t::static_strides[3] == 1);

  idxm_t idxm{};
  auto [offset, slice] = idxm.slice(nda::static_range<0, 4, 2>{}, nda::range::all, nda::static_range<1, 5>{}, nda::range::all);

  using slice_t = decltype(slice);
  static_assert(slice_t::is_fully_static);
  static_assert(slice_t::static_extents[0] == 2);
  static_assert(slice_t::static_extents[1] == 5);
  static_assert(slice_t::static_extents[2] == 4);
  static_assert(slice_t::static_extents[3] == 7);
  static_assert(slice_t::static_strides[0] == 420);  // 210 * 2
  static_assert(slice_t::static_strides[1] == 42);
  static_assert(slice_t::static_strides[2] == 7);
  static_assert(slice_t::static_strides[3] == 1);
}

// Test: Rank-3 reducing to Rank-2 with static_range and long index.
TEST(NDA, StaticRangeRank3IdxMapReducingDimension) {
  using idxm_t = nda::idx_map<3, nda::encode(std::array{10, 8, 6}), nda::C_stride_order<3>, nda::layout_prop_e::contiguous>;
  idxm_t idxm{};

  auto [offset, slice] = idxm.slice(nda::static_range<2, 5>{}, 3L, nda::range::all);

  using slice_t = decltype(slice);
  static_assert(slice_t::rank() == 2);
  static_assert(slice_t::is_fully_static);
  static_assert(slice_t::static_extents[0] == 3);
  static_assert(slice_t::static_extents[1] == 6);
  static_assert(slice_t::static_strides[0] == 48);
  static_assert(slice_t::static_strides[1] == 1);

  EXPECT_EQ(offset, idxm(2, 3, 0));
}

// Test: Dynamic range loses fully static property.
TEST(NDA, StaticRangeDynamicRangeLosesFullyStatic) {
  using idxm_t = nda::idx_map<2, nda::encode(std::array{10, 10}), nda::C_stride_order<2>, nda::layout_prop_e::contiguous>;
  idxm_t idxm{};

  auto [offset, slice] = idxm.slice(nda::static_range<0, 5>{}, nda::range(0, 5));

  using slice_t = decltype(slice);
  static_assert(!slice_t::is_fully_static);  // Dynamic range -> not fully static
}

// Test: Mixed static_range and range::all on arrays - verifies runtime correctness.
TEST(NDA, StaticRangeMixedSlicingRuntimeCorrectness) {
  nda::stack_array<int, 10, 8, 6> A;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 6; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  // Test with step > 1
  auto v = A(nda::range::all, nda::static_range<0, 8, 2>{}, nda::range::all);
  EXPECT_EQ(v.shape()[0], 10);
  EXPECT_EQ(v.shape()[1], 4);
  EXPECT_EQ(v.shape()[2], 6);

  // Verify values - should access j=0, 2, 4, 6 in original
  EXPECT_EQ(v(0, 0, 0), 0);    // A(0, 0, 0)
  EXPECT_EQ(v(0, 1, 0), 20);   // A(0, 2, 0)
  EXPECT_EQ(v(1, 0, 0), 100);  // A(1, 0, 0)
  EXPECT_EQ(v(1, 1, 1), 121);  // A(1, 2, 1)
}

// ======================================================================================
// Tests for view's layout_t preserving static strides (via basic_layout)
// ======================================================================================

// Test: View's layout_t preserves static strides with static_range slicing.
TEST(NDA, ViewLayoutPreservesStaticStrides) {
  nda::stack_array<int, 10, 8, 6> A;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 6; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  // Slice with static_range (step=1), range::all, static_range (step=1)
  auto v = A(nda::static_range<2, 7>{}, nda::range::all, nda::static_range<1, 4>{});

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;

  // Verify static extents are preserved
  static_assert(layout_t::static_extents[0] == 5);
  static_assert(layout_t::static_extents[1] == 8);
  static_assert(layout_t::static_extents[2] == 3);

  // Verify static strides are preserved in the view's layout
  static_assert(layout_t::is_fully_static);
  static_assert(layout_t::static_strides[0] == 48);  // Original stride
  static_assert(layout_t::static_strides[1] == 6);   // Original stride
  static_assert(layout_t::static_strides[2] == 1);   // Original stride
}

// Test: View's layout_t with step > 1 has correct scaled strides.
TEST(NDA, ViewLayoutStaticStridesWithStep) {
  nda::stack_array<int, 10, 8, 6> A;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 6; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  // Slice with steps
  auto v = A(nda::range::all, nda::static_range<0, 8, 2>{}, nda::range::all);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;

  // Verify static extents
  static_assert(layout_t::static_extents[0] == 10);
  static_assert(layout_t::static_extents[1] == 4);
  static_assert(layout_t::static_extents[2] == 6);

  // Verify static strides - the middle dimension has step=2, so stride is doubled
  static_assert(layout_t::is_fully_static);
  static_assert(layout_t::static_strides[0] == 48);  // Original stride
  static_assert(layout_t::static_strides[1] == 12);  // 6 * 2 (step)
  static_assert(layout_t::static_strides[2] == 1);   // Original stride
}

// Test: Rank-4 view layout with mixed slicing.
TEST(NDA, ViewLayoutRank4StaticStrides) {
  nda::stack_array<int, 4, 5, 6, 7> A;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 5; ++j)
      for (int k = 0; k < 6; ++k)
        for (int l = 0; l < 7; ++l) A(i, j, k, l) = i * 1000 + j * 100 + k * 10 + l;

  // Slice: static_range with step 2, range::all, static_range with step 1, range::all
  auto v = A(nda::static_range<0, 4, 2>{}, nda::range::all, nda::static_range<1, 5>{}, nda::range::all);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;

  // Verify static extents
  static_assert(layout_t::static_extents[0] == 2);
  static_assert(layout_t::static_extents[1] == 5);
  static_assert(layout_t::static_extents[2] == 4);
  static_assert(layout_t::static_extents[3] == 7);

  // Verify static strides - original strides: (210, 42, 7, 1)
  static_assert(layout_t::is_fully_static);
  static_assert(layout_t::static_strides[0] == 420);  // 210 * 2
  static_assert(layout_t::static_strides[1] == 42);
  static_assert(layout_t::static_strides[2] == 7);
  static_assert(layout_t::static_strides[3] == 1);

  // Verify correct element access
  EXPECT_EQ(v(0, 0, 0, 0), 10);    // A(0, 0, 1, 0)
  EXPECT_EQ(v(1, 0, 0, 0), 2010);  // A(2, 0, 1, 0)
  EXPECT_EQ(v(0, 2, 1, 3), 223);   // A(0, 2, 2, 3)
}

// Test: Dimension reduction preserves static strides for remaining dimensions.
TEST(NDA, ViewLayoutDimensionReductionStaticStrides) {
  nda::stack_array<int, 10, 8, 6> A;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 6; ++k) A(i, j, k) = i * 100 + j * 10 + k;

  // Reduce middle dimension with long index
  auto v = A(nda::static_range<2, 5>{}, 3L, nda::range::all);

  using view_type = decltype(v);
  using layout_t  = typename view_type::layout_t;

  // Should be rank 2
  static_assert(layout_t::rank() == 2);

  // Verify static extents
  static_assert(layout_t::static_extents[0] == 3);
  static_assert(layout_t::static_extents[1] == 6);

  // Verify static strides - original dims 0 and 2 strides
  static_assert(layout_t::is_fully_static);
  static_assert(layout_t::static_strides[0] == 48);
  static_assert(layout_t::static_strides[1] == 1);

  // Verify correct element access
  EXPECT_EQ(v(0, 0), 230);  // A(2, 3, 0)
  EXPECT_EQ(v(1, 2), 332);  // A(3, 3, 2)
}
