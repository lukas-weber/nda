// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <span>
#include <vector>

// ============================================================
// Basic IndexContainer concept tests
// ============================================================

TEST(ExprIndexed, IndexContainerConcept) {
  // Types that should satisfy IndexContainer
  static_assert(nda::IndexContainer<std::vector<long>>);
  static_assert(nda::IndexContainer<std::vector<int>>);
  static_assert(nda::IndexContainer<std::array<long, 3>>);
  static_assert(nda::IndexContainer<std::span<const long>>);

  // Types that should NOT satisfy IndexContainer
  static_assert(!nda::IndexContainer<long>);
  static_assert(!nda::IndexContainer<int>);
  static_assert(!nda::IndexContainer<nda::range>);
  static_assert(!nda::IndexContainer<nda::range::all_t>);
  static_assert(!nda::IndexContainer<nda::ellipsis>);
}

TEST(ExprIndexed, ArrayConcept) {
  // expr_indexed should satisfy the Array concept
  nda::array<double, 2> arr(5, 4);
  std::vector<long> indices = {2, 0, 4};
  auto expr = arr(indices, nda::range::all);

  static_assert(nda::Array<decltype(expr)>);
  static_assert(nda::ArrayOfRank<decltype(expr), 2>);
}

// ============================================================
// 1D array indexing
// ============================================================

TEST(ExprIndexed, Basic1D) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i * i);

  std::vector<long> indices = {3, 7, 1, 5};

  auto expr = arr(indices);

  // Check shape
  EXPECT_EQ(expr.shape(), (std::array<long, 1>{4}));
  EXPECT_EQ(expr.size(), 4);

  // Check element access
  EXPECT_EQ(expr(0), arr(3)); // 9
  EXPECT_EQ(expr(1), arr(7)); // 49
  EXPECT_EQ(expr(2), arr(1)); // 1
  EXPECT_EQ(expr(3), arr(5)); // 25

  // Materialize to array
  nda::array<double, 1> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 1>{4}));
  EXPECT_EQ(result(0), 9.0);
  EXPECT_EQ(result(1), 49.0);
  EXPECT_EQ(result(2), 1.0);
  EXPECT_EQ(result(3), 25.0);
}

// ============================================================
// 2D array indexing - single dimension
// ============================================================

TEST(ExprIndexed, Basic2D_IndexFirstDim) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  // Check shape: 3 rows (from indices), 4 columns (all)
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{3, 4}));

  // Check element access
  EXPECT_EQ(expr(0, 0), arr(2, 0)); // 20
  EXPECT_EQ(expr(0, 3), arr(2, 3)); // 23
  EXPECT_EQ(expr(1, 0), arr(0, 0)); // 0
  EXPECT_EQ(expr(2, 2), arr(4, 2)); // 42

  // Materialize
  nda::array<double, 2> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(result(0, 0), 20.0);
  EXPECT_EQ(result(1, 1), 1.0);
  EXPECT_EQ(result(2, 3), 43.0);
}

TEST(ExprIndexed, Basic2D_IndexSecondDim) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {3, 1};

  auto expr = arr(nda::range::all, indices);

  // Check shape: 5 rows (all), 2 columns (from indices)
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{5, 2}));

  // Check element access
  EXPECT_EQ(expr(0, 0), arr(0, 3)); // 3
  EXPECT_EQ(expr(0, 1), arr(0, 1)); // 1
  EXPECT_EQ(expr(4, 0), arr(4, 3)); // 43
  EXPECT_EQ(expr(4, 1), arr(4, 1)); // 41

  // Materialize
  nda::array<double, 2> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{5, 2}));
}

// ============================================================
// 2D array indexing - both dimensions
// ============================================================

TEST(ExprIndexed, Basic2D_IndexBothDims) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> idx1 = {2, 0};
  std::vector<long> idx2 = {3, 1, 0};

  auto expr = arr(idx1, idx2);

  // Check shape: Cartesian product 2 x 3
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{2, 3}));

  // Check element access: expr(i, j) = arr(idx1[i], idx2[j])
  EXPECT_EQ(expr(0, 0), arr(2, 3)); // 23
  EXPECT_EQ(expr(0, 1), arr(2, 1)); // 21
  EXPECT_EQ(expr(0, 2), arr(2, 0)); // 20
  EXPECT_EQ(expr(1, 0), arr(0, 3)); // 3
  EXPECT_EQ(expr(1, 1), arr(0, 1)); // 1
  EXPECT_EQ(expr(1, 2), arr(0, 0)); // 0

  // Materialize
  nda::array<double, 2> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{2, 3}));
  EXPECT_EQ(result(0, 0), 23.0);
  EXPECT_EQ(result(1, 2), 0.0);
}

// ============================================================
// Mixed indexing with ranges
// ============================================================

TEST(ExprIndexed, MixedWithRange) {
  nda::array<double, 2> arr(10, 8);
  for (long i = 0; i < 10; ++i)
    for (long j = 0; j < 8; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {5, 2, 8};

  // Index first dim, range on second dim
  auto expr = arr(indices, nda::range(2, 6));

  // Check shape: 3 rows, 4 columns
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{3, 4}));

  // Check element access
  EXPECT_EQ(expr(0, 0), arr(5, 2)); // 52
  EXPECT_EQ(expr(0, 3), arr(5, 5)); // 55
  EXPECT_EQ(expr(2, 0), arr(8, 2)); // 82
}

// ============================================================
// 3D array indexing
// ============================================================

TEST(ExprIndexed, Basic3D) {
  nda::array<double, 3> arr(4, 5, 6);
  for (long i = 0; i < 4; ++i)
    for (long j = 0; j < 5; ++j)
      for (long k = 0; k < 6; ++k) arr(i, j, k) = static_cast<double>(i * 100 + j * 10 + k);

  std::vector<long> indices = {1, 3};

  auto expr = arr(indices, nda::range::all, nda::range::all);

  // Check shape: 2 x 5 x 6
  EXPECT_EQ(expr.shape(), (std::array<long, 3>{2, 5, 6}));

  // Check element access
  EXPECT_EQ(expr(0, 0, 0), arr(1, 0, 0)); // 100
  EXPECT_EQ(expr(1, 2, 3), arr(3, 2, 3)); // 323
}

// ============================================================
// Ellipsis support
// ============================================================

TEST(ExprIndexed, WithEllipsis) {
  nda::array<double, 3> arr(4, 5, 6);
  for (long i = 0; i < 4; ++i)
    for (long j = 0; j < 5; ++j)
      for (long k = 0; k < 6; ++k) arr(i, j, k) = static_cast<double>(i * 100 + j * 10 + k);

  std::vector<long> indices = {1, 3};

  // Index first dim, ellipsis for rest
  auto expr = arr(indices, nda::ellipsis{});

  // Check shape: 2 x 5 x 6
  EXPECT_EQ(expr.shape(), (std::array<long, 3>{2, 5, 6}));

  // Check element access
  EXPECT_EQ(expr(0, 2, 3), arr(1, 2, 3)); // 123
  EXPECT_EQ(expr(1, 4, 5), arr(3, 4, 5)); // 345
}

// ============================================================
// Different container types
// ============================================================

TEST(ExprIndexed, StdArrayContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::array<long, 3> indices = {5, 2, 8};

  auto expr   = arr(indices);
  auto result = nda::array<double, 1>{expr};

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 8.0);
}

TEST(ExprIndexed, SpanContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> idx_vec = {5, 2, 8};
  std::span<const long> indices(idx_vec);

  auto expr   = arr(indices);
  auto result = nda::array<double, 1>{expr};

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 8.0);
}

TEST(ExprIndexed, IntVectorContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  // int should be convertible to long
  std::vector<int> indices = {5, 2, 8};

  auto expr   = arr(indices);
  auto result = nda::array<double, 1>{expr};

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 8.0);
}

// ============================================================
// Expression composition
// ============================================================

TEST(ExprIndexed, ArithmeticComposition) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> indices = {3, 7, 1};

  // Compose with scalar multiplication
  auto expr                    = 2.0 * arr(indices);
  nda::array<double, 1> result = expr;

  EXPECT_EQ(result(0), 6.0);  // 2 * 3
  EXPECT_EQ(result(1), 14.0); // 2 * 7
  EXPECT_EQ(result(2), 2.0);  // 2 * 1
}

TEST(ExprIndexed, TwoExprIndexedComposition) {
  nda::array<double, 1> arr1(10);
  nda::array<double, 1> arr2(10);
  for (long i = 0; i < 10; ++i) {
    arr1(i) = static_cast<double>(i);
    arr2(i) = static_cast<double>(i * 2);
  }

  std::vector<long> indices = {3, 7, 1};

  // Add two indexed expressions
  auto expr                    = arr1(indices) + arr2(indices);
  nda::array<double, 1> result = expr;

  EXPECT_EQ(result(0), 3.0 + 6.0);   // arr1[3] + arr2[3]
  EXPECT_EQ(result(1), 7.0 + 14.0);  // arr1[7] + arr2[7]
  EXPECT_EQ(result(2), 1.0 + 2.0);   // arr1[1] + arr2[1]
}

// ============================================================
// Matrix algebra preservation
// ============================================================

TEST(ExprIndexed, MatrixAlgebra) {
  nda::matrix<double> mat(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) mat(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4};

  auto expr = mat(indices, nda::range::all);

  // Should inherit matrix algebra
  static_assert(nda::get_algebra<decltype(expr)> == 'M');

  // Materialize as matrix
  nda::matrix<double> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{3, 4}));
}

// ============================================================
// Edge cases
// ============================================================

TEST(ExprIndexed, EmptyIndexContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> empty_indices = {};

  auto expr = arr(empty_indices);

  // Shape should have zero in the indexed dimension
  EXPECT_EQ(expr.shape(), (std::array<long, 1>{0}));
  EXPECT_EQ(expr.size(), 0);

  // Materialization should produce empty array
  nda::array<double, 1> result = expr;
  EXPECT_EQ(result.size(), 0);
}

TEST(ExprIndexed, SingleElementContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i * i);

  std::vector<long> indices = {5};

  auto expr = arr(indices);

  EXPECT_EQ(expr.shape(), (std::array<long, 1>{1}));
  EXPECT_EQ(expr.size(), 1);
  EXPECT_EQ(expr(0), 25.0); // 5^2
}

TEST(ExprIndexed, DuplicateIndices) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  // Same index repeated multiple times
  std::vector<long> indices = {3, 3, 3, 3};

  auto expr = arr(indices);

  EXPECT_EQ(expr.shape(), (std::array<long, 1>{4}));

  // All elements should be arr(3) = 3.0
  EXPECT_EQ(expr(0), 3.0);
  EXPECT_EQ(expr(1), 3.0);
  EXPECT_EQ(expr(2), 3.0);
  EXPECT_EQ(expr(3), 3.0);
}

TEST(ExprIndexed, ReversedIndices) {
  nda::array<double, 1> arr(5);
  for (long i = 0; i < 5; ++i) arr(i) = static_cast<double>(i);

  // Reverse order
  std::vector<long> indices = {4, 3, 2, 1, 0};

  auto expr                    = arr(indices);
  nda::array<double, 1> result = expr;

  EXPECT_EQ(result(0), 4.0);
  EXPECT_EQ(result(1), 3.0);
  EXPECT_EQ(result(2), 2.0);
  EXPECT_EQ(result(3), 1.0);
  EXPECT_EQ(result(4), 0.0);
}

// ============================================================
// Write access through expr_indexed
// ============================================================

TEST(ExprIndexed, WriteAccess1D) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> indices = {3, 7, 1};

  auto expr = arr(indices);

  // Write through expr_indexed
  expr(0) = 100.0;
  expr(1) = 200.0;

  // Verify writes went to underlying array
  EXPECT_EQ(arr(3), 100.0);
  EXPECT_EQ(arr(7), 200.0);
  EXPECT_EQ(arr(1), 1.0); // Unchanged
}

TEST(ExprIndexed, WriteAccess2D) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = 0.0;

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  expr(0, 0) = 42.0;
  expr(1, 3) = 99.0;

  EXPECT_EQ(arr(2, 0), 42.0);
  EXPECT_EQ(arr(0, 3), 99.0);
}

TEST(ExprIndexed, WriteAccessDuplicateIndices) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = 0.0;

  // Same index repeated - writes to same location
  std::vector<long> indices = {3, 3, 3};

  auto expr = arr(indices);

  expr(0) = 10.0;
  EXPECT_EQ(arr(3), 10.0);

  expr(1) = 20.0;
  EXPECT_EQ(arr(3), 20.0);

  expr(2) = 30.0;
  EXPECT_EQ(arr(3), 30.0);
}

// ============================================================
// Direct assignment to array
// ============================================================

TEST(ExprIndexed, DirectAssignment) {
  nda::array<double, 2> arr(10, 8);
  for (long i = 0; i < 10; ++i)
    for (long j = 0; j < 8; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {5, 2, 8, 1};

  // Direct assignment to new array
  nda::array<double, 2> result = arr(indices, nda::range::all);

  EXPECT_EQ(result.shape(), (std::array<long, 2>{4, 8}));

  // Check values
  EXPECT_EQ(result(0, 0), arr(5, 0)); // 50
  EXPECT_EQ(result(1, 3), arr(2, 3)); // 23
  EXPECT_EQ(result(2, 7), arr(8, 7)); // 87
  EXPECT_EQ(result(3, 0), arr(1, 0)); // 10
}

// ============================================================
// Assignment to expr_indexed from arrays/scalars
// ============================================================

TEST(ExprIndexed, AssignFromArray1D) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = 0.0;

  std::vector<long> indices = {3, 7, 1};

  nda::array<double, 1> src = {100.0, 200.0, 300.0};

  arr(indices) = src;

  EXPECT_EQ(arr(3), 100.0);
  EXPECT_EQ(arr(7), 200.0);
  EXPECT_EQ(arr(1), 300.0);
  EXPECT_EQ(arr(0), 0.0); // Unchanged
}

TEST(ExprIndexed, AssignFromArray2D) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = 0.0;

  std::vector<long> indices = {2, 0};

  nda::array<double, 2> src(2, 4);
  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 4; ++j) src(i, j) = static_cast<double>((i + 1) * 10 + j);

  arr(indices, nda::range::all) = src;

  // Row 2 should have 10, 11, 12, 13
  // Row 0 should have 20, 21, 22, 23
  EXPECT_EQ(arr(2, 0), 10.0);
  EXPECT_EQ(arr(2, 3), 13.0);
  EXPECT_EQ(arr(0, 0), 20.0);
  EXPECT_EQ(arr(0, 3), 23.0);
}

TEST(ExprIndexed, AssignFromScalar) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = 0.0;

  std::vector<long> indices = {3, 7, 1};

  arr(indices) = 42.0;

  EXPECT_EQ(arr(3), 42.0);
  EXPECT_EQ(arr(7), 42.0);
  EXPECT_EQ(arr(1), 42.0);
  EXPECT_EQ(arr(0), 0.0); // Unchanged
}

TEST(ExprIndexed, AssignFromExpression) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = 0.0;

  nda::array<double, 1> src = {1.0, 2.0, 3.0};

  std::vector<long> indices = {3, 7, 1};

  arr(indices) = 2.0 * src;

  EXPECT_EQ(arr(3), 2.0);
  EXPECT_EQ(arr(7), 4.0);
  EXPECT_EQ(arr(1), 6.0);
}

// ============================================================
// Slicing expr_indexed
// ============================================================

TEST(ExprIndexed, SliceNonIndexedDimWithRangeAll) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  // Slice the non-indexed dimension with range::all (should keep same shape)
  auto sliced = expr(nda::range::all, nda::range::all);

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 0)); // 20
  EXPECT_EQ(sliced(1, 3), arr(0, 3)); // 3
}

TEST(ExprIndexed, SliceNonIndexedDimWithRange) {
  nda::array<double, 2> arr(5, 8);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 8; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  // Slice the non-indexed dimension (columns 2-5)
  auto sliced = expr(nda::range::all, nda::range(2, 6));

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 2)); // 22
  EXPECT_EQ(sliced(0, 3), arr(2, 5)); // 25
  EXPECT_EQ(sliced(2, 0), arr(4, 2)); // 42
}

TEST(ExprIndexed, SliceIndexedDimWithRangeAll) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4, 1};

  auto expr = arr(indices, nda::range::all);

  // Slice the indexed dimension with range::all (should keep same shape)
  auto sliced = expr(nda::range::all, nda::range::all);

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{4, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 0)); // 20
  EXPECT_EQ(sliced(3, 0), arr(1, 0)); // 10
}

TEST(ExprIndexed, SliceIndexedDimWithRange) {
  nda::array<double, 2> arr(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4, 1};

  auto expr = arr(indices, nda::range::all);

  // Slice the indexed dimension (select first 2 from index container)
  auto sliced = expr(nda::range(0, 2), nda::range::all);

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{2, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 0)); // indices[0] = 2, so arr(2, 0) = 20
  EXPECT_EQ(sliced(1, 0), arr(0, 0)); // indices[1] = 0, so arr(0, 0) = 0
}

TEST(ExprIndexed, SliceBothDimsWithRange) {
  nda::array<double, 2> arr(10, 8);
  for (long i = 0; i < 10; ++i)
    for (long j = 0; j < 8; ++j) arr(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {5, 2, 8, 1, 9};

  auto expr = arr(indices, nda::range::all);

  // Slice both dims
  auto sliced = expr(nda::range(1, 4), nda::range(2, 6));

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{3, 4}));
  // sliced(0,0) = expr(1, 2) = arr(indices[1], 2) = arr(2, 2) = 22
  EXPECT_EQ(sliced(0, 0), arr(2, 2));
  // sliced(2,3) = expr(3, 5) = arr(indices[3], 5) = arr(1, 5) = 15
  EXPECT_EQ(sliced(2, 3), arr(1, 5));
}
