// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

TEST(NDA, StackArrayConstructionAndAssignment) {
  nda::stack_array<long, 3, 3> A;
  nda::array<long, 2> B(3, 3);

  A = 3;
  B = 3;
  EXPECT_ARRAY_NEAR(A, B);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 10 * j;
      B(i, j) = i + 10 * j;
    }
  }
  EXPECT_ARRAY_NEAR(A, B);

  auto C = A;
  C      = A + B;
  EXPECT_ARRAY_NEAR(C, 2 * B);
}

TEST(NDA, StackArraySlice) {
  nda::stack_array<long, 3, 3> A;
  A = 3;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) { A(i, j) = i + 10 * j; }
  }

  auto A_v1 = A(nda::range::all, 1);
  nda::array<long, 2> B1{A};
  nda::array<long, 1> B2{A_v1};
  EXPECT_ARRAY_NEAR(A, B1);
  EXPECT_ARRAY_NEAR(A_v1, B2);

  auto A_v2 = A(1, nda::range::all);
  nda::array<long, 2> C1{A};
  nda::array<long, 1> C2{A_v2};
  EXPECT_ARRAY_NEAR(A, C1);
  EXPECT_ARRAY_NEAR(A_v2, C2);
}

TEST(NDA, StackArrayFromStaticRangeSlice) {
  constexpr int N = 4;
  nda::array<double, 2> mat(N, N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { mat(i, j) = i * N + j; }
  }

  // Slicing with static_range<1, N> gives an (N-1) x (N-1) view
  auto view = mat(nda::static_range<1, N>{}, nda::static_range<1, N>{});
  EXPECT_EQ(view.shape()[0], N - 1);
  EXPECT_EQ(view.shape()[1], N - 1);

  // Constructing a stack_array with matching size should work
  nda::stack_array<double, N - 1, N - 1> lmat(view);
  EXPECT_EQ(lmat.shape()[0], N - 1);
  EXPECT_EQ(lmat.shape()[1], N - 1);

#ifndef NDEBUG
  // Constructing a stack_array with wrong size should fail (std::terminate via EXPECTS)
  // stack_array<double, N, N> expects shape (N, N), but view has shape (N-1, N-1)
  EXPECT_DEATH(([&]() { nda::stack_array<double, N, N> wrong(view); }()), "");
#endif
}
