// Copyright (c) 2019-2023 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#include <cstdlib>
#include <string>
#include "cutensor.h"

#include "nda/macros.hpp"
#include "nda/exceptions.hpp"
#include "cuda_runtime.h"

#include "nda/tensor/interface/cutensor_interface.hpp"

namespace nda::tensor::cutensor {

  namespace {

    /*
     * Owns the cutensor handle and the plan cache. The cache is cleared explicitly in
     * the destructor: the body of ~handle_t runs before the member destructors, so
     * otherwise the plans would be destroyed after cutensorDestroy.
     */
    struct handle_t {
      handle_t() { cutensorCreate(&h); }
      ~handle_t() {
        cache.clear();
        CUTENSOR_CHECK(cutensorDestroy, h);
      }

      handle_t(handle_t const &)            = delete;
      handle_t(handle_t &&)                 = delete;
      handle_t &operator=(handle_t const &) = delete;
      handle_t &operator=(handle_t &&)      = delete;

      cutensorHandle_t h = {};
      plan_cache_t cache = {};
    };

    handle_t &get_handle() {
      static handle_t h;
      return h;
    }

  } // namespace

  cutensorHandle_t &get_handle_ptr() { return get_handle().h; }

  plan_cache_t &get_plan_cache() { return get_handle().cache; }

  void clear_plan_cache() { get_handle().cache.clear(); }

  std::size_t plan_cache_size() { return get_handle().cache.size(); }

  // control device synchronization during cutensor calls
  bool synchronize = true;
  bool get_synchronization() { return synchronize; }
  void set_synchronization(bool s_) { synchronize = s_; }

} // namespace nda::tensor::cutensor
