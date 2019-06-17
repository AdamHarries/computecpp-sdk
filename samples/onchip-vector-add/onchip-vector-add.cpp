/***************************************************************************
 *
 *  Copyright (C) 2018 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  onchip-vector-add.cpp
 *
 *  Description:
 *    Example of a vector addition in SYCL.
 *
 **************************************************************************/

/* This example is a very small one designed to show how compact SYCL code
 * can be. That said, it includes no error checking and is rather terse. */
#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

#include <array>
#include <chrono>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <int CacheLineSize, bool UseOnchipMemory, bool SinglePrivateArray,
          typename T>
class VAdd;

template <int CacheLineSize, bool UseOnchipMemory, bool SinglePrivateArray,
          typename T>
std::tuple<double, double>
// cl::sycl::event
vadd(cl::sycl::queue& q, cl::sycl::context& c, cl::sycl::nd_range<1> exec_range,
     const std::vector<T>& VA, const std::vector<T>& VB, std::vector<T>& VC) {
  // Get the size of the input(s), and verify that they're all the same
  assert((VA.size() == VB.size()) && (VB.size() == VC.size()));

  const size_t N = VA.size();

  cl::sycl::range<1> item_range{N};

  const auto buffer_properties =
      UseOnchipMemory
          ? cl::sycl::property_list(
                {cl::sycl::codeplay::property::buffer::use_onchip_memory(
                     cl::sycl::codeplay::property::prefer),
                 cl::sycl::property::buffer::context_bound(c)})
          : cl::sycl::property_list(
                {cl::sycl::property::buffer::context_bound(c)});

  // Create onchip buffers.
  cl::sycl::buffer<T, 1> bufferA(item_range, buffer_properties);
  cl::sycl::buffer<T, 1> bufferB(item_range, buffer_properties);
  cl::sycl::buffer<T, 1> bufferC(item_range, buffer_properties);

  auto wallclock_start = std::chrono::system_clock::now();

  // Manually copy data from the host buffers (vectors) to the onchip buffers
  q.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_write>(cgh);
    cgh.copy(VA.data(), accessorA);
  });

  q.submit([&](cl::sycl::handler& cgh) {
    auto accessorB = bufferB.template get_access<sycl_write>(cgh);
    cgh.copy(VB.data(), accessorB);
  });

  cl::sycl::codeplay::flush(q);

  const int kernel_n = static_cast<int>(N);

  // Perform the actual vector add!
  auto e = q.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class VAdd<CacheLineSize, UseOnchipMemory, SinglePrivateArray, T>>(exec_range,
		[kernel_n, accessorA, accessorB, accessorC](cl::sycl::nd_item<1> work_item)
    {
      if (SinglePrivateArray) {
        T private_A[CacheLineSize];
        // T private_B[CacheLineSize];

        int global_id = static_cast<int>(work_item.get_global_id(0));
        int global_range = static_cast<int>(work_item.get_global_range()[0]);

        int id = global_id * CacheLineSize;
        for (int i = id; i < kernel_n; i += CacheLineSize * global_range) {
#pragma unroll 8
          for (int j = 0; j < CacheLineSize; j++) {
            private_A[j] = accessorA[i + j];
          }
          work_item.mem_fence(cl::sycl::access::fence_space::local_space);
#pragma unroll 8
          for (int j = 0; j < CacheLineSize; j++) {
            private_A[j] += accessorB[i + j];
          }
          // for (int j = 0; j < CacheLineSize; j++) {
          //   private_B[j] = accessorB[i + j];
          // }
          // work_item.mem_fence(cl::sycl::access::fence_space::local_space);
#pragma unroll 8
          for (int j = 0; j < CacheLineSize; j++) {
            accessorC[i + j] = private_A[j];
          }
        }
      } else {
        T private_A[CacheLineSize];
        T private_B[CacheLineSize];
        // T private_C[CacheLineSize];

        int global_id = static_cast<int>(work_item.get_global_id(0));
        int global_range = static_cast<int>(work_item.get_global_range()[0]);

        int id = global_id * CacheLineSize;
        for (int i = id; i < kernel_n; i += CacheLineSize * global_range) {
#pragma unroll 8
          for (int j = 0; j < CacheLineSize; j++) {
            private_A[j] = accessorA[i + j];
          }
          work_item.mem_fence(cl::sycl::access::fence_space::local_space);
#pragma unroll 8
          for (int j = 0; j < CacheLineSize; j++) {
            private_B[j] = accessorB[i + j];
          }
          // work_item.mem_fence(cl::sycl::access::fence_space::local_space);
#pragma unroll 8
          for (int j = 0; j < CacheLineSize; j++) {
            accessorC[j] = private_A[j] + private_B[j];
          }

          // #pragma unroll 8
          //           for (int j = 0; j < CacheLineSize; j++) {
          // accessorC[i + j] = private_C[j];
          //           }
        }
      }
  });
  });

  cl::sycl::codeplay::flush(q);

  q.submit([&](cl::sycl::handler& cgh) {
    auto accessorC = bufferC.template get_access<sycl_read>(cgh);
    cgh.copy(accessorC, VC.data());
  });

  q.wait();

  auto wallclock_end = std::chrono::system_clock::now();

  cl_ulong event_start = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();
  cl_ulong event_end = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();
  double event_time = static_cast<double>(event_end - event_start);
  double wallclock_time = (wallclock_end - wallclock_start).count();

  return std::make_tuple(event_time, wallclock_time);
}

template <int iterations, typename function_t>
static int measure_vadd(std::string name, unsigned int elems,
                        cl::sycl::queue& q, cl::sycl::context& c,
                        cl::sycl::nd_range<1> exec_range, function_t func) {
  // Generate data for the onchip vector add.
  std::vector<cl::sycl::cl_float> A(elems), B(elems), C(elems);
  std::generate(A.begin(), A.end(), [n = 0]() mutable { return n++; });
  std::generate(B.begin(), B.end(), [n = elems]() mutable { return n--; });

  // Run the vector add `iterations` times, and time it.
  std::cout << "Running \"" << name << "\"" << std::endl;
  // double total_wallclock_time = 0;
  double best_event_time = std::numeric_limits<float>::max();

  for (int i = 0; i < iterations; i++) {
    double event_time, wallclock_time;
    std::tie(event_time, wallclock_time) = func(q, c, exec_range, A, B, C);

    // check the result
    for (unsigned int i = 0; i < elems; i++) {
      if (C[i] != A[i] + B[i]) {
        std::cout << "\t === The results are incorrect (element " << i << " is "
                  << C[i] << "!)\n";
        return 1;
      }
    }

    if (event_time < best_event_time) {
      best_event_time = event_time;
    }
    // total_event_time += event_time;
    // total_wallclock_time += wallclock_time;
  }
  // std::cout << "\t- Mean wallclock time: "
  //           << (1e-6 * total_wallclock_time / iterations) << " ms" <<
  //           std::endl;
  std::cout << "\t- Best event time: " << (1e-6 * best_event_time) << " ms"
            << std::endl;

  return 0;
}

int main() {
  constexpr const size_t elems = 112 * 1024;
  constexpr const int iterations = 1024;

  std::cout << "Elems: " << elems << std::endl
            << "Iterations: " << iterations << std::endl;

  std::vector<cl::sycl::cl_float> A(elems), B(elems), C(elems);
  std::generate(A.begin(), A.end(), [n = 0]() mutable { return n++; });
  std::generate(B.begin(), B.end(), [n = elems]() mutable { return n--; });

  cl::sycl::queue q =
      cl::sycl::queue(cl::sycl::default_selector(),
                      {cl::sycl::property::queue::enable_profiling()});

  cl::sycl::context c = q.get_context();
  cl::sycl::device d = q.get_device();

  std::cout << "Max work group size: "
            << d.get_info<cl::sycl::info::device::max_work_group_size>()
            << std::endl;

  std::cout << "Max compute units: "
            << d.get_info<cl::sycl::info::device::max_compute_units>()
            << std::endl;

  std::cout << "Name: " << d.get_info<cl::sycl::info::device::name>()
            << std::endl;

  cl::sycl::cl_uint compute_units =
      d.get_info<cl::sycl::info::device::max_compute_units>();
  cl::sycl::cl_uint local_size =
      d.get_info<cl::sycl::info::device::max_work_group_size>();
  cl::sycl::cl_uint global_size = local_size * compute_units;

  std::cout << "Global size: " << global_size << std::endl;
  std::cout << "Local size: " << local_size << std::endl;

  cl::sycl::nd_range<1> exec_range{global_size, local_size};

  // benchmark various private memory sizes of the "onchip" vector add.
  std::cout << std::endl << "Benchmarking `onchip` vector adds: " << std::endl;
  {
    // measure_vadd<iterations>(
    //     "Onchip vector add (multiple private memory arrays of size 1)",
    //     elems, q, c, exec_range, vadd<1, true, false, cl::sycl::cl_float>);
    // measure_vadd<iterations>(
    //     "Onchip vector add (single private memory array of size 1)", elems,
    //     q, c, exec_range, vadd<1, true, true, cl::sycl::cl_float>);
    // std::cout << std::endl;

    // measure_vadd<iterations>(
    //     "Onchip vector add (multiple private memory arrays of size 4)",
    //     elems, q, c, exec_range, vadd<4, true, false, cl::sycl::cl_float>);
    // measure_vadd<iterations>(
    //     "Onchip vector add (single private memory array of size 4)", elems,
    //     q, c, exec_range, vadd<4, true, true, cl::sycl::cl_float>);
    // std::cout << std::endl;

    measure_vadd<iterations>(
        "Onchip vector add (multiple private memory arrays of size 8)", elems,
        q, c, exec_range, vadd<8, true, false, cl::sycl::cl_float>);
    measure_vadd<iterations>(
        "Onchip vector add (single private memory array of size 8)", elems, q,
        c, exec_range, vadd<8, true, true, cl::sycl::cl_float>);
    std::cout << std::endl;

    measure_vadd<iterations>(
        "Onchip vector add (multiple private memory arrays of size 16)", elems,
        q, c, exec_range, vadd<16, true, false, cl::sycl::cl_float>);
    measure_vadd<iterations>(
        "Onchip vector add (single private memory array of size 16)", elems, q,
        c, exec_range, vadd<16, true, true, cl::sycl::cl_float>);
    std::cout << std::endl;

    measure_vadd<iterations>(
        "Onchip vector add (multiple private memory arrays of size 32)", elems,
        q, c, exec_range, vadd<32, true, false, cl::sycl::cl_float>);
    measure_vadd<iterations>(
        "Onchip vector add (single private memory array of size 32)", elems, q,
        c, exec_range, vadd<32, true, true, cl::sycl::cl_float>);
    std::cout << std::endl;

    measure_vadd<iterations>(
        "Onchip vector add (multiple private memory arrays of size 64)", elems,
        q, c, exec_range, vadd<64, true, false, cl::sycl::cl_float>);
    measure_vadd<iterations>(
        "Onchip vector add (single private memory array of size 64)", elems, q,
        c, exec_range, vadd<64, true, true, cl::sycl::cl_float>);
    std::cout << std::endl;

    // measure_vadd<iterations>(
    //     "Onchip vector add (multiple private memory arrays of size 128)",
    //     elems, q, c, exec_range, vadd<128, true, false, cl::sycl::cl_float>);
    // measure_vadd<iterations>(
    //     "Onchip vector add (single private memory array of size 128)", elems,
    //     q, c, exec_range, vadd<128, true, true, cl::sycl::cl_float>);
    // std::cout << std::endl;
  }

  // Benchmark various private memory sizes of the "simple" vector add.
  // std::cout << std::endl << "Benchmarking `simple` vector adds: " <<
  // std::endl;
  // {
  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 1)",
  //       elems, q, c, exec_range, vadd<1, false, false, cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 1)", elems,
  //       q, c, exec_range, vadd<1, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;

  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 4)",
  //       elems, q, c, exec_range, vadd<4, false, false, cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 4)", elems,
  //       q, c, exec_range, vadd<4, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;

  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 8)",
  //       elems, q, c, exec_range, vadd<8, false, false, cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 8)", elems,
  //       q, c, exec_range, vadd<8, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;

  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 16)",
  //       elems, q, c, exec_range, vadd<16, false, false, cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 16)", elems,
  //       q, c, exec_range, vadd<16, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;

  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 32)",
  //       elems, q, c, exec_range, vadd<32, false, false, cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 32)", elems,
  //       q, c, exec_range, vadd<32, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;

  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 64)",
  //       elems, q, c, exec_range, vadd<64, false, false, cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 64)", elems,
  //       q, c, exec_range, vadd<64, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;

  //   measure_vadd<iterations>(
  //       "Simple vector add (multiple private memory arrays of size 128)",
  //       elems, q, c, exec_range, vadd<128, false, false,
  //       cl::sycl::cl_float>);
  //   measure_vadd<iterations>(
  //       "Simple vector add (single private memory array of size 128)", elems,
  //       q, c, exec_range, vadd<128, false, true, cl::sycl::cl_float>);
  //   std::cout << std::endl;
  // }
  return 0;
}
