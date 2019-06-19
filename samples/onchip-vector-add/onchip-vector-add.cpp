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

template <int CacheLineSize, typename T, typename InAccessorT,
          typename OutAccessorT>
class VectorAddition {
 private:
  const int _N;
  const InAccessorT _accessorA, _accessorB;
  const OutAccessorT _accessorC;

 public:
  VectorAddition(int N, InAccessorT accessorA, InAccessorT accessorB,
                 OutAccessorT accessorC)
      : _N(N),
        _accessorA(accessorA),
        _accessorB(accessorB),
        _accessorC(accessorC) {}

  void operator()(cl::sycl::nd_item<1> work_item) {
    T private_A[CacheLineSize];
    T private_B[CacheLineSize];

    int global_id = static_cast<int>(work_item.get_global_id(0));
    int global_range = static_cast<int>(work_item.get_global_range()[0]);

    int id = global_id * CacheLineSize;
    for (int i = id; i < _N; i += CacheLineSize * global_range) {
#pragma unroll 8
      for (int j = 0; j < CacheLineSize; j++) {
        private_A[j] = _accessorA[i + j];
      }
      work_item.mem_fence(cl::sycl::access::fence_space::local_space);
#pragma unroll 8
      for (int j = 0; j < CacheLineSize; j++) {
        private_B[j] = _accessorB[i + j];
      }
      work_item.mem_fence(cl::sycl::access::fence_space::local_space);
#pragma unroll 8
      for (int j = 0; j < CacheLineSize; j++) {
        _accessorC[i + j] = private_A[j] + private_B[j];
      }
    }
  }

  //   void operator()(cl::sycl::nd_item<1> work_item) {
  //     T private_A[CacheLineSize];

  //     int global_id = static_cast<int>(work_item.get_global_id(0));
  //     int global_range = static_cast<int>(work_item.get_global_range()[0]);

  //     int id = global_id * CacheLineSize;
  //     for (int i = id; i < _N; i += CacheLineSize * global_range) {
  // #pragma unroll 8
  //       for (int j = 0; j < CacheLineSize; j++) {
  //         private_A[j] = _accessorA[i + j];
  //       }
  //       work_item.mem_fence(cl::sycl::access::fence_space::local_space);
  // #pragma unroll 8
  //       for (int j = 0; j < CacheLineSize; j++) {
  //         private_A[j] += _accessorB[i + j];
  //       }
  //       work_item.mem_fence(cl::sycl::access::fence_space::local_space);
  // #pragma unroll 8
  //       for (int j = 0; j < CacheLineSize; j++) {
  //         _accessorC[i + j] = private_A[j];
  //       }
  //     }
  //   }
};

template <int CacheLineSize, typename T, typename InAccessorT,
          typename OutAccessorT>
VectorAddition<CacheLineSize, T, InAccessorT, OutAccessorT> make_vector_add(
    int N, InAccessorT accessorA, InAccessorT accessorB,
    OutAccessorT accessorC) {
  return VectorAddition<CacheLineSize, T, InAccessorT, OutAccessorT>(
      N, accessorA, accessorB, accessorC);
}

template <int CacheLineSize, bool UseOnchipMemory, typename T>
std::tuple<double, double> vadd(cl::sycl::queue& q, cl::sycl::context& c,
                                cl::sycl::nd_range<1> exec_range,
                                const std::vector<T>& VA,
                                const std::vector<T>& VB, std::vector<T>& VC) {
  // Get the size of the input(s), and verify that they're all the same
  assert((VA.size() == VB.size()) && (VB.size() == VC.size()));

  const size_t N = VA.size();

  cl::sycl::range<1> item_range{N};

  auto context_bound_property = cl::sycl::property::buffer::context_bound(c);
  auto use_onchip_memory_property =
      cl::sycl::codeplay::property::buffer::use_onchip_memory(
          cl::sycl::codeplay::property::prefer);

  // Create straightforward DDR buffers
  // As they are context bound, the data will be immediately copied into them.
  cl::sycl::buffer<T, 1> ddr_bufferA(VA.data(), item_range,
                                     {context_bound_property});
  cl::sycl::buffer<T, 1> ddr_bufferB(VB.data(), item_range,
                                     {context_bound_property});
  cl::sycl::buffer<T, 1> ddr_bufferC(VC.data(), item_range,
                                     {context_bound_property});

  // create onchip buffers, in case `UseOnchipMemory` is specified
  cl::sycl::buffer<T, 1> onchip_bufferA(
      item_range, {context_bound_property, use_onchip_memory_property});
  cl::sycl::buffer<T, 1> onchip_bufferB(
      item_range, {context_bound_property, use_onchip_memory_property});
  cl::sycl::buffer<T, 1> onchip_bufferC(
      item_range, {context_bound_property, use_onchip_memory_property});

  // Flush before starting timings, to make sure that we're accurate in our
  // timings!
  cl::sycl::codeplay::flush(q);

  auto wallclock_start = std::chrono::system_clock::now();

  // If we're using onchip memory, manually copy from the DDR buffers to the
  // onchip buffers
  if (UseOnchipMemory) {
    q.submit([&](cl::sycl::handler& cgh) {
      auto ddr_accessor = ddr_bufferA.template get_access<sycl_read>(cgh);
      auto onchip_accessor =
          onchip_bufferA.template get_access<sycl_write>(cgh);
      cgh.copy(ddr_accessor, onchip_accessor);
    });
    q.submit([&](cl::sycl::handler& cgh) {
      auto ddr_accessor = ddr_bufferB.template get_access<sycl_read>(cgh);
      auto onchip_accessor =
          onchip_bufferB.template get_access<sycl_write>(cgh);
      cgh.copy(ddr_accessor, onchip_accessor);
    });
  }

  /* Get the size of the intput data as an integer, as opposed to a size_t.
   This is to reduce the number of registers that each index variable uses.
   As `size_t` is a 64-bit type on Renesas hardware variables of type `size_t`
   use two registers while variable of `int`, which is a 32-bit type, use a
   single register
  */
  const int kernel_n = static_cast<int>(N);

  // Perform the actual vector add!
  auto e = q.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = UseOnchipMemory
                         ? onchip_bufferA.template get_access<sycl_read>(cgh)
                         : ddr_bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = UseOnchipMemory
                         ? onchip_bufferB.template get_access<sycl_read>(cgh)
                         : ddr_bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = UseOnchipMemory
                         ? onchip_bufferC.template get_access<sycl_write>(cgh)
                         : ddr_bufferC.template get_access<sycl_write>(cgh);

    auto vector_add_kernel = make_vector_add<CacheLineSize, float>(
        kernel_n, accessorA, accessorB, accessorC);

    cgh.parallel_for(exec_range, vector_add_kernel);
  });

  // Enqueue a copy operation to manually copy the data back to DDR memory from
  // onchip memory
  if (UseOnchipMemory) {
    q.submit([&](cl::sycl::handler& cgh) {
      auto ddr_accessor = ddr_bufferC.template get_access<sycl_write>(cgh);
      auto onchip_accessor = onchip_bufferC.template get_access<sycl_read>(cgh);
      cgh.copy(onchip_accessor, ddr_accessor);
    });
  }

  // Flush after running all kernels and copies
  cl::sycl::codeplay::flush(q);

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

  auto best_wallclock_time = std::tuple<double, double>(
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

  auto best_event_time = std::tuple<double, double>(
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

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

    if (wallclock_time < std::get<0>(best_wallclock_time)) {
      best_wallclock_time =
          std::tuple<double, double>(wallclock_time, event_time);
    }

    if (event_time < std::get<1>(best_event_time)) {
      best_event_time = std::tuple<double, double>(wallclock_time, event_time);
    }
  }

  std::cout << "\t- Best wallclock time / Assoc event time: " << std::endl
            << "\t\t" << 1e-6 * std::get<0>(best_wallclock_time) << " ms"
            << " / " << 1e-6 * std::get<1>(best_wallclock_time) << " ms"
            << std::endl;

  std::cout << "\t- Assoc wallclock time / Best event time: " << std::endl
            << "\t\t" << 1e-6 * std::get<0>(best_event_time) << " ms"
            << " / " << 1e-6 * std::get<1>(best_event_time) << " ms"
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

  std::cout << std::endl;

  cl::sycl::nd_range<1> exec_range{global_size, local_size};

  measure_vadd<iterations>("Onchip vector add (1*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<1, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (1*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<1, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  measure_vadd<iterations>("Onchip vector add (4*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<4, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (4*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<4, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  measure_vadd<iterations>("Onchip vector add (8*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<8, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (8*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<8, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  measure_vadd<iterations>("Onchip vector add (1*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<16, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (1*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<16, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  measure_vadd<iterations>("Onchip vector add (3*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<32, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (3*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<32, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  measure_vadd<iterations>("Onchip vector add (6*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<64, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (6*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<64, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  measure_vadd<iterations>("Onchip vector add (1*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<128, true, cl::sycl::cl_float>);
  measure_vadd<iterations>("DDR vector add (1*2*4 bytes of private memory)",
                           elems, q, c, exec_range,
                           vadd<128, false, cl::sycl::cl_float>);
  std::cout << std::endl;

  return 0;
}
