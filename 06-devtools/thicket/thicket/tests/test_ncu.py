# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from hatchet.node import Node

from thicket.ncu import (
    _match_call_trace_regex,
    _match_kernel_str_to_cali,
    _multi_match_fallback_similarity,
)


def test_match_call_trace_regex():

    # Base_CUDA variant
    (
        kernel_str,
        demangled_kernel_name,
        instance_num,
        instance_exists,
        skip_kernel,
    ) = _match_call_trace_regex(
        ["RAJAPerf", "Basic", "Basic_DAXPY"],
        "void rajaperf::basic::daxpy<(unsigned long)128>(double *, double *, double, long)",
        debug=False,
    )
    assert kernel_str == "daxpy"

    # lambda_CUDA variant
    (
        kernel_str,
        demangled_kernel_name,
        instance_num,
        instance_exists,
        skip_kernel,
    ) = _match_call_trace_regex(
        ["RAJAPerf", "Polybench", "Polybench_ATAX"],
        "void rajaperf::polybench::poly_atax_lam<(unsigned long)128, void rajaperf::polybench::POLYBENCH_ATAX::runCudaVariantImpl<(unsigned long)128>(rajaperf::VariantID)::[lambda(long) (instance 2)]>(long, T2)",
        debug=False,
    )
    assert kernel_str == "poly_atax_lam"

    # RAJA_CUDA variant
    (
        kernel_str,
        demangled_kernel_name,
        instance_num,
        instance_exists,
        skip_kernel,
    ) = _match_call_trace_regex(
        ["RAJAPerf", "Apps", "Apps_ENERGY"],
        "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, (int)128, (int)0>, RAJA::cuda::MaxOccupancyConcretizer, (unsigned long)1, (bool)1>, (unsigned long)1, RAJA::Iterators::numeric_iterator<long, long, long *>, void rajaperf::apps::ENERGY::runCudaVariantImpl<(unsigned long)128>(rajaperf::VariantID)::[lambda() (instance 1)]::operator ()() const::[lambda(long) (instance 4)], long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, (int)128, (int)0>, (unsigned long)128>(T4, T3, T5)",
        debug=False,
    )
    assert kernel_str == "ENERGY"


def test_match_kernel_str_to_cali():
    # RAJA_CUDA variant
    (
        kernel_str,
        demangled_kernel_name,
        instance_num,
        instance_exists,
        skip_kernel,
    ) = _match_call_trace_regex(
        ["RAJAPerf", "Apps", "Apps_ENERGY"],
        "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, (int)128, (int)0>, RAJA::cuda::MaxOccupancyConcretizer, (unsigned long)1, (bool)1>, (unsigned long)1, RAJA::Iterators::numeric_iterator<long, long, long *>, void rajaperf::apps::ENERGY::runCudaVariantImpl<(unsigned long)128>(rajaperf::VariantID)::[lambda() (instance 1)]::operator ()() const::[lambda(long) (instance 4)], long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, (int)128, (int)0>, (unsigned long)128>(T4, T3, T5)",
        debug=False,
    )
    # Test multi-instance (for energy4)
    node_set = [
        Node({"name": "RAJAPerf", "type": "function"}),
        Node({"name": "Apps", "type": "function"}),
        Node({"name": "Apps_ENERGY", "type": "function"}),
        Node({"name": "cudaLaunchKernel", "type": "function"}),
        # energy1
        Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#1}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#1}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ),
        # energy2
        Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#2}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#2}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ),
        # energy3
        Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#3}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#3}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ),
        # energy4
        Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#4}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#4}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ),
        # energy5
        Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#5}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#5}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ),
        # energy6
        Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#6}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#6}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ),
    ]
    matched_nodes = _match_kernel_str_to_cali(
        node_set, kernel_str, instance_num, True, instance_exists
    )
    assert len(matched_nodes) == 1
    # energy4
    assert (
        matched_nodes[0].frame["name"]
        == Node(
            {
                "name": "void RAJA::policy::cuda::impl::forall_cuda_kernel<RAJA::policy::cuda::cuda_exec_explicit<RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, RAJA::cuda::MaxOccupancyConcretizer, 1ul, true>, 1ul, RAJA::Iterators::numeric_iterator<long, long, long*>, void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#4}, long, RAJA::iteration_mapping::Direct, RAJA::cuda::IndexGlobal<(RAJA::named_dim)0, 128, 0>, 128ul>(void rajaperf::apps::ENERGY::runCudaVariantImpl<128ul>(rajaperf::VariantID)::{lambda()#1}::operator()() const::{lambda(long)#4}, RAJA::Iterators::numeric_iterator<long, long, long*>, long)",
                "type": "kernel",
            }
        ).frame["name"]
    )


def test_multi_match_fallback_similarity():
    # CUB kernels
    demangled_kernel_name = "void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<double, cub::NullType, int>::Policy700, (bool)1, (bool)0, double, int>(const T4 *, T5 *, T5, int, int, cub::GridEvenShare<T5>)"
    (
        kernel_str,
        demangled_kernel_name,
        instance_num,
        instance_exists,
        skip_kernel,
    ) = _match_call_trace_regex(
        ["RAJAPerf", "Algorithm", "Algorithm_SORT", "DeviceRadixSortUpsweepKernel"],
        demangled_kernel_name=demangled_kernel_name,
        debug=False,
    )
    node_set = [
        Node({"name": "RAJAPerf", "type": "function"}),
        Node({"name": "Algorithm", "type": "function"}),
        Node({"name": "Algorithm_SORT", "type": "function"}),
        Node({"name": "cudaLaunchKernel", "type": "function"}),
        # "false, false" wrong match
        Node(
            {
                "name": "void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<double, cub::NullType, int>::Policy700, false, false, double, int>(double const*, int*, int, int, int, cub::GridEvenShare<int>)",
                "type": "kernel",
            }
        ),
        # "true, false" correct match
        Node(
            {
                "name": "void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<double, cub::NullType, int>::Policy700, true, false, double, int>(double const*, int*, int, int, int, cub::GridEvenShare<int>)",
                "type": "kernel",
            }
        ),
    ]
    matched_nodes = _match_kernel_str_to_cali(
        node_set, kernel_str, instance_num, True, instance_exists
    )
    matched_node = _multi_match_fallback_similarity(
        matched_nodes, demangled_kernel_name, debug=False
    )
    assert (
        matched_node.frame["name"]
        == "void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<double, cub::NullType, int>::Policy700, true, false, double, int>(double const*, int*, int, int, int, cub::GridEvenShare<int>)"
    )
