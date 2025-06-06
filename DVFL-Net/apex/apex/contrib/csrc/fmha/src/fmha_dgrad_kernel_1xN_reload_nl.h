/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include "fmha_kernel.h"
#include <fmha/kernel_traits.h>
#include <fmha/gemm.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int CHUNKS, typename Kernel_traits, typename Params>
inline __device__ void compute_dv_1xN_nl(const Params &params) {

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dv = fmha::Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;

    static_assert(Cta_tile_dv::M == 512 || Cta_tile_dv::M == 384 || Cta_tile_dv::M == 256 || Cta_tile_dv::M == 128);
    static_assert(Cta_tile_dv::N == 64);
    static_assert(Cta_tile_dv::K == 16);

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dv = fmha::Hmma_tile<Cta_tile_dv>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;
    // The shared memory tile to reload Q as fragment b.
    using Smem_tile_qt = fmha::Smem_tile_b<Cta_tile_dv, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store dV.
    using Gmem_tile_dv = fmha::Gmem_tile_qkv<typename Kernel_traits::Cta_tile_o, 
                                             fmha::BITS_PER_ELEMENT_B, 
                                             Cta_tile_p::N, //S, 
                                             Cta_tile_p::K, //D, 
                                             2*CHUNKS>;

    // The shared memory tile to swizzle dV.
    using Smem_tile_dv = fmha::Smem_tile_mma_epilogue<Cta_tile_dv>;
    static_assert(Smem_tile_dv::NUM_LDS == Gmem_tile_dv::LDGS);
    static_assert(Smem_tile_dv::THREADS_PER_ROW == Gmem_tile_dv::THREADS_PER_ROW);

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;
    using Smem_tile_st = typename Kernel_traits::Smem_tile_st;
    using Gmem_tile_do = typename Kernel_traits::Gmem_tile_do;

    // Shared memory.
    extern __shared__ char smem_[];

    // The block index for the chunk.
    const int bidc = blockIdx.z;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.x;
    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() )
        return;
    fmha::Mask<Cta_tile_p> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_do gmem_q(params, binfo, tidx);  // treating dout as Q
    // Allocate the shared memory tile loader for Q.
    Smem_tile_q smem_q(&smem_[0], tidx);
    Smem_tile_qt smem_qt(&smem_[0], tidx);
    Smem_tile_st smem_s(&smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 2, binfo, tidx);  // treating V as K
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    Gmem_tile_s gmem_s(params, binfo, tidx);

    using Noloop = Noloop_traits<CHUNKS, Cta_tile_p>;

    Noloop nl_traits(bidc, binfo);
    nl_traits.move_all(gmem_q, gmem_s);

    // Trigger the loads for Q.
    gmem_q.load(smem_q);
    // Trigger the loads for K.
    gmem_k.load(smem_k);

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_q);
    gmem_k.commit(smem_k);

    // Make sure the data is in shared memory.
    __syncthreads();

    // Load the fragments for Q.
    typename Smem_tile_q::Fragment frag_q[2][Mma_tile_p::MMAS_M];
    smem_q.load(frag_q[0], 0);

    typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dv::MMAS_N];
    static_assert(Smem_tile_qt::Fragment::NUM_REGS == 4);
    static_assert(Mma_tile_dv::MMAS_K == 1);
    smem_qt.load(frag_qt[0], 0);

    // Load the fragments for K. We keep the data in registers during the entire kernel.
    typename Smem_tile_k::Fragment frag_k[2][Mma_tile_p::MMAS_N];
    smem_k.load(frag_k[0], 0);

    enum { BITS_PER_ELT_S = sizeof(fmha::A_type) * 8 };

    // Create the object to do the softmax.
    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;
    Softmax softmax(
        params, &smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], bidb, tidx);

    enum { THREADS_PER_ROW = 32 };
    enum { M = Mma_tile_p::MMAS_M };
    enum { N = Mma_tile_p::MMAS_N };

    // Declare the accumulators for the 2nd gemm.
    fmha::Fragment_accumulator acc_dv[Mma_tile_dv::MMAS_M][Mma_tile_dv::MMAS_N];
    fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_dv::WARPS_K>::apply(acc_dv);

    // Load over the entire sequence length.
    for(int l = 0; l < nl_traits.num_steps_;l++) {

        uint4 s_regs[M][N];
        gmem_s.load(s_regs, mask);
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_q.load(frag_q[ki & 1], ki);
            smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }

        smem_s.store(s_regs);

        // Declare the accumulators for the 1st gemm.
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm(acc_p, frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
        // Trigger the load for the next Q values. We're using double buffering, so reading qt is safe
        if(l < nl_traits.num_steps_ - 1) {
            smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_q);
        }
        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack(acc_p);

        float s_mat[2 * M][4 * N];

        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                uint4 &dst = s_regs[mi][ni];
                fmha::half2_to_float2(s_mat[2 * mi + 0][4 * ni + 0], s_mat[2 * mi + 0][4 * ni + 1], dst.x);
                fmha::half2_to_float2(s_mat[2 * mi + 0][4 * ni + 2], s_mat[2 * mi + 0][4 * ni + 3], dst.y);
                fmha::half2_to_float2(s_mat[2 * mi + 1][4 * ni + 0], s_mat[2 * mi + 1][4 * ni + 1], dst.z);
                fmha::half2_to_float2(s_mat[2 * mi + 1][4 * ni + 2], s_mat[2 * mi + 1][4 * ni + 3], dst.w);
            }
        }

        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ii = 0; ii < 2; ii++ ) {
                #pragma unroll
                for( int ni = 0; ni < N; ni++ ) {
                    #pragma unroll
                    for( int jj = 0; jj < 4; jj++ ) {
                         float & s_dmask = s_mat[2 * mi + ii][4 * ni + jj];
                        const bool drop = reinterpret_cast<const uint32_t &>(s_dmask) & 0x80000000;
                        const float d_s= drop ? 0.f : softmax.elt_[2 * mi + ii][4 * ni + jj] * params.rp_dropout;
                        s_dmask = fabsf(s_dmask);
                        softmax.elt_[2 * mi + ii][4 * ni + jj] = d_s * (s_dmask);
                    }
                }
            }
        }

        float p_sum[2 * M];
        softmax.reduce_sum(p_sum);

        const float scalef = reinterpret_cast<const float &>(params.scale_softmax);
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ii = 0; ii < 2; ii++ ) {
                #pragma unroll
                for( int ni = 0; ni < N; ni++ ) {
                    #pragma unroll
                    for( int jj = 0; jj < 4; jj++ ) {
                        softmax.elt_[2 * mi + ii][4 * ni + jj] -= p_sum[2 * mi + ii] * (s_mat[2 * mi + ii][4 * ni + jj]) ;
                        softmax.elt_[2 * mi + ii][4 * ni + jj] *= scalef;
                    }
                }
            }
        }

        typename Smem_tile_st::Fragment frag_s[Mma_tile_dv::MMAS_K][Mma_tile_dv::MMAS_M];
        smem_s.load(frag_s);
        for( int ki = 0; ki < Mma_tile_dv::MMAS_K; ki++ ) {
            for( int mi = 0; mi < Mma_tile_dv::MMAS_M; mi++ ) {
                for( int ii = 0; ii < Smem_tile_st::Fragment::NUM_REGS; ii++ ) {
                    frag_s[ki][mi].reg(ii) = fmha::hmul2(frag_s[ki][mi].reg(ii), params.scale_dropout);
                    frag_s[ki][mi].reg(ii) = fmha::hrelu2(frag_s[ki][mi].reg(ii));
                }
            }
        }

        gmem_s.store(softmax.elt_, mask);
        gmem_s.move();

        static_assert(Mma_tile_dv::MMAS_K == 1);  // DEBUG
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dv::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_qt.load(frag_qt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dv, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dv::MMAS_K;
            fmha::gemm(acc_dv, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }
        // Commit the values for Q into shared memory.
        if(l < nl_traits.num_steps_ - 1) {
            gmem_q.commit(smem_q);
        }

        // Make sure we are reading from the correct buffer.
        smem_q.move_to_next_read_buffer();
        smem_qt.move_to_next_read_buffer();

        // Make sure the data is in shared memory.
        __syncthreads();

        // Trigger the loads for the values of Q for the next iteration.
        smem_q.load(frag_q[0], 0);
        smem_k.load(frag_k[0], 0);
        smem_qt.load(frag_qt[0], 0);

    }  // Outer loop over the sequence length.

    // Epilogue for dV = (S * D)' * dout'. We're fully exposed to this!

    // Epilogue swizzle for dV
    Smem_tile_dv smem_dv(&smem_[Kernel_traits::Smem_tile_q::BYTES_PER_TILE], tidx);
    smem_dv.store(acc_dv);

    __syncthreads();

    uint4 dv_out[Smem_tile_dv::NUM_LDS];
    smem_dv.load(dv_out);
    Qkv_params dv_params;
    dv_params.qkv_ptr = params.dkv_ptr;
    dv_params.qkv_stride_in_bytes = params.h * 2 * CHUNKS * params.d * sizeof(half);
    dv_params.h = params.h;
    Gmem_tile_dv gmem_dv(dv_params, nl_traits.get_idx_dv(), binfo, tidx);
    gmem_dv.store(dv_out);
}

template<int CHUNKS, typename Kernel_traits, typename Params>
inline __device__ void compute_dq_dk_1xN_nl(const Params &params) {

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dk = fmha::Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;

    static_assert(Cta_tile_dk::M == 512 || Cta_tile_dk::M == 384 || Cta_tile_dk::M == 256 || Cta_tile_dk::M == 128);
    static_assert(Cta_tile_dk::N == 64);
    static_assert(Cta_tile_dk::K == 16);

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    using Mma_tile_o = fmha::Hmma_tile<Cta_tile_o>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dk = fmha::Hmma_tile<Cta_tile_dk>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to swizzle Q.
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle K.
    using Smem_tile_k = typename Kernel_traits::Smem_tile_v;  // K is used like V in fprop

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = Gmem_tile_dq<Cta_tile_o>;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    // The global memory tile to store dK.
    using Gmem_tile_dk = fmha::Gmem_tile_qkv<typename Kernel_traits::Cta_tile_o, 
                                             fmha::BITS_PER_ELEMENT_B, 
                                             Cta_tile_p::N, //S, 
                                             Cta_tile_p::K, //D, 
                                             2*CHUNKS>;

    // The shared memory tile to swizzle dK.
    using Smem_tile_dk = fmha::Smem_tile_mma_epilogue<Cta_tile_dk>;
    static_assert(Smem_tile_dk::NUM_LDS == Gmem_tile_dk::LDGS);
    static_assert(Smem_tile_dk::THREADS_PER_ROW == Gmem_tile_dk::THREADS_PER_ROW);

    // The shared memory tile to reload Q transposed.
    using Smem_tile_qt = fmha::Smem_tile_b<Cta_tile_dk, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 1>;

    // The global memory tile to load dP, stored in S
    using Gmem_tile_s = Gmem_tile_mma_s<Cta_tile_p>;
    // The shared memory tile to transpose dP.
    using Smem_tile_st = Smem_tile_mma_transposed<Cta_tile_p>;  

    using Noloop = Noloop_traits<CHUNKS, Cta_tile_p>;

    enum { M = Mma_tile_p::MMAS_M };
    enum { N = Mma_tile_p::MMAS_N };
    static_assert(M == Mma_tile_o::MMAS_M);
    static_assert(N == Mma_tile_o::MMAS_K);
    // Shared memory.
    extern __shared__ char smem_[];

    const int bidc = blockIdx.z;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.x;
    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() )
        return;

    fmha::Mask<Cta_tile_p> mask(params, binfo, tidx);

    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params, 0, binfo, tidx);
    // Allocate the shared memory tile loader for Q (as B).
    Smem_tile_qt smem_qt(&smem_[0], tidx);
    // Allocate the global memory tile loader for dP.
    Gmem_tile_s gmem_s(params, binfo, tidx);
    // Allocate the shared memory tile loader for dP.
    Smem_tile_st smem_s(&smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params, 1, binfo, tidx);
    // Allocate the shared memory tile loader for K.
    Smem_tile_k smem_k(&smem_[Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params, binfo, tidx);
    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE], tidx);

    Noloop nl_traits(bidc, binfo);

    nl_traits.move_all(gmem_q, gmem_o, gmem_s);

    // Trigger the loads for Q.
    gmem_q.load(smem_qt);
    // Trigger the loads for K.
    gmem_k.load(smem_k);

    uint4 s_regs[M][N];
    gmem_s.load(s_regs, mask);

    // Commit the data for Q and K to shared memory.
    gmem_q.commit(smem_qt);
    gmem_k.commit(smem_k);

    // Make sure the data is in shared memory.
    __syncthreads();

    typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dk::MMAS_N];
    smem_qt.load(frag_qt[0], 0);
    typename Smem_tile_k::Fragment frag_k[2][Mma_tile_o::MMAS_N];
    smem_k.load(frag_k[0], 0);

    enum { BITS_PER_ELT_S = sizeof(fmha::A_type) * 8 };

    enum { THREADS_PER_ROW = 32 };

    // Declare the accumulators for the 2nd gemm.
    fmha::Fragment_accumulator acc_dk[Mma_tile_dk::MMAS_M][Mma_tile_dk::MMAS_N];
    fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_dk::WARPS_K>::apply(acc_dk);

    // Load over the entire sequence length.
    for(int l=0;l < nl_traits.num_steps_; l++) {

        // Pack dP as Fragment_a
        fmha::Fragment_a<fmha::Row> frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                uint4 &dst = s_regs[mi][ni];
                frag_p[ni][mi].reg(0) = dst.x;
                frag_p[ni][mi].reg(1) = dst.z;
                frag_p[ni][mi].reg(2) = dst.y;
                frag_p[ni][mi].reg(3) = dst.w;
            }
        }
        smem_s.store(s_regs);
        if(l < nl_traits.num_steps_- 1) {
            // Load next part of S
            gmem_s.move();
            gmem_s.load(s_regs, mask);
            // Trigger the load for the next Q values.
            smem_qt.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load(smem_qt);
        }
        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        // Do this part of O = P^T * V^T. dQ = dP x dK
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_o::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_o, frag_p[ki - 1], frag_k[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_o::MMAS_K;
            fmha::gemm(acc_o, frag_p[ki - 1], frag_k[(ki - 1) & 1]);
        }

        static_assert(Gmem_tile_o::LOOPS == 1); //DEBUG
        // Loop over MMAS_M.
        #pragma unroll
        for( int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii ) {

            // Swizzle the elements and do the final reduction.
            smem_o.store(acc_o, ii);

            // Make sure the data is in shared memory.
            __syncthreads();

            // Load from shared memory.
            uint4 out[Gmem_tile_o::STGS_PER_LOOP];
            smem_o.load(out);

            // Make sure the data was read from shared memory.
            if( ii < Gmem_tile_o::LOOPS - 1 ) {
                __syncthreads();
            }

            // Output the values.
            gmem_o.store(out, ii);
        }

        // Move to the next part of the output.
        gmem_o.move();

        typename Smem_tile_st::Fragment frag_s[Mma_tile_dk::MMAS_K][Mma_tile_dk::MMAS_M];
        smem_s.load(frag_s);

        static_assert(Mma_tile_dk::MMAS_K == 1);  // DEBUG

        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dk::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_qt.load(frag_qt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm(acc_dk, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dk::MMAS_K;
            fmha::gemm(acc_dk, frag_s[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Commit the values for Q into shared memory.
        if(l < nl_traits.num_steps_- 1) {
            gmem_q.commit(smem_qt);
            __syncthreads();
            // Trigger the loads for the values of Q for the next iteration.
            smem_qt.load(frag_qt[0], 0);
            smem_k.load(frag_k[0], 0);
        }

    }  // Outer loop over the sequence length.

    // Epilogue for dK = dP' * dq. We're fully exposed to this!

    // Epilogue swizzle for dK
    Smem_tile_dk smem_dk(&smem_[0], tidx);
    smem_dk.store(acc_dk);
    
    __syncthreads();
    
    uint4 dk_out[Smem_tile_dk::NUM_LDS];
    smem_dk.load(dk_out);
    Qkv_params dk_params;
    dk_params.qkv_ptr = params.dkv_ptr;
    dk_params.qkv_stride_in_bytes = params.h * 2 * CHUNKS * params.d * sizeof(half);
    dk_params.h = params.h;
    Gmem_tile_dk gmem_dk(dk_params, nl_traits.get_idx_dk(), binfo, tidx);
    gmem_dk.store(dk_out);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
