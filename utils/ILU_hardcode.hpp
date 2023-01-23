#ifndef GRAPES_UTILS_ILU_HARDCODE_HPP
#define GRAPES_UTILS_ILU_HARDCODE_HPP

#include "common.hpp"
#include "kernels_3d7.hpp"
#include "kernels_3d15.hpp"
#include "kernels_3d19.hpp"
#include "kernels_3d27.hpp"

#define NEON_LEN 4

template<typename idx_t, typename data_t, typename oper_t>
void struct_2d5_trsv_forward_hardCode(const data_t * l, const oper_t * b, oper_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos, bool shuffled=false)
{
    // 假定不会在一头一尾处的x越界而访问到nan/inf值，也假定在边界位置的超出本进程范围的L元素为0
    assert(lnz == 4);
    assert(!shuffled);
    if (sizeof(data_t) == sizeof(oper_t)) {
        for (idx_t i = 0; i < dim_0; i++) {
            const oper_t * b_col = b + i * dim_1;
                  oper_t * x_col = x + i * dim_1;
            const oper_t * x_col_prev = x_col - dim_1;
            const data_t * L_col = l + lnz * (i * dim_1);
            for (idx_t j = 0; j < dim_1; j++) {
                oper_t res = * b_col;
                res -= L_col[0] * x_col_prev[-1];
                res -= L_col[1] * x_col_prev[ 0];
                res -= L_col[2] * x_col_prev[ 1];
                res -= L_col[3] * x_col[-1];
                *x_col = res;

                b_col ++;
                x_col ++; x_col_prev ++;
                L_col += lnz;
            }// j loop
        }
    }
    else {
#if 0
        assert(sizeof(data_t) == 2 && shuffled);
        const idx_t max_j = ((dim_1)&(~(NEON_LEN-1)));
        const idx_t tot_elems = dim_0 * dim_1;

        for (idx_t i = 0; i < dim_0; i++) {
            const idx_t offset = i * dim_1;
            const oper_t * b_i = b + offset;
                  oper_t * x_i = x + offset;
            const oper_t * x1 = x_i - dim_1;//, * x0 = x1 - 1, * x2 = x1 + 1;
            const data_t * L0 = l + offset    , * L1 = L0 + tot_elems,
                         * L2 = L1 + tot_elems, * L3 = L2 + tot_elems;
            {// 预取 一条cacheline有64字节，相当于32个半精度数，16个单精度数
            __builtin_prefetch(L0 + 32, 0, 0); __builtin_prefetch(L0 + 64, 0, 0);
            __builtin_prefetch(L1 + 32, 0, 0); __builtin_prefetch(L1 + 64, 0, 0);
            __builtin_prefetch(L2 + 32, 0, 0); __builtin_prefetch(L2 + 64, 0, 0);
            __builtin_prefetch(L3 + 32, 0, 0); __builtin_prefetch(L3 + 64, 0, 0);
            __builtin_prefetch(x_i + 16, 0, 0); __builtin_prefetch(x_i + 32, 0, 0);
            __builtin_prefetch(x_i + 48, 0, 0); __builtin_prefetch(x_i + 64, 0, 0);
            __builtin_prefetch(b_i + 16, 0, 0); __builtin_prefetch(b_i + 32, 0, 0);
            __builtin_prefetch(b_i + 48, 0, 0); __builtin_prefetch(b_i + 64, 0, 0);
            }

            register idx_t j = 0;
            for ( ; j < max_j; j += NEON_LEN) {
                float16x4_t L0_16 = vld1_f16((__fp16*) L0), L1_16 = vld1_f16((__fp16*) L1),
                            L2_16 = vld1_f16((__fp16*) L2), L3_16 = vld1_f16((__fp16*) L3);
                float32x4_t L0_32 = vcvt_f32_f16(L0_16), L1_32 = vcvt_f32_f16(L1_16),
                            L2_32 = vcvt_f32_f16(L2_16), L3_32 = vcvt_f32_f16(L3_16);
                float32x4_t x0_32 = vld1q_f32(x1 - 1), x1_32 = vld1q_f32(x1),
                            x2_32 = vld1q_f32(x1 + 1);
                // 先计算非本柱的
                float32x4_t tmp = vld1q_f32(b_i);
                tmp = vmlsq_f32(tmp, L0_32, x0_32);
                tmp = vmlsq_f32(tmp, L1_32, x1_32);
                tmp = vmlsq_f32(tmp, L2_32, x2_32);

                x_i[0] = vgetq_lane_f32(tmp, 0) - vgetq_lane_f32(L3_32, 0) * x_i[-1];
                x_i[1] = vgetq_lane_f32(tmp, 1) - vgetq_lane_f32(L3_32, 1) * x_i[ 0];
                x_i[2] = vgetq_lane_f32(tmp, 2) - vgetq_lane_f32(L3_32, 2) * x_i[ 1];
                x_i[3] = vgetq_lane_f32(tmp, 3) - vgetq_lane_f32(L3_32, 3) * x_i[ 2];

                L0 += NEON_LEN; L1 += NEON_LEN; L2 += NEON_LEN; L3 += NEON_LEN;
                x1 += NEON_LEN;
                b_i += NEON_LEN; x_i += NEON_LEN; 
            }// 循环结束时 j == max_j
            for (j = 0; j < dim_1 - max_j; j++) {
                oper_t res = b_i[j] 
                    - L0[j] * x1[j-1] - L1[j] * x1[j] - L2[j] * x1[j+1] - L3[j] * x_i[j-1];
                x_i[j] = res;
            }// j loop
        }
#endif
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void struct_2d5_trsv_backward_hardCode(const data_t * u, const oper_t * b, oper_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos, bool shuffled=false)
{
    assert(rnz == 5);
    assert(!shuffled);
    if (sizeof(data_t) == sizeof(oper_t)) {
        for (idx_t i = dim_0 - 1; i >= 0; i--) {
            const oper_t * b_col = b + i * dim_1 + dim_1 - 1;
                  oper_t * x_col = x + i * dim_1 + dim_1 - 1;
            const oper_t * x_col_next = x_col + dim_1;
            const data_t * U_col = u + rnz *(i * dim_1 + dim_1 - 1);
            for (idx_t j = dim_1 - 1; j >= 0; j--) {// 注意这个遍历顺序是从后到前
                oper_t res = * b_col;
                oper_t para = U_col[0];

                res -= U_col[1] * x_col[1];
                res -= U_col[2] * x_col_next[-1];
                res -= U_col[3] * x_col_next[ 0];
                res -= U_col[4] * x_col_next[ 1];

                *x_col = res / para;

                b_col --;
                x_col --; x_col_next --;
                U_col -= rnz;
            }// j loop
        }
    }
    else {
#if 0
        assert(sizeof(data_t) == 2 && shuffled);
        const idx_t min_j = ((dim_1) & (NEON_LEN-1));
        const idx_t tot_elems = dim_0 * dim_1;
        float32x4_t v_ones = vdupq_n_f32(1.0);

        for (idx_t i = dim_0 - 1; i >= 0; i--) {
            const idx_t offset = i * dim_1 + dim_1;// 注意在这里先不减，进到循环里再减
            const oper_t * b_i = b + offset;
                  oper_t * x_i = x + offset;// x4
            const oper_t * x7 = x_i + dim_1;
            const data_t* U4 = u + offset,
                        * U5 = U4 + tot_elems, * U6 = U5 + tot_elems,
                        * U7 = U6 + tot_elems, * U8 = U7 + tot_elems;
            {// 预取 一条cacheline有64字节，相当于32个半精度数，16个单精度数
            __builtin_prefetch(U4 - 32, 0, 0); __builtin_prefetch(U4 - 64, 0, 0);
            __builtin_prefetch(U5 - 32, 0, 0); __builtin_prefetch(U5 - 64, 0, 0);
            __builtin_prefetch(U6 - 32, 0, 0); __builtin_prefetch(U6 - 64, 0, 0);
            __builtin_prefetch(U7 - 32, 0, 0); __builtin_prefetch(U7 - 64, 0, 0);
            __builtin_prefetch(U8 - 32, 0, 0); __builtin_prefetch(U8 - 64, 0, 0);
            __builtin_prefetch(x_i - 16, 0, 0); __builtin_prefetch(x_i - 32, 0, 0);
            __builtin_prefetch(x_i - 48, 0, 0); __builtin_prefetch(x_i - 64, 0, 0);
            __builtin_prefetch(b_i - 16, 0, 0); __builtin_prefetch(b_i - 32, 0, 0);
            __builtin_prefetch(b_i - 48, 0, 0); __builtin_prefetch(b_i - 64, 0, 0);
            }

            register idx_t j = dim_1;
            for ( ; j > min_j; j -= NEON_LEN) {// 注意这个遍历顺序是从后到前
                U4 -= NEON_LEN; U5 -= NEON_LEN; U6 -= NEON_LEN; U7 -= NEON_LEN; U8 -= NEON_LEN;
                x_i -= NEON_LEN; b_i -= NEON_LEN; x7 -= NEON_LEN;

                float16x4_t U4_16 = vld1_f16((__fp16*) U4), U5_16 = vld1_f16((__fp16*) U5),
                            U6_16 = vld1_f16((__fp16*) U6), U7_16 = vld1_f16((__fp16*) U7),
                            U8_16 = vld1_f16((__fp16*) U8);
                float32x4_t U4_32 = vcvt_f32_f16(U4_16), U5_32 = vcvt_f32_f16(U5_16),
                            U6_32 = vcvt_f32_f16(U6_16), U7_32 = vcvt_f32_f16(U7_16),
                            U8_32 = vcvt_f32_f16(U8_16);
                float32x4_t x6_32 = vld1q_f32(x7 - 1), x7_32 = vld1q_f32(x7),
                            x8_32 = vld1q_f32(x7 + 1);
                // 先计算非本柱的
                float32x4_t tmp = vld1q_f32(b_i);
                tmp = vmlsq_f32(tmp, U6_32, x6_32);
                tmp = vmlsq_f32(tmp, U7_32, x7_32);
                tmp = vmlsq_f32(tmp, U8_32, x8_32);

                U4_32 = vdivq_f32(v_ones, U4_32);
                tmp = vmulq_f32(tmp, U4_32);
                U5_32 = vmulq_f32(U5_32, U4_32);

                x_i[3] = vgetq_lane_f32(tmp, 3) - vgetq_lane_f32(U5_32, 3) * x_i[4];
                x_i[2] = vgetq_lane_f32(tmp, 2) - vgetq_lane_f32(U5_32, 2) * x_i[3];
                x_i[1] = vgetq_lane_f32(tmp, 1) - vgetq_lane_f32(U5_32, 1) * x_i[2];
                x_i[0] = vgetq_lane_f32(tmp, 0) - vgetq_lane_f32(U5_32, 0) * x_i[1];
            }// 循环结束时 j==min_j
            assert(j == min_j);
            U4 -= min_j; U5 -= min_j; U6 -= min_j; U7 -= min_j; U8 -= min_j;
            x_i -= min_j; b_i -= min_j; x7 -= min_j;
            for (j = min_j - 1; j >= 0; j--) {
                oper_t res = b_i[j];
                oper_t para = U4[j];
                res -= U5[j] * x_i[j+1];
                res -= U6[j] * x7[j-1];
                res -= U7[j] * x7[j  ];
                res -= U8[j] * x7[j+1];
                x_i[j] = res / para;
            }
        }
#endif
    }
}


#undef NEON_LEN


template<typename idx_t, typename data_t, typename oper_t>
void group_sptrsv_2d5_levelbased_forward_hardCode(const data_t * L_3D, const oper_t * B_3D, oper_t * X_3D, const idx_t num_slices,
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos, 
    const idx_t group_size, const int * group_tid_arr, const idx_t * slc_id_arr)
{
    assert(lnz == 4);
    idx_t * flag_3D = new idx_t [(dim_0 + 1) * num_slices];
    for (idx_t s = 0; s < num_slices; s++)
        flag_3D[s * (dim_0 + 1)] = dim_1 - 1;// 每一个面的边界柱标记已完成
    
    // 并行区：一次性开够所有线程
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for collapse(2) schedule(static)
        for (idx_t s = 0; s < num_slices; s++)// 标记每一面
        for (idx_t i = 0; i < dim_0; i++)
                flag_3D[s * (dim_0 + 1) + i + 1] = -1;// 的第i柱完成到哪个高度，初始化为-1

        int glb_tid = omp_get_thread_num();// 全局线程号
        int nt = group_size;
        int tid = group_tid_arr[glb_tid];
        idx_t slc_id = slc_id_arr[glb_tid];
        
        if (slc_id != -1) {
            // 根据自己负责的面定位到2D数据
            idx_t * flag = flag_3D + slc_id * (dim_0 + 1);
            register const data_t * l = L_3D + slc_id * dim_0 * dim_1 * lnz;
            register const oper_t * b = B_3D + slc_id * dim_0 * dim_1;
            register       oper_t * x = X_3D + slc_id * dim_0 * dim_1;

            // 各自开始计算
            idx_t last_il = dim_0, last_jl = -1;
            idx_t last_ir = -1   , last_jr = dim_1;
            idx_t nlevs = dim_1 + (dim_0 - 1) * 2;// 根据2d5的特征确定一共有多少个level

            idx_t ibeg, jbeg, t_beg, t_end;
            {   idx_t ilev = 0;// cold start：计算第0层的任务范围和边界位置
                ibeg = MIN(ilev >> 1, dim_0 - 1);
                jbeg = ilev - 2 * ibeg;
                idx_t ntasks = MIN(ibeg + 1, ((dim_1-1 - jbeg) >> 1) + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                t_end = t_beg + my_cnt;
            }

            /* // no effect since no vectorization could be utilized
            if (sizeof(data_t) == sizeof(oper_t)) {
            */
                for (idx_t ilev = 0; ilev < nlevs; ilev++) {
                    idx_t next_ilev = ilev + 1;
                    bool write_beg = true, write_end = true;// 标记右边界和左边界是否需要原子写回
                    
                    idx_t next_ir, next_jr, next_il, next_jl;
                    idx_t next_ibeg, next_jbeg, next_t_beg, next_t_end;

                    if (next_ilev < nlevs) {// 计算下一层的任务范围和边界位置
                        next_ibeg = MIN(next_ilev >> 1, dim_0 - 1);
                        next_jbeg = next_ilev - 2 * next_ibeg;
                        idx_t next_ntasks = MIN(next_ibeg + 1, ((dim_1-1 - next_jbeg) >> 1) + 1);
                        // 确定自己分到的task范围
                        idx_t next_my_cnt = next_ntasks / nt;
                        next_t_beg = tid * next_my_cnt;
                        idx_t next_remain = next_ntasks - next_my_cnt * nt;
                        next_t_beg += MIN(next_remain, tid);
                        if (tid < next_remain) next_my_cnt ++;
                        next_t_end = next_t_beg + next_my_cnt;

                        // 下面这5行是为计算下一层边界位置而引入的额外开销
                        next_ir = next_ibeg - next_t_beg;
                        next_jr = next_jbeg + (next_t_beg << 1);

                        next_t_end -= 1;// 开区间转实
                        next_il = next_ibeg - next_t_end;
                        next_jl = next_jbeg + (next_t_end << 1);
                        next_t_end += 1;// 恢复开区间，以为后续update
                        {// 预取下一层的数据
                        idx_t gap = next_ir * dim_1 + next_jr;
                        const oper_t * b_ptr = b + gap;
                        __builtin_prefetch(b_ptr, 0, 0); __builtin_prefetch(b_ptr + 16, 0, 0); __builtin_prefetch(b_ptr + 32, 0, 0); __builtin_prefetch(b_ptr + 48, 0, 0);
                        const data_t * L_ptr = l + lnz * gap;
                        __builtin_prefetch(L_ptr, 0, 0); __builtin_prefetch(L_ptr + 32, 0, 0);
                        }
                    }

                    for (idx_t it = t_end - 1; it >= t_beg; it--) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        const idx_t offset = i * dim_1 + j;
                        oper_t res = b[offset];
                        const data_t * L_ptr = l + lnz * offset;
                        register oper_t L0 = L_ptr[0], L1 = L_ptr[1], L2 = L_ptr[2], L3 = L_ptr[3];
                        oper_t * x_ptr = x + offset, * x_left = x_ptr - dim_1;

                        if (it == t_beg) {// 右边界只需检查S依赖
                            if (!(i <= last_ir && j > last_jr)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) < j-1) {  }
                            }
                            last_ir = i;
                            last_jr = j;
                            // 同时只可能作为别人的WN依赖：下一层依赖于该点作WN依赖的位置仍由自己做<=>next_ir>last_ir且next_jr<last_jr
                            write_beg = (next_ir <= last_ir|| next_jr >= last_jr);
                        }
                        if (it == t_end - 1) {// 左边界只需检查WN依赖
                            if (!(i > last_il && j < last_jl)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                idx_t wait_left_j = (j==dim_1-1) ? j : (j+1);
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) < wait_left_j) {  }
                            }
                            last_il = i;
                            last_jl = j;
                            // 同时可能作为别人的S依赖：下一层依赖于该点作S依赖的位置仍由自己做<=>next_il==last_il且next_jl>last_jl
                            write_end = (next_il != last_il || next_jl <= last_jl);
                        }
                        
                        res -= L0 * x_left[-1];
                        res -= L1 * x_left[ 0];
                        res -= L2 * x_left[ 1];
                        res -= L3 * x_ptr[-1];
                        *x_ptr = res;

                        if ((it == t_beg && write_beg) || (it == t_end - 1 && write_end))// 只要一头一尾的原子操作括起来，中间的写回可以不按序
                            __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                        else 
                            flag[i+1] = j;
                    }
                    // update for next lev
                    ibeg = next_ibeg;
                    jbeg = next_jbeg;
                    t_beg = next_t_beg;
                    t_end = next_t_end;
                }// level loop
            /* // no effect since no vectorization could be utilized
            }// 32-bit
            else {
                assert(sizeof(data_t) == 2);
                for (idx_t ilev = 0; ilev < nlevs; ilev++) {
                    idx_t next_ilev = ilev + 1;
                    bool write_beg = true, write_end = true;// 标记右边界和左边界是否需要原子写回
                    
                    idx_t next_ir, next_jr, next_il, next_jl;
                    idx_t next_ibeg, next_jbeg, next_t_beg, next_t_end;

                    if (next_ilev < nlevs) {// 计算下一层的任务范围和边界位置
                        next_ibeg = MIN(next_ilev >> 1, dim_0 - 1);
                        next_jbeg = next_ilev - 2 * next_ibeg;
                        idx_t next_ntasks = MIN(next_ibeg + 1, ((dim_1-1 - next_jbeg) >> 1) + 1);
                        // 确定自己分到的task范围
                        idx_t next_my_cnt = next_ntasks / nt;
                        next_t_beg = tid * next_my_cnt;
                        idx_t next_remain = next_ntasks - next_my_cnt * nt;
                        next_t_beg += MIN(next_remain, tid);
                        if (tid < next_remain) next_my_cnt ++;
                        next_t_end = next_t_beg + next_my_cnt;

                        // 下面这5行是为计算下一层边界位置而引入的额外开销
                        next_ir = next_ibeg - next_t_beg;
                        next_jr = next_jbeg + (next_t_beg << 1);

                        next_t_end -= 1;// 开区间转实
                        next_il = next_ibeg - next_t_end;
                        next_jl = next_jbeg + (next_t_end << 1);
                        next_t_end += 1;// 恢复开区间，以为后续update
                        {// 预取下一层的数据
                        idx_t gap = next_ir * dim_1 + next_jr;
                        const oper_t * b_ptr = b + gap;
                        __builtin_prefetch(b_ptr, 0, 0); __builtin_prefetch(b_ptr + 16, 0, 0); __builtin_prefetch(b_ptr + 32, 0, 0); __builtin_prefetch(b_ptr + 48, 0, 0);
                        const data_t * L_ptr = l + lnz * gap;
                        __builtin_prefetch(L_ptr, 0, 0); __builtin_prefetch(L_ptr + 32, 0, 0);
                        }
                    }

                    for (idx_t it = t_end - 1; it >= t_beg; it--) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        register oper_t res = b[i * dim_1 + j];
                        const data_t * L_ptr = l + lnz * (i * dim_1 + j);
                        register oper_t * x_ptr = x + i * dim_1 + j;
                        float16x4_t L_16 = vld1_f16((__fp16*)L_ptr);
                        float32x4_t L_32 = vcvt_f32_f16(L_16);

                        if (it == t_beg) {// 右边界只需检查S依赖
                            if (!(i <= last_ir && j > last_jr)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) < j-1) {  }
                            }
                            last_ir = i;
                            last_jr = j;
                            // 同时只可能作为别人的WN依赖：下一层依赖于该点作WN依赖的位置仍由自己做<=>next_ir>last_ir且next_jr<last_jr
                            write_beg = (next_ir <= last_ir|| next_jr >= last_jr);
                        }
                        if (it == t_end - 1) {// 左边界只需检查WN依赖
                            if (!(i > last_il && j < last_jl)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                idx_t wait_left_j = (j==dim_1-1) ? j : (j+1);
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) < wait_left_j) {  }
                            }
                            last_il = i;
                            last_jl = j;
                            // 同时可能作为别人的S依赖：下一层依赖于该点作S依赖的位置仍由自己做<=>next_il==last_il且next_jl>last_jl
                            write_end = (next_il != last_il || next_jl <= last_jl);
                        }

                        float32x4_t x_32 = vld1q_f32(x_ptr - dim_1 - 1);
                        x_32 = vmulq_f32(L_32, x_32);
                        res -= (vgetq_lane_f32(x_32, 0) + vgetq_lane_f32(x_32, 1) + vgetq_lane_f32(x_32, 2));
                        res -=  vgetq_lane_f32(L_32, 3) * x_ptr[-1];
                        *x_ptr = res;

                        if ((it == t_beg && write_beg) || (it == t_end - 1 && write_end))// 只要一头一尾的原子操作括起来，中间的写回可以不按序
                            __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                        else 
                            flag[i+1] = j;
                    }
                    // update for next lev
                    ibeg = next_ibeg;
                    jbeg = next_jbeg;
                    t_beg = next_t_beg;
                    t_end = next_t_end;
                }// level loop
            }// 16-bit
            */
        }
        else {
            // if (my_pid == 0) printf(" t%d/%d got no work\n", glb_tid, omp_get_num_threads());
            assert(tid == -1);
        }
    }// omp para
    delete flag_3D;
}

template<typename idx_t, typename data_t, typename oper_t>
void group_sptrsv_2d5_levelbased_backward_hardCode(const data_t * U_3D, const oper_t * B_3D, oper_t * X_3D, const idx_t num_slices, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos,
    const idx_t group_size, const int * group_tid_arr, const idx_t * slc_id_arr)
{
    assert(rnz == 5);
    idx_t * flag_3D = new idx_t [(dim_0 + 1) * num_slices];
    for (idx_t s = 0; s < num_slices; s++)
        flag_3D[s * (dim_0 + 1) + dim_0] = 0;// 边界柱标记已完成

    // 并行区：一次性开够所有线程
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for collapse(2) schedule(static)
        for (idx_t s = 0; s < num_slices; s++)// 标记每一面
        for (idx_t i = dim_0; i >= 1; i--) 
            flag_3D[s * (dim_0 + 1) + i - 1] = dim_1;// 的第i柱完成到哪个高度，初始化

        int glb_tid = omp_get_thread_num();// 全局线程号
        int nt = group_size;
        int tid = group_tid_arr[glb_tid];
        idx_t slc_id = slc_id_arr[glb_tid];

        if (slc_id != -1) {
            // 根据自己负责的面定位到2D数据
            idx_t * flag = flag_3D + slc_id * (dim_0 + 1);
            register const data_t * u = U_3D + slc_id * dim_0 * dim_1 * rnz;
            register const oper_t * b = B_3D + slc_id * dim_0 * dim_1;
            register       oper_t * x = X_3D + slc_id * dim_0 * dim_1;

            // 各自开始计算
            idx_t last_il = dim_0, last_jl = -1;
            idx_t last_ir = -1   , last_jr = dim_1;
            idx_t nlevs = dim_1 + (dim_0 - 1) * 2;// 根据2d5的特征确定一共有多少个level
            
            idx_t ibeg, jbeg, t_beg, t_end;
            {   idx_t ilev = nlevs - 1;// cold start：计算第0层的任务范围和边界位置
                ibeg = MIN(ilev >> 1, dim_0 - 1);
                jbeg = ilev - 2 * ibeg;
                idx_t ntasks = MIN(ibeg + 1, ((dim_1-1 - jbeg) >> 1) + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                t_end = t_beg + my_cnt;
            }
            /* // no effect since no vectorization could be utilized 
            if (sizeof(data_t) == sizeof(oper_t)) {
            */
                for (idx_t ilev = nlevs - 1; ilev >= 0; ilev--) {
                    idx_t next_ilev = ilev - 1;
                    bool write_beg = true, write_end = true;// 标记右边界和左边界是否需要原子写回
                    
                    idx_t next_ir, next_jr, next_il, next_jl;
                    idx_t next_ibeg, next_jbeg, next_t_beg, next_t_end;

                    if (next_ilev >= 0) {// 计算下一层的任务范围和边界位置
                        next_ibeg = MIN(next_ilev >> 1, dim_0 - 1);
                        next_jbeg = next_ilev - 2 * next_ibeg;
                        idx_t next_ntasks = MIN(next_ibeg + 1, ((dim_1-1 - next_jbeg) >> 1) + 1);
                        // 确定自己分到的task范围
                        idx_t next_my_cnt = next_ntasks / nt;
                        next_t_beg = tid * next_my_cnt;
                        idx_t next_remain = next_ntasks - next_my_cnt * nt;
                        next_t_beg += MIN(next_remain, tid);
                        if (tid < next_remain) next_my_cnt ++;
                        next_t_end = next_t_beg + next_my_cnt;

                        // 下面这5行是为计算下一层边界位置而引入的额外开销
                        next_ir = next_ibeg - next_t_beg;
                        next_jr = next_jbeg + (next_t_beg << 1);

                        next_t_end -= 1;// 开区间转实
                        next_il = next_ibeg - next_t_end;
                        next_jl = next_jbeg + (next_t_end << 1);
                        next_t_end += 1;// 恢复开区间，以为后续update
                        {// 预取下一层的数据
                        idx_t gap = next_il * dim_1 + next_jl;
                        const oper_t * b_ptr = b + gap;
                        __builtin_prefetch(b_ptr, 0, 0); __builtin_prefetch(b_ptr - 16, 0, 0); __builtin_prefetch(b_ptr - 32, 0, 0); __builtin_prefetch(b_ptr - 48, 0, 0);
                        const data_t * U_ptr = u + rnz * gap;
                        __builtin_prefetch(U_ptr, 0, 0); __builtin_prefetch(U_ptr - 32, 0, 0);
                        }
                    }

                    for (idx_t it = t_beg; it < t_end; it++) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        const idx_t offset = i * dim_1 + j;
                        oper_t res = b[offset];
                        const data_t * U_ptr = u + rnz * offset;
                        register oper_t U0 = U_ptr[0], U1 = U_ptr[1], U2 = U_ptr[2], U3 = U_ptr[3], U4 = U_ptr[4];
                        oper_t * x_ptr = x + offset, * x_right = x_ptr + dim_1;

                        if (it == t_beg) {// 右边界只需检查ES依赖
                            if (!(i < last_ir && j > last_jr)) {
                                idx_t wait_right_j = (j==0) ? j : (j-1);
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) > wait_right_j) {  }
                            }
                            last_ir = i;
                            last_jr = j;
                            // 同时有可能作为别人的N依赖：下一层依赖于该点作N依赖的位置仍由自己做<=>next_ir==last_ir且next_jr<last_jr
                            write_beg = (next_ir != last_ir) || (next_jr >= last_jr); 
                        }
                        if (it == t_end - 1) {// 左边界只需检查N依赖
                            if (!(i >= last_il && j < last_jl)) {
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) > j+1) {  }
                                // while (flag[i] > j+1) {smp_rmb();} smp_rmb();
                            }
                            last_il = i;
                            last_jl = j;
                            // 同时有可能作为别人的ES依赖：下一层依赖于该点作N依赖的位置仍由自己做<=>next_il<last_il且next_jl>last_jl
                            write_end = (next_il == last_il) || (next_jl <= last_jl);
                        }
                        // 中间的不需等待
                        res -= U1 * x_ptr[1];
                        res -= U2 * x_right[-1];
                        res -= U3 * x_right[ 0];
                        res -= U4 * x_right[ 1];
                        *x_ptr = res / U0;

                        if ((it == t_beg && write_beg) || (it == t_end - 1 && write_end))
                            __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                        else
                            flag[i] = j;
                    }
                    // update for next lev
                    ibeg = next_ibeg;
                    jbeg = next_jbeg;
                    t_beg = next_t_beg;
                    t_end = next_t_end;
                }
            /*
            }// 32-bit
            else {
                for (idx_t ilev = nlevs - 1; ilev >= 0; ilev--) {
                    idx_t next_ilev = ilev - 1;
                    bool write_beg = true, write_end = true;// 标记右边界和左边界是否需要原子写回
                    
                    idx_t next_ir, next_jr, next_il, next_jl;
                    idx_t next_ibeg, next_jbeg, next_t_beg, next_t_end;

                    if (next_ilev >= 0) {// 计算下一层的任务范围和边界位置
                        next_ibeg = MIN(next_ilev >> 1, dim_0 - 1);
                        next_jbeg = next_ilev - 2 * next_ibeg;
                        idx_t next_ntasks = MIN(next_ibeg + 1, ((dim_1-1 - next_jbeg) >> 1) + 1);
                        // 确定自己分到的task范围
                        idx_t next_my_cnt = next_ntasks / nt;
                        next_t_beg = tid * next_my_cnt;
                        idx_t next_remain = next_ntasks - next_my_cnt * nt;
                        next_t_beg += MIN(next_remain, tid);
                        if (tid < next_remain) next_my_cnt ++;
                        next_t_end = next_t_beg + next_my_cnt;

                        // 下面这5行是为计算下一层边界位置而引入的额外开销
                        next_ir = next_ibeg - next_t_beg;
                        next_jr = next_jbeg + (next_t_beg << 1);

                        next_t_end -= 1;// 开区间转实
                        next_il = next_ibeg - next_t_end;
                        next_jl = next_jbeg + (next_t_end << 1);
                        next_t_end += 1;// 恢复开区间，以为后续update
                        {// 预取下一层的数据
                        idx_t gap = next_il * dim_1 + next_jl;
                        const oper_t * b_ptr = b + gap;
                        __builtin_prefetch(b_ptr, 0, 0); __builtin_prefetch(b_ptr - 16, 0, 0); __builtin_prefetch(b_ptr - 32, 0, 0); __builtin_prefetch(b_ptr - 48, 0, 0);
                        const data_t * U_ptr = u + rnz * gap;
                        __builtin_prefetch(U_ptr, 0, 0); __builtin_prefetch(U_ptr - 32, 0, 0);
                        }
                    }

                    for (idx_t it = t_beg; it < t_end; it++) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        oper_t res = b[i * dim_1 + j];
                        const data_t * U_ptr = u + rnz * (i * dim_1 + j);

                        oper_t para = U_ptr[0];
                        float16x4_t U_16 = vld1_f16((__fp16*) U_ptr + 1);// 跨过U0   
                        float32x4_t U_32 = vcvt_f32_f16(U_16);                 
                        oper_t * x_ptr = x + i * dim_1 + j;

                        if (it == t_beg) {// 右边界只需检查ES依赖
                            if (!(i < last_ir && j > last_jr)) {
                                idx_t wait_right_j = (j==0) ? j : (j-1);
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) > wait_right_j) {  }
                            }
                            last_ir = i;
                            last_jr = j;
                            // 同时有可能作为别人的N依赖：下一层依赖于该点作N依赖的位置仍由自己做<=>next_ir==last_ir且next_jr<last_jr
                            write_beg = (next_ir != last_ir) || (next_jr >= last_jr); 
                        }
                        if (it == t_end - 1) {// 左边界只需检查N依赖
                            if (!(i >= last_il && j < last_jl)) {
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) > j+1) {  }
                                // while (flag[i] > j+1) {smp_rmb();} smp_rmb();
                            }
                            last_il = i;
                            last_jl = j;
                            // 同时有可能作为别人的ES依赖：下一层依赖于该点作N依赖的位置仍由自己做<=>next_il<last_il且next_jl>last_jl
                            write_end = (next_il == last_il) || (next_jl <= last_jl);
                        }
                        // 中间的不需等待
                        float32x4_t x_32 = vld1q_f32(x_ptr + dim_1 - 2);// 使x[(i+1)*dim_1 + j-1]放在向量寄存器的第二道，与U_32匹配
                        x_32 = vmulq_f32(U_32, x_32);
                        res -= (vgetq_lane_f32(x_32, 1) + vgetq_lane_f32(x_32, 2) + vgetq_lane_f32(x_32, 3));
                        res -=  vgetq_lane_f32(U_32, 0) * x_ptr[1];
                        *x_ptr = res / para;

                        if ((it == t_beg && write_beg) || (it == t_end - 1 && write_end))
                            __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                        else
                            flag[i] = j;
                    }
                    // update for next lev
                    ibeg = next_ibeg;
                    jbeg = next_jbeg;
                    t_beg = next_t_beg;
                    t_end = next_t_end;
                }
            }// 16-bit
            */
        } else {
            assert(tid == -1);
        }
    }// omp para
}

#if 0

template<typename idx_t, typename data_t, typename oper_t>
void group_sptrsv_2d5_levelbased_forward_hardCode(const data_t * L_3D, const oper_t * B_3D, oper_t * X_3D, const idx_t num_slices,
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos, 
    const idx_t group_size, const int * group_tid_arr, const idx_t * slc_id_arr)
{
    assert(lnz == 4 && memcmp(check_2d5_forward, pos, sizeof(idx_t) * lnz * 2)==0);
    idx_t * flag_3D = new idx_t [(dim_0 + 1) * num_slices];
    for (idx_t s = 0; s < num_slices; s++)
        flag_3D[s * (dim_0 + 1)] = dim_1 - 1;// 每一个面的边界柱标记已完成
    
    // 并行区：一次性开够所有线程
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for collapse(2) schedule(static)
        for (idx_t s = 0; s < num_slices; s++)// 标记每一面
        for (idx_t i = 0; i < dim_0; i++)
                flag_3D[s * (dim_0 + 1) + i + 1] = -1;// 的第i柱完成到哪个高度，初始化为-1

        int glb_tid = omp_get_thread_num();// 全局线程号
        int nt = group_size;
        int tid = group_tid_arr[glb_tid];
        idx_t slc_id = slc_id_arr[glb_tid];
        
        if (slc_id != -1) {
            // 根据自己负责的面定位到2D数据
            idx_t * flag = flag_3D + slc_id * (dim_0 + 1);
            const data_t * l = L_3D + slc_id * dim_0 * dim_1 * lnz;
            const oper_t * b = B_3D + slc_id * dim_0 * dim_1;
                  oper_t * x = X_3D + slc_id * dim_0 * dim_1;

            // 各自开始计算
            idx_t last_il = dim_0, last_jl = -1;
            idx_t last_ir = -1   , last_jr = dim_1;
            idx_t nlevs = dim_1 + (dim_0 - 1) * 2;// 根据2d5的特征确定一共有多少个level

            if (sizeof(data_t) == sizeof(oper_t)) {
            // if (true) {
                for (idx_t ilev = 0; ilev < nlevs; ilev++) {
                    idx_t ibeg = MIN(ilev >> 1, dim_0 - 1);
                    idx_t jbeg = ilev - 2 * ibeg;
                    idx_t ntasks = MIN(ibeg + 1, ((dim_1-1 - jbeg) >> 1) + 1);
                    // 确定自己分到的task范围
                    idx_t my_cnt = ntasks / nt;
                    idx_t t_beg = tid * my_cnt;
                    idx_t remain = ntasks - my_cnt * nt;
                    t_beg += MIN(remain, tid);
                    if (tid < remain) my_cnt ++;
                    idx_t t_end = t_beg + my_cnt;

                    for (idx_t it = t_end - 1; it >= t_beg; it--) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        oper_t res = b[i * dim_1 + j];
                        const data_t * L_ptr = l + lnz * (i * dim_1 + j);

                        if (it == t_beg) {// 右边界只需检查S依赖
                            if (!(i <= last_ir && j > last_jr)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) < j-1) {  }
                            }
                            last_ir = i;
                            last_jr = j;
                        }
                        if (it == t_end - 1) {// 左边界只需检查WN依赖
                            if (!(i > last_il && j < last_jl)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                idx_t wait_left_j = (j==dim_1-1) ? j : (j+1);
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) < wait_left_j) {  }
                            }
                            last_il = i;
                            last_jl = j;
                        }
                        // 中间的不需等待
                        
                        res -= L_ptr[0] * x[(i-1)*dim_1 + j-1];
                        res -= L_ptr[1] * x[(i-1)*dim_1 + j  ];
                        res -= L_ptr[2] * x[(i-1)*dim_1 + j+1];
                        res -= L_ptr[3] * x[ i   *dim_1 + j-1];

                        x[i * dim_1 + j] = res;
                        if (it == t_beg || it == t_end - 1)// 只要一头一尾的原子操作括起来，中间的写回可以不按序
                            __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                        else 
                            flag[i+1] = j;
                    }
                }// level loop
            }
            else {
                assert(sizeof(data_t) == 2);
                for (idx_t ilev = 0; ilev < nlevs; ilev++) {
                    idx_t ibeg = MIN(ilev >> 1, dim_0 - 1);
                    idx_t jbeg = ilev - 2 * ibeg;
                    idx_t ntasks = MIN(ibeg + 1, ((dim_1-1 - jbeg) >> 1) + 1);
                    // 确定自己分到的task范围
                    idx_t my_cnt = ntasks / nt;
                    idx_t t_beg = tid * my_cnt;
                    idx_t remain = ntasks - my_cnt * nt;
                    t_beg += MIN(remain, tid);
                    if (tid < remain) my_cnt ++;
                    idx_t t_end = t_beg + my_cnt;

                    for (idx_t it = t_end - 1; it >= t_beg; it--) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        oper_t res = b[i * dim_1 + j];

                        const data_t * L_ptr = l + lnz * (i * dim_1 + j);
                        float16x4_t L_16 = vld1_f16((__fp16*) L_ptr);
                        float32x4_t L_32 = vcvt_f32_f16(L_16);
                        oper_t * x_ptr = x + i * dim_1 + j;
                        {// 预取
                        __builtin_prefetch(L_ptr + lnz * dim_1, 0, 0);
                        __builtin_prefetch(b + (i+1)*dim_1 + j, 0, 0);
                        }

                        if (it == t_beg) {// 右边界只需检查S依赖
                            if (!(i <= last_ir && j > last_jr)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) < j-1) {  }
                            }
                            last_ir = i;
                            last_jr = j;
                        }
                        if (it == t_end - 1) {// 左边界只需检查WN依赖
                            if (!(i > last_il && j < last_jl)) {// 如果上一次本线程做的范围没有覆盖此依赖
                                idx_t wait_left_j = (j==dim_1-1) ? j : (j+1);
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) < wait_left_j) {  }
                            }
                            last_il = i;
                            last_jl = j;
                        }
                        // 中间的不需等待

                        float32x4_t x_32 = vld1q_f32(x_ptr - dim_1 - 1);
                        x_32 = vmulq_f32(L_32, x_32);
                        res -= (vgetq_lane_f32(x_32, 0) + vgetq_lane_f32(x_32, 1) + vgetq_lane_f32(x_32, 2));
                        res -=  vgetq_lane_f32(L_32, 3) * x_ptr[-1];

                        *x_ptr = res;
                        if (it == t_beg || it == t_end - 1)// 只要一头一尾的原子操作括起来，中间的写回可以不按序
                            __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                        else 
                            flag[i+1] = j;
                    }
                }// level loop
            }
        } 
        else {
            // if (my_pid == 0) printf(" t%d/%d got no work\n", glb_tid, omp_get_num_threads());
            assert(tid == -1);
        }
    }// omp para

    delete flag_3D;
}

template<typename idx_t, typename data_t, typename oper_t>
void group_sptrsv_2d5_levelbased_backward_hardCode(const data_t * U_3D, const oper_t * B_3D, oper_t * X_3D, const idx_t num_slices, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos,
    const idx_t group_size, const int * group_tid_arr, const idx_t * slc_id_arr)
{
    assert(rnz == 5 && memcmp(check_2d5_backward, pos, sizeof(idx_t) * rnz * 2)==0);
    idx_t * flag_3D = new idx_t [(dim_0 + 1) * num_slices];
    for (idx_t s = 0; s < num_slices; s++)
        flag_3D[s * (dim_0 + 1) + dim_0] = 0;// 边界柱标记已完成

    // 并行区：一次性开够所有线程
    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for collapse(2) schedule(static)
        for (idx_t s = 0; s < num_slices; s++)// 标记每一面
        for (idx_t i = dim_0; i >= 1; i--) 
            flag_3D[s * (dim_0 + 1) + i - 1] = dim_1;// 的第i柱完成到哪个高度，初始化

        int glb_tid = omp_get_thread_num();// 全局线程号
        int nt = group_size;
        int tid = group_tid_arr[glb_tid];
        idx_t slc_id = slc_id_arr[glb_tid];

        if (slc_id != -1) {
            // 根据自己负责的面定位到2D数据
            idx_t * flag = flag_3D + slc_id * (dim_0 + 1);
            const data_t * u = U_3D + slc_id * dim_0 * dim_1 * rnz;
            const oper_t * b = B_3D + slc_id * dim_0 * dim_1;
                  oper_t * x = X_3D + slc_id * dim_0 * dim_1;

            // 各自开始计算
            idx_t last_il = dim_0, last_jl = -1;
            idx_t last_ir = -1   , last_jr = dim_1;
            idx_t nlevs = dim_1 + (dim_0 - 1) * 2;// 根据2d5的特征确定一共有多少个level
                
            if (sizeof(data_t) == sizeof(oper_t)) {
            // if (true) {
                for (idx_t ilev = nlevs - 1; ilev >= 0; ilev--) {
                    idx_t ibeg = MIN(ilev >> 1, dim_0 - 1);
                    idx_t jbeg = ilev - 2 * ibeg;
                    idx_t ntasks = MIN(ibeg + 1, ((dim_1-1 - jbeg) >> 1) + 1);
                    // 确定自己分到的task范围
                    idx_t my_cnt = ntasks / nt;
                    idx_t t_beg = tid * my_cnt;
                    idx_t remain = ntasks - my_cnt * nt;
                    t_beg += MIN(remain, tid);
                    if (tid < remain) my_cnt ++;
                    idx_t t_end = t_beg + my_cnt;

                    for (idx_t it = t_beg; it < t_end; it++) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        oper_t res = b[i * dim_1 + j];
                        const data_t * U_ptr = u + rnz * (i * dim_1 + j);

                        if (it == t_beg) {// 右边界只需检查ES依赖
                            if (!(i < last_ir && j > last_jr)) {
                                idx_t wait_right_j = (j==0) ? j : (j-1);
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) > wait_right_j) {  }
                                // while (flag[i+1] > wait_right_j) {smp_rmb();} smp_rmb();
                            }
                            last_ir = i;
                            last_jr = j;
                        }
                        if (it == t_end - 1) {// 左边界只需检查N依赖
                            if (!(i >= last_il && j < last_jl)) {
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) > j+1) {  }
                                // while (flag[i] > j+1) {smp_rmb();} smp_rmb();
                            }
                            last_il = i;
                            last_jl = j;
                        }
                        // 中间的不需等待
                        oper_t para = U_ptr[0];
                        res -= U_ptr[1] * x[ i   *dim_1 + j+1];
                        res -= U_ptr[2] * x[(i+1)*dim_1 + j-1];
                        res -= U_ptr[3] * x[(i+1)*dim_1 + j  ];
                        res -= U_ptr[4] * x[(i+1)*dim_1 + j+1];
                        
                        x[i * dim_1 + j] = res / para;

                        if (it == t_beg || it == t_end - 1)
                            __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                        else
                            flag[i] = j;
                    }
                }
            }
            else {
                assert(sizeof(data_t) == 2);
                for (idx_t ilev = nlevs - 1; ilev >= 0; ilev--) {
                    idx_t ibeg = MIN(ilev >> 1, dim_0 - 1);
                    idx_t jbeg = ilev - 2 * ibeg;
                    idx_t ntasks = MIN(ibeg + 1, ((dim_1-1 - jbeg) >> 1) + 1);
                    // 确定自己分到的task范围
                    idx_t my_cnt = ntasks / nt;
                    idx_t t_beg = tid * my_cnt;
                    idx_t remain = ntasks - my_cnt * nt;
                    t_beg += MIN(remain, tid);
                    if (tid < remain) my_cnt ++;
                    idx_t t_end = t_beg + my_cnt;

                    for (idx_t it = t_beg; it < t_end; it++) {
                        idx_t i = ibeg - it;
                        idx_t j = jbeg + (it << 1);
                        oper_t res = b[i * dim_1 + j];
                        const data_t * U_ptr = u + rnz * (i * dim_1 + j);
                        
                        oper_t para = U_ptr[0];
                        float16x4_t U_16 = vld1_f16((__fp16*) U_ptr + 1);// 跨过U0   
                        float32x4_t U_32 = vcvt_f32_f16(U_16);                 
                        oper_t * x_ptr = x + i * dim_1 + j;
                        {// 预取
                        __builtin_prefetch(U_ptr - rnz * dim_1, 0, 0);
                        __builtin_prefetch(b + (i-1)*dim_1 + j, 0, 0);
                        }

                        if (it == t_beg) {// 右边界只需检查ES依赖
                            if (!(i < last_ir && j > last_jr)) {
                                idx_t wait_right_j = (j==0) ? j : (j-1);
                                while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) > wait_right_j) {  }
                                // while (flag[i+1] > wait_right_j) {smp_rmb();} smp_rmb();
                            }
                            last_ir = i;
                            last_jr = j;
                        }
                        if (it == t_end - 1) {// 左边界只需检查N依赖
                            if (!(i >= last_il && j < last_jl)) {
                                while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) > j+1) {  }
                                // while (flag[i] > j+1) {smp_rmb();} smp_rmb();
                            }
                            last_il = i;
                            last_jl = j;
                        }
                        // 中间的不需等待
    
                        float32x4_t x_32 = vld1q_f32(x_ptr + dim_1 - 2);// 使x[(i+1)*dim_1 + j-1]放在向量寄存器的第二道，与U_32匹配
                        x_32 = vmulq_f32(U_32, x_32);
                        res -= (vgetq_lane_f32(x_32, 1) + vgetq_lane_f32(x_32, 2) + vgetq_lane_f32(x_32, 3));
                        res -=  vgetq_lane_f32(U_32, 0) * x_ptr[1];
                        *x_ptr = res / para;

                        if (it == t_beg || it == t_end - 1)
                            __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                        else
                            flag[i] = j;
                    }
                }
            }
        } else {
            assert(tid == -1);
        }
    }// omp para
}

#endif

template<typename idx_t, typename data_t, typename oper_t>
void struct_sptrsv_3d_forward_frame_hardCode(const data_t * l, const oper_t * b, oper_t * x,
    const idx_t dim_0, const idx_t dim_1, const idx_t dim_2, const idx_t lnz, const idx_t * pos)
{
    // 执行一柱计算的函数
    void (*kernel)(const idx_t, const idx_t, const data_t*, const oper_t*, oper_t*) = nullptr;

    if        (lnz == 3) {assert(memcmp(pos, stencil_offset_3d7 ,  9*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_forward_zero_3d7;
    } else if (lnz == 7) {assert(memcmp(pos, stencil_offset_3d15, 21*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_forward_zero_3d15;
    } else if (lnz == 9) {assert(memcmp(pos, stencil_offset_3d19, 27*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_forward_zero_3d19;
    } else if (lnz ==13) {assert(memcmp(pos, stencil_offset_3d27, 39*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_forward_zero_3d27;
    } else assert(false);

#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_proc; MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    double t, mint, maxt;
    MPI_Barrier(MPI_COMM_WORLD);
    t = wall_time();
#endif

    const int num_threads = omp_get_max_threads();
    if (num_threads > 1) {
        // level是等值线 j + slope*i = Const, 对于3d7和3d15 斜率为1, 对于3d19和3d27 斜率为2
        const idx_t slope = (lnz == 3 || lnz == 7) ? 1 : 2;
        idx_t flag[dim_0 + 1];
        flag[0] = dim_1 - 1;// 边界标记已完成
        for (idx_t i = 0; i < dim_0; i++) 
            flag[i + 1] = -1;// 初始化为-1
        const idx_t wait_offj = slope - 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = 0; lid < nlevs; lid++) {
                // 每层的起始点位于左上角
                idx_t ibeg = MIN(lid / slope, dim_0 - 1);
                idx_t jbeg = lid - slope * ibeg;
                idx_t ntasks = MIN(ibeg + 1, (dim_1-1 - jbeg) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_end - 1; it >= t_beg; it--) {
                    idx_t i = ibeg - it;
                    idx_t j = jbeg + it * slope;
                    idx_t wait_left_j = (j == dim_1 - 1) ? j : (j + wait_offj);
                    const idx_t vec_off = (i * dim_1 + j) * dim_2;
                    const oper_t * b_jik = b + vec_off;
                    oper_t * x_jik = x + vec_off;
                    const data_t * L_jik = l + vec_off * lnz;
                    // 线程边界处等待
                    if (it == t_beg) while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) < j-1) {  } // 只需检查W依赖
                    if (it == t_end - 1) while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) < wait_left_j) {  }
                    
                    kernel(dim_2, dim_1, L_jik, b_jik, x_jik);
                    
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                    else flag[i+1] = j;;
                }
            }
        }
    }
    else {
        for (idx_t i = 0; i < dim_0; i++)
        for (idx_t j = 0; j < dim_1; j++) {
            const idx_t vec_off = (i * dim_1 + j) * dim_2;
            const oper_t * b_jik = b + vec_off;
            oper_t * x_jik = x + vec_off;
            const data_t * L_jik = l + vec_off * lnz;
            kernel(dim_2, dim_1, L_jik, b_jik, x_jik);
        }
    }

#ifdef PROFILE
    t = wall_time() - t;
    MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (my_pid == 0) {
        double bytes = dim_0 * dim_1 * dim_2 * (lnz * sizeof(data_t) + 2 * sizeof(oper_t));
        bytes = bytes * num_proc / (1024*1024*1024);
        printf("Forw BILU data_t %ld oper_t %ld total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
            sizeof(data_t), sizeof(oper_t), bytes, mint, maxt, bytes/maxt, bytes/mint);
    }
#endif
}

template<typename idx_t, typename data_t, typename oper_t>
void struct_sptrsv_3d_backward_frame_hardCode(const data_t * u, const oper_t * b, oper_t * x,
    const idx_t dim_0, const idx_t dim_1, const idx_t dim_2, const idx_t rnz, const idx_t * pos)
{
    // 执行一柱计算的函数
    void (*kernel)(const idx_t, const idx_t, const data_t*, const oper_t*, oper_t*) = nullptr;

    if        (rnz == 4) {assert(memcmp(pos, stencil_offset_3d7  +  9, 12*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_backward_zero_3d7;
    } else if (rnz == 8) {assert(memcmp(pos, stencil_offset_3d15 + 21, 24*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_backward_zero_3d15;
    } else if (rnz ==10) {assert(memcmp(pos, stencil_offset_3d19 + 27, 30*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_backward_zero_3d19;
    } else if (rnz ==14) {assert(memcmp(pos, stencil_offset_3d27 + 39, 42*sizeof(idx_t)) == 0);
        kernel = AOS_ilu_backward_zero_3d27;
    } else assert(false);

#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_proc; MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    double t, mint, maxt;
    MPI_Barrier(MPI_COMM_WORLD);
    t = wall_time();
#endif

    const int num_threads = omp_get_max_threads();
    if (num_threads > 1) {
        // level是等值线 j + slope*i = Const, 对于3d7和3d15 斜率为1, 对于3d19和3d27 斜率为2
        const idx_t slope = (rnz == 4 || rnz == 8) ? 1 : 2;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界柱标记已完成
        for (idx_t i = dim_0; i >= 1; i--)
            flag[i - 1] = dim_1;
        const idx_t wait_offj = - (slope - 1);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = nlevs - 1; lid >= 0; lid--) {
                // 每层的起始点位于左上角
                idx_t ibeg = MIN(lid / slope, dim_0 - 1);
                idx_t jbeg = lid - slope * ibeg;
                idx_t ntasks = MIN(ibeg + 1, (dim_1-1 - jbeg) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_beg; it < t_end; it++) {
                    idx_t i = ibeg - it;
                    idx_t j = jbeg + it * slope;
                    idx_t wait_right_j = (j == 0) ? j : (j + wait_offj);
                    const idx_t vec_off = (i * dim_1 + j) * dim_2 + dim_2 - 1;
                    const oper_t * b_jik = b + vec_off;
                    oper_t * x_jik = x + vec_off;
                    const data_t * U_jik = u + vec_off * rnz;
                    // 线程边界处等待
                    if (it == t_beg) while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) > wait_right_j) {  }
                    if (it == t_end - 1) while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) > j+1) {  }

                    kernel(dim_2, dim_1, U_jik, b_jik, x_jik);

                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                    else flag[i] = j;
                }
            }
        }
    }
    else {
        for (idx_t i = dim_0 - 1; i >= 0; i--)
        for (idx_t j = dim_1 - 1; j >= 0; j--) {
            const idx_t vec_off = (i * dim_1 + j) * dim_2 + dim_2 - 1;
            const oper_t * b_jik = b + vec_off;
            oper_t * x_jik = x + vec_off;
            const data_t * U_jik = u + vec_off * rnz;
            kernel(dim_2, dim_1, U_jik, b_jik, x_jik);
        }
    }

#ifdef PROFILE
        t = wall_time() - t;
        MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (my_pid == 0) {
            double bytes = dim_0 * dim_1 * dim_2 * (rnz * sizeof(data_t) + 2 * sizeof(oper_t));
            bytes = bytes * num_proc / (1024*1024*1024);
            printf("Back ILU data_t %ld oper_t %ld total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                sizeof(data_t), sizeof(oper_t), bytes, mint, maxt, bytes/maxt, bytes/mint);
        }
#endif
}

#endif