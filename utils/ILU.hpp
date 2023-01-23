#ifndef SMG_UTILS_ILU_HPP
#define SMG_UTILS_ILU_HPP

#include "common.hpp"

#define STENCIL_OFF_RANGE 1
#define STENCIL_RANGE (2 * STENCIL_OFF_RANGE + 1)


template<typename idx_t, typename data_t>
void struct_ILU_2d(const data_t * a, data_t * l, data_t * u, 
    const idx_t dim_0, const idx_t dim_1, const idx_t nnz, const idx_t * pos)
{
    data_t (*A)[nnz] = (data_t (*)[nnz])a;

    idx_t v[STENCIL_RANGE][STENCIL_RANGE];
    // 相当于一个flag，从原点开始，每个维度[ - STENCIL_OFF_RANGE : STENCIL_OFF_RANGE]范围，
	// 一共STENCIL_RANGE x STENCIL_RANGE 个点，某个点有非零元就置位
    for (idx_t i = 0; i < STENCIL_RANGE; i++)
    for (idx_t j = 0; j < STENCIL_RANGE; j++)
        v[i][j] = -1;
    
    for (idx_t i = 0; i < nnz; i++) {
        idx_t oi = pos[2 * i    ] + STENCIL_OFF_RANGE;
        idx_t oj = pos[2 * i + 1] + STENCIL_OFF_RANGE;
        v[oi][oj] = i;
    }

    idx_t lnz = 0;
    for ( ; lnz < nnz; lnz++) {
        idx_t oi = pos[2 * lnz    ];
        idx_t oj = pos[2 * lnz + 1];
        if (oi * 10 + oj >= 0) break;// 跳出时lnz已经指向对角元
    }

    idx_t rnz = nnz - lnz;// rnz包含对角元

    data_t rbuf[nnz];
    for (idx_t i = 0; i < dim_0; i++)
    for (idx_t j = 0; j < dim_1; j++) {// 从前到后处理结构点，记为P
        idx_t pt_idx_1D = i * dim_1 + j;
        for (idx_t ii = 0; ii < nnz; ii++)// 拷贝该结构点的非零元素值到缓冲区
            rbuf[ii] = A[pt_idx_1D][ii];

        for (idx_t ii = 0; ii < lnz; ii++) {// 逐个遍历该结构点（P）的左非零元，试图将其消元
            idx_t oi = pos[2 * ii    ];// 该左非零元在二维方向上的stencil偏移（offset）
            idx_t oj = pos[2 * ii + 1];
            idx_t ti = oi + i;// 这一行（P点）内要处理的这个左非零元，对应的列（该结构点记为Q）的坐标
            idx_t tj = oj + j;
            if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1) {// 该左非零元对应的列（结构点Q）落在本进程的处理范围内
                idx_t u_pos = (ti * dim_1 + tj) * rnz;
                rbuf[ii] /= u[u_pos];
                for (idx_t jj = 1; jj < rnz; jj++) {
                    // 消元的视角：从Q的那一行来看，遍历其对角元以右的非零元
                    idx_t temp = lnz + jj;
                    // 见上面，oi和oj是P到Q的偏移量，再加一遍pos[...]，后者代表Q到其右非零元对应的结构点（记为R）的偏移量
					// 从而得到P到R的偏移量，然后检查是否与P相邻（即P行内也有该非零元）
                    idx_t tii = oi + pos[2 * temp    ] + STENCIL_OFF_RANGE;
                    idx_t tjj = oj + pos[2 * temp + 1] + STENCIL_OFF_RANGE;
                    // 当本行相同位置也有非零元时才进行更新（目的：不产生额外填充）
                    if (tii >= 0 && tii < STENCIL_RANGE &&
                        tjj >= 0 && tjj < STENCIL_RANGE && v[tii][tjj] != -1) {
                        rbuf[v[tii][tjj]] -= rbuf[ii] * u[u_pos + jj];
                    }
                }// jj
            }
            else {// 对于不在本进程范围内的列直接赋0？？？并不需要，因为forward和backward中会把这些点直接跳过
                rbuf[ii] = 0.0;
            }
        }// ii
        for (idx_t ii = lnz; ii < nnz; ii++) {// 对不在本进程负责范围内的右非零元赋0：为了方便在前代回代时不写if分支
            idx_t oi = pos[2 * ii    ];// 该左非零元在二维方向上的stencil偏移（offset）
            idx_t oj = pos[2 * ii + 1];
            idx_t ti = oi + i;// 这一行（P点）内要处理的这个左非零元，对应的列（该结构点记为Q）的坐标
            idx_t tj = oj + j;
            if (!(ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1)) {
                rbuf[ii] = 0.0;
            }
        }
        // 将该结构点分解后的值（对应于非结构CSR矩阵中的一行）拷贝到L和U中
        for (idx_t ii = 0; ii < lnz; ii++)
            l[pt_idx_1D * lnz + ii] = rbuf[ii];
        for (idx_t jj = 0; jj < rnz; jj++)
            u[pt_idx_1D * rnz + jj] = rbuf[lnz + jj];
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void struct_trsv_forward_2d(const data_t * l, const oper_t * b, oper_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos)
{
    for (idx_t i = 0; i < dim_0; i++) {
        const oper_t * b_col = b + i * dim_1;
                oper_t * x_col = x + i * dim_1;
        const data_t * L_col = l + lnz * (i * dim_1);      
        for (idx_t j = 0; j < dim_1; j++) {
            oper_t res = * b_col;
            for (idx_t iter = 0; iter < lnz; iter++) {// 所有的左非零元
                idx_t oi = pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                idx_t oj = pos[2 * iter + 1];
                // 这个邻居结构点不在本进程负责范围内，就不管
                idx_t ti = i + oi, tj = j + oj;
                if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1)
                    res -= L_col[iter] * x_col[oi * dim_1 + oj];
            }
            *x_col = res;

            b_col ++;
            x_col ++;
            L_col += lnz;
        }// j loop
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void struct_trsv_backward_2d(const data_t * u, const oper_t * b, oper_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos)
{
    for (idx_t i = dim_0 - 1; i >= 0; i--) {
        const oper_t * b_col = b + i * dim_1 + dim_1 - 1;
                oper_t * x_col = x + i * dim_1 + dim_1 - 1;
        const data_t * U_col = u + rnz *(i * dim_1 + dim_1 - 1);
        for (idx_t j = dim_1 - 1; j >= 0; j--) {// 注意这个遍历顺序是从后到前
            oper_t para = 1.0;
            oper_t res = * b_col;
            for (idx_t iter = 0; iter < rnz; iter++) {// 所有的右非零元（包含对角元）
                idx_t oi = pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                idx_t oj = pos[2 * iter + 1];
                idx_t ti = i + oi, tj = j + oj;
                if (ti < 0 || ti >= dim_0 || tj < 0 || tj >= dim_1)
                    continue;

                oper_t val = U_col[iter];
                if (ti == i && tj == j)
                    para = val;
                else
                    res -= val * x_col[oi * dim_1 + oj];
            }
            *x_col = res / para;

            b_col --;
            x_col --;
            U_col -= rnz;
        }// j loop
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void group_sptrsv_2d5_levelbased_forward(const data_t * L_3D, const oper_t * B_3D, oper_t * X_3D, const idx_t num_slices,
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos, 
    const idx_t group_size, const int * group_tid_arr, const idx_t * slc_id_arr)
{
    idx_t * flag_3D = new idx_t [(dim_0 + 1) * num_slices];
    for (idx_t s = 0; s < num_slices; s++)
        flag_3D[s * (dim_0 + 1)] = dim_1 - 1;// 每一个面的边界柱标记已完成
    
    // int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
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
        //  if (my_pid == 0) printf(" t%d/%d got slc_id %d\n", glb_tid, omp_get_num_threads(), slc_id);
        
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
                    for (idx_t iter = 0; iter < lnz; iter++) {// 所有的左非零元
                        idx_t ti = i + pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                        idx_t tj = j + pos[2 * iter + 1];
                        if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1)
                            res -= L_ptr[iter] * x[ti * dim_1 + tj];
                    }
                    x[i * dim_1 + j] = res;
                    if (it == t_beg || it == t_end - 1)// 只要一头一尾的原子操作括起来，中间的写回可以不按序
                        __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                    else 
                        flag[i+1] = j;
                }
            }// level loop
        } 
        else {
            // if (my_pid == 0) printf(" t%d/%d got no work\n", glb_tid, omp_get_num_threads());
            assert(tid == -1);
        }
    }// omp para

    delete flag_3D;
}

template<typename idx_t, typename data_t, typename oper_t>
void group_sptrsv_2d5_levelbased_backward(const data_t * U_3D, const oper_t * B_3D, oper_t * X_3D, const idx_t num_slices, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos,
    const idx_t group_size, const int * group_tid_arr, const idx_t * slc_id_arr)
{
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
                    oper_t para = 1.0;
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
                    for (idx_t iter = 0; iter < rnz; iter++) {// 所有的左非零元
                        idx_t ti = i + pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                        idx_t tj = j + pos[2 * iter + 1];
                        if (ti < 0 || ti >= dim_0 || tj < 0 || tj >= dim_1)
                            continue;
                        oper_t val = U_ptr[iter];
                        if (ti == i && tj == j)
                            para = val;
                        else
                            res -= val * x[ti * dim_1 + tj];
                    }
                    x[i * dim_1 + j] = res / para;
                    if (it == t_beg || it == t_end - 1)
                        __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                    else
                        flag[i] = j;
                }
            }
        } else {
            assert(tid == -1);
        }
    }// omp para
}

#if 0
/*  以下两个函数
    struct_trsv_forward_2d5_levelbased
    struct_trsv_backward_2d5_levelbased
    是多做一的，以面为单位，难以做openmp的多线程嵌套并行
 */
template<typename idx_t, typename data_t>
void struct_trsv_forward_2d5_levelbased(const data_t * l, const data_t * b, data_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos, int nt, idx_t* flag)
{
    flag[0] = dim_1-1;// 边界柱标记已完成
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (idx_t i = 0; i < dim_0; i++) 
        flag[i + 1] = -1;// 标记第i柱完成到哪个高度，初始化为-1

    #pragma omp parallel num_threads(nt) proc_bind(spread)
    {
        int tid = omp_get_thread_num();
        int tot = omp_get_num_threads(); assert(tot == nt);

        // if (my_pid == 0 && out_tid == 0) {
        //     for (int t = 0; t < omp_get_num_threads(); t++) {
        //         if (t == tid) {
        //             printf(" it%d/ot%d ", tid, out_tid);
        //             print_affinity();
        //         }
        //         #pragma omp barrier
        //     }
        // }

        idx_t last_il = dim_0, last_jl = -1;
        idx_t last_ir = -1   , last_jr = dim_1;
        idx_t nlevs = dim_1 + (dim_0 - 1) * 2;// 根据2d5的特征确定一共有多少个level

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
                data_t res = b[i * dim_1 + j];
                const data_t * L_ptr = l + lnz * (i * dim_1 + j);

                if (it == t_beg) {// 右边界只需检查S依赖
                    if (!(i <= last_ir && j > last_jr)) {// 如果上一次本线程做的范围没有覆盖此依赖
                        while (__atomic_load_n(&flag[i+1], __ATOMIC_ACQUIRE) < j-1) {  }
                        // while (flag[i+1] < j - 1) {smp_rmb();} smp_rmb();
                    }
                    last_ir = i;
                    last_jr = j;
                }
                if (it == t_end - 1) {// 左边界只需检查WN依赖
                    if (!(i > last_il && j < last_jl)) {// 如果上一次本线程做的范围没有覆盖此依赖
                        idx_t wait_left_j = (j==dim_1-1) ? j : (j+1);
                        while (__atomic_load_n(&flag[i  ], __ATOMIC_ACQUIRE) < wait_left_j) {  }
                        // while (flag[i] < wait_left_j) { smp_rmb(); } smp_rmb();
                    }
                    last_il = i;
                    last_jl = j;
                }
                // 中间的不需等待
                for (idx_t iter = 0; iter < lnz; iter++) {// 所有的左非零元
                    idx_t ti = i + pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                    idx_t tj = j + pos[2 * iter + 1];
                    if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1)
                        res -= L_ptr[iter] * x[ti * dim_1 + tj];
                }
                x[i * dim_1 + j] = res;
                if (it == t_beg || it == t_end - 1)// 只要一头一尾的原子操作括起来，中间的写回可以不按序
                    __atomic_store_n(&flag[i+1], j, __ATOMIC_RELEASE);
                else 
                    flag[i+1] = j;
            }
        }// level loop
    }// parallel region
}

template<typename idx_t, typename data_t>
void struct_trsv_backward_2d5_levelbased(const data_t * u, const data_t * b, data_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos, int nt, idx_t* flag)
{
    flag[dim_0] = 0;// 边界柱标记已完成
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (idx_t i = dim_0; i >= 1; i--) 
        flag[i - 1] = dim_1;// 标记第i柱完成到哪个高度，初始化

    #pragma omp parallel num_threads(nt) proc_bind(spread)
    {
        int tid = omp_get_thread_num();
        int tot = omp_get_num_threads(); assert(tot == nt);

        idx_t last_il = dim_0, last_jl = -1;
        idx_t last_ir = -1   , last_jr = dim_1;
        idx_t nlevs = dim_1 + (dim_0 - 1) * 2;// 根据2d5的特征确定一共有多少个level
            
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
                data_t para = 1.0;
                data_t res = b[i * dim_1 + j];
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
                for (idx_t iter = 0; iter < rnz; iter++) {// 所有的左非零元
                    idx_t ti = i + pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                    idx_t tj = j + pos[2 * iter + 1];
                    if (ti < 0 || ti >= dim_0 || tj < 0 || tj >= dim_1)
                        continue;
                    data_t val = U_ptr[iter];
                    if (ti == i && tj == j)
                        para = val;
                    else
                        res -= val * x[ti * dim_1 + tj];
                }
                x[i * dim_1 + j] = res / para;
                if (it == t_beg || it == t_end - 1)
                    __atomic_store_n(&flag[i], j, __ATOMIC_RELEASE);
                else
                    flag[i] = j;
            }
        }
    }
}

template<typename idx_t, typename data_t>
void struct_trsv_forward_2d5_pipelined(const data_t * l, const data_t * b, data_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t lnz, const idx_t * pos, int nt, const idx_t gran, idx_t* flag)
{
    // Pipeline based：即使不同步，带宽利用也很差；加了同步之后，带宽利用基本不变
    flag[0] = dim_1;// 边界柱标记已完成
    #pragma omp parallel num_threads(nt) proc_bind(spread)
    {
        #pragma omp for schedule(static)
        for (idx_t i = 0; i < dim_0; i++) flag[i + 1] = -1;// 标记第i柱完成到哪个高度，初始化为-1

        data_t L_buf[gran * lnz], b_buf[gran];
        #pragma omp for schedule(static, 1)
        for (idx_t i = 0; i < dim_0; i++) {

            for (idx_t gj = 0; gj < dim_1; gj += gran) {// group j
                const idx_t jbeg = gj, jend = (jbeg + gran > dim_1) ? dim_1 : (jbeg + gran);
                // copy mat vals into buf, to overlap with waiting
                for (idx_t j = jbeg; j < jend; j++) {
                    for (idx_t k = 0; k < lnz; k++)
                        L_buf[(j - jbeg) * lnz + k] = l[lnz * (i * dim_1 + j) + k];
                    b_buf[j - jbeg] = b[i * dim_1 + j];
                }

                while (flag[i] < jend) { smp_rmb(); }// 等待依赖就绪，注意jend的一端是开区间
                smp_rmb();// 读屏障：保证该屏障前后的读取指令的重排不会跨过此屏障
                for (idx_t j = jbeg; j < jend; j++) {// local j
                    data_t res = b_buf[j - jbeg];
                    for (idx_t iter = 0; iter < lnz; iter++) {
                        idx_t ti = i + pos[2 * iter    ];
                        idx_t tj = j + pos[2 * iter + 1];
                        if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1)
                            res -= L_buf[lnz * (j - jbeg) + iter] * x[ti * dim_1 + tj];
                    }
                    x[i * dim_1 + j] = res;
                }
                smp_wmb();// 写屏障：保证该屏障前后的写入指令的重排不会跨过此屏障。硬件上保证store buffer数据全部清空才往后执行
                flag[i + 1] = (jend >= dim_1) ? dim_1 : (jend - 1);// jend指向的就是下一次的jbeg
            }
        }// i loop
    }// parallel region
}

template<typename idx_t, typename data_t>
void struct_trsv_backward_2d5_pipelined(const data_t * u, const data_t * b, data_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t rnz, const idx_t * pos, int nt, const idx_t gran, idx_t* flag)
{
    // Pipeline based：即使不同步，带宽利用也很差；加了同步之后，带宽利用基本不变
    flag[dim_0] = -1;// 边界柱标记已完成
    #pragma omp parallel num_threads(nt) proc_bind(spread)
    {
        #pragma omp for schedule(static)
        for (idx_t i = dim_0; i >= 1; i--) flag[i - 1] = dim_1;// 标记第i柱完成到哪个高度，初始化

        data_t U_buf[gran * rnz], b_buf[gran];

        #pragma omp for schedule(static, 1)
        for (idx_t i = dim_0 - 1; i >= 0; i--) {

            for (idx_t gj = dim_1 - 1; gj >= 0; gj -= gran) {
                const idx_t jbeg = gj, jend = (jbeg - gran < -1) ? -1 : (jbeg - gran);
                // copy mat vals into buf, to overlap with waiting
                for (idx_t j = jbeg; j > jend; j--) {
                    for (idx_t k = 0; k < rnz; k++)
                        U_buf[(jbeg - j) * rnz + k] = u[rnz * (i * dim_1 + j) + k];
                    b_buf[jbeg - j] = b[i * dim_1 + j];
                }

                while (flag[i + 1] > jend) { smp_rmb(); }// 注意jend的一端是开区间
                smp_rmb();// 读屏障
                for (idx_t j = jbeg; j > jend; j--) {
                    data_t para = 1.0;
                    data_t res = b_buf[jbeg - j];
                    for (idx_t iter = 0; iter < rnz; iter++) {// 所有的右非零元（包含对角元）
                        idx_t ti = i + pos[2 * iter    ];// 该非零元对应的邻居结构点坐标
                        idx_t tj = j + pos[2 * iter + 1];
                        if (ti < 0 || ti >= dim_0 || tj < 0 || tj >= dim_1)
                            continue;

                        data_t val = U_buf[rnz * (jbeg - j) + iter];
                        if (ti == i && tj == j)
                            para = val;
                        else
                            res -= val * x[ti * dim_1 + tj];
                    }
                    x[i * dim_1 + j] = res / para;
                }
                smp_wmb();// 写屏障
                flag[i] = (jend <= -1) ? -1 : (jend + 1);
            }
        }// i loop
    }// parallel region
}

#endif

template<typename idx_t, typename data_t>
void struct_ILU_3d(const data_t * a, data_t * l, data_t * u, 
    const idx_t dim_0, const idx_t dim_1, const idx_t dim_2, const idx_t nnz, const idx_t * pos)
{
    data_t (*A)[nnz] = (data_t (*)[nnz])a;

    idx_t v[STENCIL_RANGE][STENCIL_RANGE][STENCIL_RANGE];
    // 相当于一个flag，从原点开始，每个维度[ - STENCIL_OFF_RANGE : STENCIL_OFF_RANGE]范围，
	// 一共STENCIL_RANGE x STENCIL_RANGE 个点，某个点有非零元就置位
    for (idx_t i = 0; i < STENCIL_RANGE; i++)
    for (idx_t j = 0; j < STENCIL_RANGE; j++)
    for (idx_t k = 0; k < STENCIL_RANGE; k++)
        v[i][j][k] = -1;
    
    for (idx_t i = 0; i < nnz; i++) {
        idx_t oi = pos[3 * i    ] + STENCIL_OFF_RANGE;
        idx_t oj = pos[3 * i + 1] + STENCIL_OFF_RANGE;
        idx_t ok = pos[3 * i + 2] + STENCIL_OFF_RANGE;
        v[oi][oj][ok] = i;
    }

    idx_t lnz = 0;
    for ( ; lnz < nnz; lnz++) {
        idx_t oi = pos[3 * lnz    ];
        idx_t oj = pos[3 * lnz + 1];
        idx_t ok = pos[3 * lnz + 2];
        if (oi * 100 + oj * 10 + ok >= 0) break;// 跳出时lnz已经指向对角元
    }

    idx_t rnz = nnz - lnz;// rnz包含对角元

    data_t rbuf[nnz];
    for (idx_t i = 0; i < dim_0; i++)
    for (idx_t j = 0; j < dim_1; j++)
    for (idx_t k = 0; k < dim_2; k++) {// 从前到后处理结构点，记为P
        idx_t pt_idx_1D = (i * dim_1 + j) * dim_2 + k;
        for (idx_t ii = 0; ii < nnz; ii++)// 拷贝该结构点的非零元素值到缓冲区
            rbuf[ii] = A[pt_idx_1D][ii];

        for (idx_t ii = 0; ii < lnz; ii++) {// 逐个遍历该结构点（P）的左非零元，试图将其消元
            idx_t oi = pos[3 * ii    ];// 该左非零元在二维方向上的stencil偏移（offset）
            idx_t oj = pos[3 * ii + 1];
            idx_t ok = pos[3 * ii + 2];
            idx_t ti = oi + i;// 这一行（P点）内要处理的这个左非零元，对应的列（该结构点记为Q）的坐标
            idx_t tj = oj + j;
            idx_t tk = ok + k;
            if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1 && tk >= 0 && tk < dim_2) {// 该左非零元对应的列（结构点Q）落在本进程的处理范围内
                idx_t u_pos = ((ti * dim_1 + tj) * dim_2 + tk) * rnz;
                rbuf[ii] /= u[u_pos];
                for (idx_t jj = 1; jj < rnz; jj++) {
                    // 消元的视角：从Q的那一行来看，遍历其对角元以右的非零元
                    idx_t temp = lnz + jj;
                    // 见上面，oi和oj是P到Q的偏移量，再加一遍pos[...]，后者代表Q到其右非零元对应的结构点（记为R）的偏移量
					// 从而得到P到R的偏移量，然后检查是否与P相邻（即P行内也有该非零元）
                    idx_t tii = oi + pos[3 * temp    ] + STENCIL_OFF_RANGE;
                    idx_t tjj = oj + pos[3 * temp + 1] + STENCIL_OFF_RANGE;
                    idx_t tkk = ok + pos[3 * temp + 2] + STENCIL_OFF_RANGE;
                    // 当本行相同位置也有非零元时才进行更新（目的：不产生额外填充）
                    if (tii >= 0 && tii < STENCIL_RANGE &&
                        tjj >= 0 && tjj < STENCIL_RANGE &&
                        tkk >= 0 && tkk < STENCIL_RANGE && v[tii][tjj][tkk] != -1) {
                        rbuf[v[tii][tjj][tkk]] -= rbuf[ii] * u[u_pos + jj];
                    }
                }// jj
            }
            else {// 对于不在本进程范围内的列直接赋0？？？并不需要，因为forward和backward中会把这些点直接跳过
                rbuf[ii] = 0.0;
            }
        }// ii
        for (idx_t ii = lnz; ii < nnz; ii++) {// 对不在本进程负责范围内的右非零元赋0：为了方便在前代回代时不写if分支
            idx_t oi = pos[3 * ii    ];// 该左非零元在二维方向上的stencil偏移（offset）
            idx_t oj = pos[3 * ii + 1];
            idx_t ok = pos[3 * ii + 2];
            idx_t ti = oi + i;// 这一行（P点）内要处理的这个左非零元，对应的列（该结构点记为Q）的坐标
            idx_t tj = oj + j;
            idx_t tk = ok + k;
            if (!(ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1 && tk >= 0 && tk < dim_2)) {
                rbuf[ii] = 0.0;
            }
        }
        // 将该结构点分解后的值（对应于非结构CSR矩阵中的一行）拷贝到L和U中
        for (idx_t ii = 0; ii < lnz; ii++)
            l[pt_idx_1D * lnz + ii] = rbuf[ii];
        for (idx_t jj = 0; jj < rnz; jj++)
            u[pt_idx_1D * rnz + jj] = rbuf[lnz + jj];
    }
}


template<typename idx_t, typename data_t, typename oper_t>
void struct_trsv_forward_3d(const data_t * l, const oper_t * b, oper_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t dim_2, const idx_t lnz, const idx_t * pos) 
{
    for (idx_t i = 0; i < dim_0; i++)
    for (idx_t j = 0; j < dim_1; j++)
    for (idx_t k = 0; k < dim_2; k++) {
        oper_t res = b[(i * dim_1 + j) * dim_2 + k];
        // printf("i %d j %d k %d b %.5e", i, j, k, res);
        for (idx_t iter = 0; iter < lnz; iter++) {// 所有的左非零元
            idx_t ti = i + pos[3 * iter    ];// 该非零元对应的邻居结构点坐标
            idx_t tj = j + pos[3 * iter + 1];
            idx_t tk = k + pos[3 * iter + 2];
            // 这个邻居结构点不在本进程负责范围内，就不管
            if (ti >= 0 && ti < dim_0 && tj >= 0 && tj < dim_1 && tk >= 0 && tk < dim_2) {
                res -= l[lnz * ((i * dim_1 + j) * dim_2 + k) + iter] * x[(ti * dim_1 + tj) * dim_2 + tk];
                // printf(" (%.5e , %.5e)", 
                //     l[lnz * ((i * dim_1 + j) * dim_2 + k) + iter], x[(ti * dim_1 + tj) * dim_2 + tk]);
            }
        }
        x[(i * dim_1 + j) * dim_2 + k] = res;

        // printf(" res %.5e\n", res);
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void struct_trsv_backward_3d(const data_t * u, const oper_t * b, oper_t * x, 
    const idx_t dim_0, const idx_t dim_1, const idx_t dim_2, const idx_t rnz, const idx_t * pos)
{
    for (idx_t i = dim_0 - 1; i >= 0; i--)
    for (idx_t j = dim_1 - 1; j >= 0; j--)
    for (idx_t k = dim_2 - 1; k >= 0; k--) {// 注意这个遍历顺序是从后到前
        oper_t para = 1.0;
        oper_t res = b[(i * dim_1 + j) * dim_2 + k];
        for (idx_t iter = 0; iter < rnz; iter++) {// 所有的右非零元（包含对角元）
            idx_t ti = i + pos[3 * iter    ];// 该非零元对应的邻居结构点坐标
            idx_t tj = j + pos[3 * iter + 1];
            idx_t tk = k + pos[3 * iter + 2];
            if (ti < 0 || ti >= dim_0 || tj < 0 || tj >= dim_1 || tk < 0 || tk >= dim_2)
                continue;

            oper_t val = u[rnz * ((i * dim_1 + j) * dim_2 + k) + iter];
            if (ti == i && tj == j && tk == k)
                para = val;
            else
                res -= val * x[(ti * dim_1 + tj) * dim_2 + tk];
        }
        x[(i * dim_1 + j) * dim_2 + k] = res / para;
    }
}

#undef STENCIL_RANGE
#undef STENCIL_OFF_RANGE

#endif