#ifndef SMG_PLANE_ILU_HPP
#define SMG_PLANE_ILU_HPP

#include "plane_Solver.hpp"
#include "../utils/par_struct_mat.hpp"
#include "../utils/ILU.hpp"
#include "../utils/ILU_hardcode.hpp"

typedef enum {ILU_2D9} PlaneILU_type;
typedef enum {ROUND_ROBIN, COWORK_1On1=1, COWORK_2ON1, COWORK_3ON1, COWORK_4ON1} CoWork_type;

template<typename idx_t, typename data_t, typename calc_t>
class PlaneILU final : public PlaneSolver<idx_t, data_t, calc_t> {
public:
    PlaneILU_type type;
    idx_t outer_dim = 0, midle_dim = 0, inner_dim = 0;
    idx_t jbeg = 0, jend = 0, ibeg = 0, iend = 0, kbeg = 0, kend = 0;
    idx_t lnz = 0, rnz = 0;
    // ILU分解后的上下三角结构化矩阵
    data_t * L = nullptr, * U = nullptr;
    bool shuffled = false;
    calc_t * buf = nullptr, * buf_mem = nullptr;// 用于数据shape转换的
    calc_t * tmp = nullptr, * tmp_mem = nullptr;// 中间变量
    // 记录stencil偏移
    idx_t num_stencil = 0;// 未初始化
    const idx_t * stencil_offset = nullptr;
    bool setup_called = false;

    CoWork_type cw_type = ROUND_ROBIN;
    idx_t * worker_jobID = nullptr;// 对于DYNAMIC_xOn1的动态负载类型，记录每个线程号对应哪一个PILU平面
    int * worker_tidInGroup = nullptr;// 对于DYNAMIC_xOn1的动态负载类型，记录每个线程在组内的局部tid

    PlaneILU(PlaneILU_type type=ILU_2D9): type(type) {
        if (type == ILU_2D9) {
            num_stencil = 9;
            stencil_offset = stencil_offset_2d9;
        } else {
            printf("Not supported ilu type %d! Only ILU_2d9 available!\n", type);
            MPI_Abort(MPI_COMM_WORLD, -99);
        }
    }
    void Setup();

    void truncate() {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: PILU truncated\n");
        assert(L != nullptr && U != nullptr);// 在截断之前务必已经做完分解了
#ifdef __aarch64__
        idx_t tot_len = outer_dim * midle_dim * inner_dim * lnz;
        for (idx_t i = 0; i < tot_len; i++) {
            __fp16 tmp = (__fp16) L[i];
            L[i] = (data_t) tmp;
        }
        tot_len = outer_dim * midle_dim * inner_dim * rnz;
        for (idx_t i = 0; i < tot_len; i++) {
            __fp16 tmp = (__fp16) U[i];
            U[i] = (data_t) tmp;
        }
#endif
    }

    /* input为右端向量，output为输出
       用作直接解方程时，output = U^{-1}*L^{-1}*input，此处的L和U是不完全分解的
       用作预条件时（相当于iterative refinement），先计算一遍残差：此时将output视作当前的近似解x：resi = input - A*output，
           然后再求解残差方程得到误差：e = U^{-1}*L^{-1}*resi，再将误差e累加回到近似解：output += e
     */
protected:
    void Mult(const par_structVector<idx_t, calc_t> & input, 
                    par_structVector<idx_t, calc_t> & output) const ;
    void apply() const ;
public:
    void Mult(const par_structVector<idx_t, calc_t> & input, 
                    par_structVector<idx_t, calc_t> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }
    void compress();
    ~PlaneILU();
};

template<typename idx_t, typename data_t, typename calc_t>
PlaneILU<idx_t, data_t, calc_t>::~PlaneILU() {
    if (L != nullptr) {delete L; L = nullptr;}
    if (U != nullptr) {delete U; U = nullptr;}
    if (buf_mem != nullptr) {delete buf_mem; buf_mem = nullptr;}
    if (tmp_mem != nullptr) {delete tmp_mem; tmp_mem = nullptr;}
    if (worker_jobID != nullptr) {delete worker_jobID; worker_jobID = nullptr;}
    if (worker_tidInGroup != nullptr) {delete worker_tidInGroup; worker_tidInGroup = nullptr;}
}

template<typename idx_t, typename data_t, typename calc_t>
void PlaneILU<idx_t, data_t, calc_t>::Setup() {
    if (setup_called) return ;

    assert(this->oper != nullptr);
    assert(this->plane_dir == XZ);// 下设y方向宽度
    // assert matrix has updated halo to prepare data 强制类型转换
    const par_structMatrix<idx_t, calc_t, calc_t> & par_A = *((par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper));
    const seq_structMatrix<idx_t, calc_t, calc_t> & A = *(par_A.local_matrix);

    if constexpr (sizeof(calc_t) != sizeof(data_t)) {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) {
            printf("  \033[1;31mWarning\033[0m: PILU::Setup() using calc_t of %ld bytes, but data_t of %ld bytes\n",
                sizeof(calc_t), sizeof(data_t));
        }
    }

    // 确定各维上是否是边界
    const bool x_lbdr = par_A.comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = par_A.comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    // const bool y_lbdr = par_A.comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = par_A.comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = par_A.comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = par_A.comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;
    outer_dim =                           A.local_y,// 注意逐面做ILU不需要管y方向的halo区
    midle_dim = (x_lbdr ? 0 : A.halo_x) + A.local_x + (x_ubdr ? 0 : A.halo_x),// 注意：ILU分解的区域包含halo区，Additive Schwartz
    inner_dim = (z_lbdr ? 0 : A.halo_z) + A.local_z + (z_ubdr ? 0 : A.halo_z);

    assert(num_stencil % 2 != 0);
    lnz = (num_stencil - 1) >> 1;
    rnz =  num_stencil - lnz;
    calc_t * A_trc = new calc_t [ outer_dim * midle_dim      * inner_dim * num_stencil];// 抽取的A矩阵
    calc_t * L_high= new calc_t [ outer_dim * midle_dim      * inner_dim * lnz];
    calc_t * U_high= new calc_t [ outer_dim * midle_dim      * inner_dim * rnz];
    if (tmp_mem != nullptr) delete tmp_mem;
    if (buf_mem != nullptr) delete buf_mem;
    buf_mem = new calc_t  [(outer_dim * midle_dim + 4) * inner_dim]; buf = buf_mem + 2 * inner_dim;// 开多两柱保越界
    tmp_mem = new calc_t  [(outer_dim * midle_dim + 4) * inner_dim]; tmp = tmp_mem + 2 * inner_dim;
    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < (outer_dim * midle_dim + 4) * inner_dim; i++) {
        buf_mem[i] = 0.0;// 预先置零，免得出幺蛾子
        tmp_mem[i] = 0.0;
    }

    ibeg = x_lbdr ? A.halo_x : 0; iend = A.halo_x + A.local_x + (x_ubdr ? 0 : A.halo_x);
    jbeg =          A.halo_y    ; jend =     jbeg + A.local_y                          ;
    kbeg = z_lbdr ? A.halo_z : 0; kend = A.halo_z + A.local_z + (z_ubdr ? 0 : A.halo_z);
    // 保证拷贝范围的一致性
    assert(jend - jbeg == outer_dim); assert(iend - ibeg == midle_dim); assert(kend - kbeg == inner_dim);

    // extract vals from A_problem
    std::vector<idx_t> extract_ids;
    if (type == ILU_2D9) {
        assert(A.num_diag >= 9);
        if (A.num_diag == 19) {
            extract_ids = {5, 6, 7, 8, 9, 10, 11, 12, 13};
        } else if (A.num_diag == 27) {
            extract_ids = {9, 10, 11, 12, 13, 14, 15, 16, 17};
        } else if (A.num_diag == 9) {// 2d9
            extract_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        } else assert(false);
    }
    assert(extract_ids.size() == (std::size_t) num_stencil);

#define TRCIDX(d, k, i, j) (d) + num_stencil * ((k) + inner_dim * ((i) + midle_dim * (j)))
#define MATIDX(mat, d, k, i, j) (d) + (k) * (mat).num_diag + (i) * (mat).slice_dk_size + (j) * (mat).slice_dki_size
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
        const calc_t * src_ptr = A.data + MATIDX(A, 0, k, i, j);
        calc_t * dst_ptr = A_trc + TRCIDX(0, real_k, real_i, real_j);
        for (idx_t d = 0; d < num_stencil; d++)
            dst_ptr[d] = src_ptr[extract_ids[d]];
    }
#undef TRCIDX
#undef MATIDX

    #pragma omp parallel for schedule(static)
    for (idx_t j = jbeg; j < jend; j++) {// 逐面执行
        const idx_t off_s = (j-jbeg) * midle_dim * inner_dim;
        struct_ILU_2d(  A_trc + off_s * num_stencil,
                        L_high+ off_s * lnz, 
                        U_high+ off_s * rnz, 
                        midle_dim, inner_dim, num_stencil, stencil_offset);
    }

    delete A_trc;
    setup_called = true;

    {// 建立线程-任务的关系
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        idx_t num_slices = jend - jbeg;
        int max_nt = omp_get_max_threads(); assert(max_nt > 0);
        worker_jobID = new idx_t [max_nt];    for (int i = 0; i < max_nt; i++) worker_jobID[i] = -1;// initialized as -1
        worker_tidInGroup = new int [max_nt]; for (int i = 0; i < max_nt; i++) worker_tidInGroup[i] = -1;// initialized as -1
        if (max_nt <= num_slices) {
            cw_type = ROUND_ROBIN;
            if (my_pid == 0) printf(" max threads %d as Round-Robin to do %d PILU\n", max_nt, num_slices);
            delete worker_jobID; worker_jobID = nullptr;// 此时不需要记录
            delete worker_tidInGroup; worker_tidInGroup = nullptr;
        } else if (max_nt < 2 * num_slices || A.local_z <= 30) {// 数据内维太小了，不考虑多线程并行
            cw_type = COWORK_1On1;
            int worker_ids[num_slices];
            uniformly_distributed_integer(max_nt, num_slices, worker_ids);
            if (my_pid == 0) {
                printf(" max threads %d as DYN 1On1 to do %d PILU @cores ", max_nt, num_slices);
                for (idx_t i = 0; i < num_slices; i++) printf(" %d", worker_ids[i]);
                printf("\n");
            }
            for (idx_t s = 0; s < num_slices; s++) {
                int wid = worker_ids[s];
                worker_jobID[wid] = s;
                worker_tidInGroup[wid] = 0;
            }
        }
        // it must hold that nz > 30 for below branches
        else if (max_nt < 4 * num_slices) {
            cw_type = COWORK_2ON1;
            int worker_ids[2 * num_slices];
            uniformly_distributed_integer(max_nt, 2 * num_slices, worker_ids);
            if (my_pid == 0) {
                printf(" max threads %d as DYN 2On1 to do %d PILU @cores ", max_nt, num_slices);
                for (idx_t i = 0; i < num_slices; i++) printf(" (%d,%d)", worker_ids[i*2], worker_ids[i*2+1]);
                printf("\n");
            }
            for (idx_t s = 0; s < num_slices; s++) {
                int wid_0 = worker_ids[s*2], wid_1 = worker_ids[s*2+1];
                worker_jobID[wid_0] = worker_jobID[wid_1] = s;
                worker_tidInGroup[wid_0] = 0;
                worker_tidInGroup[wid_1] = 1;
            }
        } else if (max_nt < 8 * num_slices) {
            cw_type = COWORK_3ON1;
            int worker_ids[3 * num_slices];
            uniformly_distributed_integer(max_nt, 3 * num_slices, worker_ids);
            if (my_pid == 0) {
                printf(" max threads %d as DYN 3On1 to do %d PILU @cores ", max_nt, num_slices);
                for (idx_t i = 0; i < num_slices; i++) printf(" (%d,%d,%d)", worker_ids[i*3], worker_ids[i*3+1], worker_ids[i*3+2]);
                printf("\n");
            }
            for (idx_t s = 0; s < num_slices; s++) {
                int wid_0 = worker_ids[s*3], wid_1 = worker_ids[s*3+1], wid_2 = worker_ids[s*3+2];
                worker_jobID[wid_0] = worker_jobID[wid_1] = worker_jobID[wid_2] = s;
                worker_tidInGroup[wid_0] = 0;
                worker_tidInGroup[wid_1] = 1;
                worker_tidInGroup[wid_2] = 2;
            }
        } else {// max_nt >= 8 * num_slices
            cw_type = COWORK_4ON1;
            int worker_ids[4 * num_slices];
            uniformly_distributed_integer(max_nt, 4 * num_slices, worker_ids);
            if (my_pid == 0) {
                printf(" max threads %d as DYN 4On1 to do %d PILU @cores ", max_nt, num_slices);
                for (idx_t i = 0; i < num_slices; i++) printf(" (%d,%d,%d,%d)", worker_ids[i*4], worker_ids[i*4+1], worker_ids[i*4+2], worker_ids[i*4+3]);
                printf("\n");
            }
            for (idx_t s = 0; s < num_slices; s++) {
                int wid_0 = worker_ids[s*4], wid_1 = worker_ids[s*4+1], wid_2 = worker_ids[s*4+2], wid_3 = worker_ids[s*4+3];
                worker_jobID[wid_0] = worker_jobID[wid_1] = worker_jobID[wid_2] = worker_jobID[wid_3] = s;
                worker_tidInGroup[wid_0] = 0;
                worker_tidInGroup[wid_1] = 1;
                worker_tidInGroup[wid_2] = 2;
                worker_tidInGroup[wid_3] = 3;
            }
        }
    }

    if (L != nullptr) delete L;
    if (U != nullptr) delete U;
    shuffled = false;
    if constexpr (sizeof(data_t) == sizeof(calc_t)) {// 直接指向已开辟的内存即可，不需要释放
        L = L_high;
        U = U_high;
    } else {// 混合精度
        const idx_t tot_elems = outer_dim * midle_dim * inner_dim;
        L = new data_t [tot_elems * lnz];
        U = new data_t [tot_elems * rnz];
        if (cw_type == ROUND_ROBIN || cw_type == COWORK_1On1) {// 此时需要做AOS=>SOA的转换
            // 注意是逐层转换！！
            const idx_t plane_elems = midle_dim * inner_dim;// 每层的总元素数
            if (num_stencil == 9) {// 2d9
                assert(lnz == 4 && rnz == 5);
                // L:(0,1,2,3), U(0,1) (2,3,4)
                #pragma omp parallel for collapse(2) schedule(static)
                for (idx_t s = 0; s < outer_dim; s++) // 逐层
                for (idx_t e = 0; e < plane_elems; e++) {
                    data_t  *L_soa_0 = (L + s * plane_elems * lnz) + 0 * plane_elems + e * 4;
                    data_t  *U_soa_0 = (U + s * plane_elems * rnz) + 0 * plane_elems + e * 2,
                            *U_soa_1 = (U + s * plane_elems * rnz) + 2 * plane_elems + e * 3;
                    const calc_t* Lh_aos = L_high + (s * plane_elems + e)* lnz,
                                * Uh_aos = U_high + (s * plane_elems + e)* rnz;
                    L_soa_0[0] = Lh_aos[0]; L_soa_0[1] = Lh_aos[1]; L_soa_0[2] = Lh_aos[2]; L_soa_0[3] = Lh_aos[3];
                    U_soa_0[0] = Uh_aos[0]; U_soa_0[1] = Uh_aos[1];
                    U_soa_1[0] = Uh_aos[2]; U_soa_1[1] = Uh_aos[3]; U_soa_1[2] = Uh_aos[4];
                }
            }
            else assert(false);

            shuffled = true;
        } else {// 否则采用level-based的求解时不需要转换
            #pragma omp parallel for schedule(static)
            for (idx_t i = 0; i < tot_elems * lnz; i++)
                L[i] = L_high[i];
            #pragma omp parallel for schedule(static)
            for (idx_t i = 0; i < tot_elems * rnz; i++)
                U[i] = U_high[i];
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void PlaneILU<idx_t, data_t, calc_t>::apply() const {
    
#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    MPI_Barrier(MPI_COMM_WORLD);
    double t = wall_time();
#endif
    const idx_t vec_off = midle_dim * inner_dim;
    const idx_t num_slices = jend - jbeg;
    if (cw_type == ROUND_ROBIN) {
        #pragma omp parallel
        {
            // 注意务必配合ILU分解时，对不在本进程范围内的列直接赋0
            #pragma omp for schedule(static)
            for (idx_t s = 0; s < num_slices; s++) {// 逐面执行
                // 前代（Forward），求解Ly=b，注意stencil_offset从0（第一个左非零元位置）开始传
                struct_2d5_trsv_forward_hardCode(
                    L + s*vec_off*lnz, buf + s*vec_off, tmp + s*vec_off,// 注意偏移
                    midle_dim, inner_dim, lnz, stencil_offset, shuffled);
                // 回代（Backward），求解Ux=y，注意stencil_offset从第一个右非零元（含对角元）位置开始传
                struct_2d5_trsv_backward_hardCode(// 要与前代结果存储的位置相符
                    U + s*vec_off*rnz, tmp + s*vec_off, buf + s*vec_off,
                    midle_dim, inner_dim, rnz, stencil_offset + 2 * lnz, shuffled);
            }
        }
    }
    else if (cw_type == COWORK_1On1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            idx_t j = worker_jobID[tid];
            if (j != -1) {
                struct_2d5_trsv_forward_hardCode(
                    L + j*vec_off*lnz, buf + j*vec_off, tmp + j*vec_off,// 注意偏移
                    midle_dim, inner_dim, lnz, stencil_offset, shuffled);
                struct_2d5_trsv_backward_hardCode(
                    U + j*vec_off*rnz, tmp + j*vec_off, buf + j*vec_off,
                    midle_dim, inner_dim, rnz, stencil_offset + 2 * lnz, shuffled);
            }
        }
    }
    else {// DYNAMIC_2On1 || DYNAMIC_3On1 || DYNAMIC_4On1
        assert(!shuffled);
        const idx_t group_size = cw_type;
        group_sptrsv_2d5_levelbased_forward_hardCode(
            L, buf, tmp, num_slices, midle_dim, inner_dim, lnz, stencil_offset          , group_size, worker_tidInGroup, worker_jobID);
        group_sptrsv_2d5_levelbased_backward_hardCode(
            U, tmp, buf, num_slices, midle_dim, inner_dim, rnz, stencil_offset + 2 * lnz, group_size, worker_tidInGroup, worker_jobID);
    }
#ifdef PROFILE
    t = wall_time() - t;
    double mint, maxt;
    MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (my_pid == 0) {
        double bytes = outer_dim * midle_dim * inner_dim * num_stencil * sizeof(data_t);// 矩阵
        bytes       += outer_dim * midle_dim * inner_dim *         3   * sizeof(calc_t);// buf -> tmp -> buf
        bytes = bytes * num_procs / (1024 * 1024 * 1024);// GB
        printf("PILU data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                sizeof(data_t), sizeof(calc_t), num_stencil, bytes, mint, maxt, bytes/maxt, bytes/mint);
    }
#endif
}

template<typename idx_t, typename data_t, typename calc_t>
void PlaneILU<idx_t, data_t, calc_t>::Mult(const par_structVector<idx_t, calc_t> & input, par_structVector<idx_t, calc_t> & output) const
{
    CHECK_LOCAL_HALO( *(input.local_vector),  *(output.local_vector));// 检查相容性

    if (this->zero_guess) {// 特别地，当output为0向量时，与直接解方程无差异
        const seq_structVector<idx_t, calc_t> & b = *(input.local_vector);
        seq_structVector<idx_t, calc_t> & x = *(output.local_vector);
    
        input.update_halo();// Additive Schwartz，并不需要南北的halo区填充

        // 拷贝到buf里
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t real_j = j - jbeg, real_i = i - ibeg;
            const calc_t * src_ptr = b.data + j * b.slice_ki_size + i * b.slice_k_size + kbeg;
            calc_t * dst_ptr = buf + (real_j * midle_dim + real_i) * inner_dim;// 不需要加kbeg
            for (idx_t k = 0; k < inner_dim; k++)// 拷贝连续的一柱
                dst_ptr[k] = src_ptr[k];
        }
        apply();
        // 从buf里拷回
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t real_j = j - jbeg, real_i = i - ibeg;
            const calc_t * src_ptr = buf + (real_j * midle_dim + real_i) * inner_dim;// 不需要加kbeg
            calc_t * dst_ptr = x.data + j * x.slice_ki_size + i * x.slice_k_size + kbeg;
            for (idx_t k = 0; k < inner_dim; k++)// 拷贝连续的一柱
                dst_ptr[k] = src_ptr[k];
        }
        if (this->weight != 1.0) vec_mul_by_scalar(this->weight, output, output);
    }
    else {
        par_structVector<idx_t, calc_t> resi(input), error(output);
        this->oper->Mult(output, resi, false);

        vec_add(input, -1.0, resi, resi);
        
        resi.update_halo();// Additive Schwartz，并不需要南北的halo区填充
        // 拷贝到buf里
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t real_j = j - jbeg, real_i = i - ibeg;
            const calc_t * src_ptr = resi.local_vector->data + j * resi.local_vector->slice_ki_size
                                    + i * resi.local_vector->slice_k_size + kbeg;
            calc_t * dst_ptr = buf + (real_j * midle_dim + real_i) * inner_dim;// 不需要加kbeg
            for (idx_t k = 0; k < inner_dim; k++)// 拷贝连续的一柱
                dst_ptr[k] = src_ptr[k];
        }
        apply();
        // 从buf里拷出
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t real_j = j - jbeg, real_i = i - ibeg;
            const calc_t * src_ptr = buf + (real_j * midle_dim + real_i) * inner_dim;// 不需要加kbeg
            calc_t * dst_ptr = error.local_vector->data + j * error.local_vector->slice_ki_size
                                                        + i * error.local_vector->slice_k_size + kbeg;
            for (idx_t k = 0; k < inner_dim; k++)// 拷贝连续的一柱
                dst_ptr[k] = src_ptr[k];
        }
        vec_add(output, this->weight, error, output);
    }
}

#endif