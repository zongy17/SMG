#ifndef SMG_BLOCK_ILU_HPP
#define SMG_BLOCK_ILU_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"
#include "../utils/ILU.hpp"
#include "../utils/ILU_hardcode.hpp"

typedef enum {ILU_3D7, ILU_3D15, ILU_3D19, ILU_3D27} BlockILU_type;

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
class BlockILU final : public Solver<idx_t, data_t, oper_t, res_t> {
public:
    BlockILU_type type;
    idx_t outer_dim = 0, midle_dim = 0, inner_dim = 0;
    idx_t jbeg = 0, jend = 0, ibeg = 0, iend = 0, kbeg = 0, kend = 0;
    idx_t lnz = 0, rnz = 0;
    data_t * L = nullptr, * U = nullptr;
    res_t * tmp = nullptr, * tmp_mem = nullptr;// 中间变量
    res_t * buf = nullptr, * buf_mem = nullptr;// 用于数据转换的
    idx_t num_stencil = 0;// 未初始化
    const idx_t * stencil_offset = nullptr;
    bool setup_called = false;

    const Operator<idx_t, oper_t, res_t> * oper = nullptr;// operator (often as matrix-A)

    BlockILU(BlockILU_type type): Solver<idx_t, data_t, oper_t, res_t>(), type(type) {
        if (type == ILU_3D7) {
            num_stencil = 7;
            stencil_offset = stencil_offset_3d7;
        }
        else if (type == ILU_3D15) {
            num_stencil = 15;
            stencil_offset = stencil_offset_3d15;
        }
        else if (type == ILU_3D19) {
            num_stencil = 19;
            stencil_offset = stencil_offset_3d19;
        }
        else if (type == ILU_3D27) {
            num_stencil = 27;
            stencil_offset = stencil_offset_3d27;
        } 
        else {
            printf("Not supported ilu type %d! Only ILU_3d7 or _3d27 available!\n", type);
            MPI_Abort(MPI_COMM_WORLD, -99);
        }
    }

    ~BlockILU();
    virtual void SetOperator(const Operator<idx_t, oper_t, res_t> & op) {
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        Setup();
    }

    void Setup();

    void truncate() {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: BILU truncated\n");
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

protected:
    void Mult(const par_structVector<idx_t, res_t> & input, 
                    par_structVector<idx_t, res_t> & output) const ;
public:
    void Mult(const par_structVector<idx_t, res_t> & input, 
                    par_structVector<idx_t, res_t> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }
};

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
BlockILU<idx_t, data_t, oper_t, res_t>::~BlockILU() {
    if (L != nullptr) {delete L; L = nullptr;}
    if (U != nullptr) {delete U; U = nullptr;}
    if (buf_mem != nullptr) {delete buf_mem; buf_mem = nullptr;}
    if (tmp_mem != nullptr) {delete tmp_mem; tmp_mem = nullptr;}
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void BlockILU<idx_t, data_t, oper_t, res_t>::Setup() {
    if (setup_called) return ;

    assert(this->oper != nullptr);
    // assert matrix has updated halo to prepare data 强制类型转换
    const par_structMatrix<idx_t, oper_t, res_t> & par_A = *((par_structMatrix<idx_t, oper_t, res_t>*)(this->oper));
    const seq_structMatrix<idx_t, oper_t, res_t> & A = *(par_A.local_matrix);

    if (sizeof(oper_t) != sizeof(data_t)) {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) {
            printf("  \033[1;31mWarning\033[0m: BILU::Setup() using oper_t of %ld bytes, but data_t of %ld bytes", sizeof(oper_t), sizeof(data_t));
            printf(" ===> still factorized it: but instead should consider to use higher precision to Setup then copy with truncation.\n");
        }
    }

    data_t * A_trc = nullptr;// 抽取的A矩阵
    // 确定各维上是否是边界
    const bool x_lbdr = par_A.comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = par_A.comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = par_A.comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = par_A.comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = par_A.comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = par_A.comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;
    outer_dim = (y_lbdr ? 0 : A.halo_y) + A.local_y + (y_ubdr ? 0 : A.halo_y),
    midle_dim = (x_lbdr ? 0 : A.halo_x) + A.local_x + (x_ubdr ? 0 : A.halo_x),// 注意：ILU分解的区域包含halo区，Additive Schwartz
    inner_dim = (z_lbdr ? 0 : A.halo_z) + A.local_z + (z_ubdr ? 0 : A.halo_z); 
    
    assert(num_stencil % 2 != 0);
    lnz = (num_stencil - 1) >> 1;
    rnz =  num_stencil - lnz;
    A_trc = new data_t [outer_dim * midle_dim * inner_dim * num_stencil];
    if (L != nullptr) delete L;
    if (U != nullptr) delete U;
    if (tmp != nullptr) delete tmp;
    if (buf != nullptr) delete buf;
    L       = new data_t [ outer_dim * midle_dim      * inner_dim * lnz];
    U       = new data_t [ outer_dim * midle_dim      * inner_dim * rnz];
    tmp_mem = new res_t  [(outer_dim * midle_dim + 4) * inner_dim]; tmp = tmp_mem + 2 * inner_dim;
    buf_mem = new res_t  [(outer_dim * midle_dim + 4) * inner_dim]; buf = buf_mem + 2 * inner_dim;
    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < (outer_dim * midle_dim + 4) * inner_dim; i++) {
        buf_mem[i] = 0.0;// 预先置零，免得出幺蛾子
        tmp_mem[i] = 0.0;
    }
    
    ibeg = x_lbdr ? A.halo_x : 0; iend = A.halo_x + A.local_x + (x_ubdr ? 0 : A.halo_x);
    jbeg = y_lbdr ? A.halo_y : 0; jend = A.halo_y + A.local_y + (y_ubdr ? 0 : A.halo_y);
    kbeg = z_lbdr ? A.halo_z : 0; kend = A.halo_z + A.local_z + (z_ubdr ? 0 : A.halo_z);
    // 保证拷贝范围的一致性
    assert(jend - jbeg == outer_dim); assert(iend - ibeg == midle_dim); assert(kend - kbeg == inner_dim);

    // int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    // printf(" proc %d out %d mid %d in %d j [%d,%d) i [%d,%d) k [%d,%d)\n", 
    //     my_pid, outer_dim, midle_dim, inner_dim, jbeg, jend, ibeg, iend, kbeg, kend);

    // extract vals from A_probelm
#define TRCIDX(d, k, i, j) (d) + num_stencil * ((k) + inner_dim * ((i) + midle_dim * (j)))
#define MATIDX(mat, d, k, i, j) (d) + (k) * (mat).num_diag + (i) * (mat).slice_dk_size + (j) * (mat).slice_dki_size

    if (type == ILU_3D7) {
        assert(A.num_diag >= 7);
        if (A.num_diag == 7) {
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++)
            for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
                idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
                for (idx_t d = 0; d < A.num_diag; d++) 
                    A_trc[TRCIDX(d, real_k, real_i, real_j)] = A.data[MATIDX(A, d, k, i, j)];
            }
        }
        else if (A.num_diag == 19) {// 从3d19矩阵里提取3d7的元素进行不完全分解
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++)
            for (idx_t k = kbeg; k < kend; k++) {
                idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
                const data_t * src_ptr = A.data + MATIDX(A, 0, k, i, j);
                data_t * dst_ptr = A_trc + TRCIDX(0, real_k, real_i, real_j);
                dst_ptr[0] = src_ptr[2];
                dst_ptr[1] = src_ptr[6];
                dst_ptr[2] = src_ptr[8];
                dst_ptr[3] = src_ptr[9];
                dst_ptr[4] = src_ptr[10];
                dst_ptr[5] = src_ptr[12];
                dst_ptr[6] = src_ptr[16];
            }
        }
        else if (A.num_diag == 27) {
            assert(false);
        }
        else {
            assert(false);
        }
    }
    else if (type == ILU_3D15) {
        assert(A.num_diag >= 15);
        if (A.num_diag == 19) {
            // for (idx_t j = jbeg; j < jend; j++)
            // for (idx_t i = ibeg; i < iend; i++)
            // for (idx_t k = kbeg; k < kend; k++) {
            //     idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            //     A_trc[TRCIDX(0, real_k, real_i, real_j)] = A.data[MATIDX(A, 1, k, i, j)];
            // }
            assert(false);
        }
        else if (A.num_diag == 27) {
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++)
            for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
                idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
                const data_t * src_ptr = A.data + MATIDX(A, 0, k, i, j);
                data_t * dst_ptr = A_trc + TRCIDX(0, real_k, real_i, real_j);
                dst_ptr[0] = src_ptr[3]; dst_ptr[1] = src_ptr[4]; dst_ptr[2] = src_ptr[5];

                dst_ptr[3] = src_ptr[9]; dst_ptr[4] = src_ptr[10];dst_ptr[5] = src_ptr[11];
                dst_ptr[6] = src_ptr[12];dst_ptr[7] = src_ptr[13];dst_ptr[8] = src_ptr[14];
                dst_ptr[9] = src_ptr[15];dst_ptr[10]= src_ptr[16];dst_ptr[11]= src_ptr[17];

                dst_ptr[12]= src_ptr[21];dst_ptr[13]= src_ptr[22];dst_ptr[14]= src_ptr[23];
            }
        }
        else assert(false);
    }
    else if (type == ILU_3D19) {
        assert(A.num_diag >= 19);
        if (A.num_diag == 19) {
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t i = ibeg; i < iend; i++)
            for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
                idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
                for (idx_t d = 0; d < A.num_diag; d++) 
                    A_trc[TRCIDX(d, real_k, real_i, real_j)] = A.data[MATIDX(A, d, k, i, j)];
            }
        }
        else {
            assert(false);
        }
    } 
    else if (type == ILU_3D27) {
        assert(A.num_diag >= 27);
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++)
        for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
            idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
            for (idx_t d = 0; d < A.num_diag; d++) 
                A_trc[TRCIDX(d, real_k, real_i, real_j)] = A.data[MATIDX(A, d, k, i, j)];
        }
    } 
    else {
        MPI_Abort(par_A.comm_pkg->cart_comm, -555663);
    }
#undef TRCIDX
#undef MATIDX
    // 执行ILU分解，并存储

#ifdef DEBUG
    {
        int my_pid; MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
        idx_t tot_elems = outer_dim * midle_dim * inner_dim;
        printf("proc %d tot elems %d\n", my_pid, tot_elems);
        FILE * fp = fopen(("p"+std::to_string(my_pid)+".A_trc").c_str(), "w+");
        for (idx_t j = 0; j < outer_dim; j++)
        for (idx_t i = 0; i < midle_dim; i++)
        for (idx_t k = 0; k < inner_dim; k++) {
            idx_t r = k + inner_dim * (i + j * midle_dim);
            for (idx_t d = 0; d < num_stencil; d++) {
                idx_t   ngb_j = j + stencil_offset[d*3  ],
                        ngb_i = i + stencil_offset[d*3+1],
                        ngb_k = k + stencil_offset[d*3+2];
                if (ngb_j < 0 || ngb_j >= outer_dim ||
                    ngb_i < 0 || ngb_i >= midle_dim ||
                    ngb_k < 0 || ngb_k >= inner_dim)
                    continue;
                idx_t c = ngb_k + inner_dim * (ngb_i + ngb_j * midle_dim);
                fprintf(fp, "%d %d %.6e\n", r, c, A_trc[r * num_stencil + d]);
            }
        }
        fclose(fp);
    }
#endif
    struct_ILU_3d(A_trc, L, U, outer_dim, midle_dim, inner_dim, num_stencil, stencil_offset);
#ifdef DEBUG
    {
        int my_pid; MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
        FILE * fp = fopen(("p"+std::to_string(my_pid)+".L").c_str(), "w+");
        for (idx_t j = 0; j < outer_dim; j++)
        for (idx_t i = 0; i < midle_dim; i++)
        for (idx_t k = 0; k < inner_dim; k++) {
            idx_t r = k + inner_dim * (i + j * midle_dim);
            for (idx_t d = 0; d < lnz; d++) {
                idx_t   ngb_j = j + stencil_offset[d*3  ],
                        ngb_i = i + stencil_offset[d*3+1],
                        ngb_k = k + stencil_offset[d*3+2];
                if (ngb_j < 0 || ngb_j >= outer_dim ||
                    ngb_i < 0 || ngb_i >= midle_dim ||
                    ngb_k < 0 || ngb_k >= inner_dim)
                    continue;
                idx_t c = ngb_k + inner_dim * (ngb_i + ngb_j * midle_dim);
                fprintf(fp, "%d %d %.6e\n", r, c, L[r * lnz + d]);
            }
        }
        fclose(fp);
    }
    {
        int my_pid; MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
        FILE * fp = fopen(("p"+std::to_string(my_pid)+".U").c_str(), "w+");
        for (idx_t j = 0; j < outer_dim; j++)
        for (idx_t i = 0; i < midle_dim; i++)
        for (idx_t k = 0; k < inner_dim; k++) {
            idx_t r = k + inner_dim * (i + j * midle_dim);
            for (idx_t d = 0; d < rnz; d++) {
                idx_t   ngb_j = j + stencil_offset[(d+lnz)*3  ],
                        ngb_i = i + stencil_offset[(d+lnz)*3+1],
                        ngb_k = k + stencil_offset[(d+lnz)*3+2];
                if (ngb_j < 0 || ngb_j >= outer_dim ||
                    ngb_i < 0 || ngb_i >= midle_dim ||
                    ngb_k < 0 || ngb_k >= inner_dim)
                    continue;
                idx_t c = ngb_k + inner_dim * (ngb_i + ngb_j * midle_dim);
                fprintf(fp, "%d %d %.6e\n", r, c, U[r * rnz + d]);
            }
        }
        fclose(fp);
    }
#endif
    delete A_trc;
    setup_called = true;
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void BlockILU<idx_t, data_t, oper_t, res_t>::Mult(const par_structVector<idx_t, res_t> & input, par_structVector<idx_t, res_t> & output) const
{
    CHECK_LOCAL_HALO(*(input.local_vector),  *(output.local_vector));// 检查相容性
    assert(outer_dim && midle_dim && inner_dim);

    int my_pid; MPI_Comm_rank(input.comm_pkg->cart_comm, &my_pid);
    
    if (this->zero_guess) {
        const seq_structVector<idx_t, res_t> & b = *(input.local_vector);
        seq_structVector<idx_t, res_t> & x = *(output.local_vector);
        
        input.update_halo();// Additive Schwartz要求右端项填充边界
        // s = L^{-1}*b

        // 拷贝到buf里
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++)
        for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
            idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
            buf[(real_j * midle_dim + real_i) * inner_dim + real_k] = b.data[j * b.slice_ki_size + i * b.slice_k_size + k];
        }
#ifdef DEBUG
        {
            FILE * fp = fopen(("p"+std::to_string(my_pid)+".rhs").c_str(), "w+");
            for (idx_t j = 0; j < outer_dim; j++)
            for (idx_t i = 0; i < midle_dim; i++)
            for (idx_t k = 0; k < inner_dim; k++) {
                idx_t r = k + inner_dim * (i + midle_dim * j);
                fprintf(fp, "%d %lf\n", r, buf[r]);
            }
            fclose(fp);
        }
#endif
        // struct_trsv_forward_3d (L, buf, tmp, outer_dim, midle_dim, inner_dim, (num_stencil - 1)>>1, stencil_offset                  );
        struct_sptrsv_3d_forward_frame_hardCode (L, buf, tmp, outer_dim, midle_dim, inner_dim, lnz, stencil_offset);
#ifdef DEBUG
        {
            FILE * fp = fopen(("p"+std::to_string(my_pid)+".tmp").c_str(), "w+");
            for (idx_t j = 0; j < outer_dim; j++)
            for (idx_t i = 0; i < midle_dim; i++)
            for (idx_t k = 0; k < inner_dim; k++) {
                idx_t r = k + inner_dim * (i + midle_dim * j);
                fprintf(fp, "%d %lf\n", r, tmp[r]);
            }
            fclose(fp);
        }
#endif
        // x = U^{-1}*s = U^{-1}*L^{-1}*b
        // struct_trsv_backward_3d(U, tmp, buf, outer_dim, midle_dim, inner_dim, (num_stencil + 1)>>1, stencil_offset + 3 * ((num_stencil - 1)>>1));
        struct_sptrsv_3d_backward_frame_hardCode(U, tmp, buf, outer_dim, midle_dim, inner_dim, rnz, stencil_offset + lnz*3);

#ifdef DEBUG
        {
            FILE * fp = fopen(("p"+std::to_string(my_pid)+".res").c_str(), "w+");
            for (idx_t j = 0; j < outer_dim; j++)
            for (idx_t i = 0; i < midle_dim; i++)
            for (idx_t k = 0; k < inner_dim; k++) {
                idx_t r = k + inner_dim * (i + midle_dim * j);
                fprintf(fp, "%d %lf\n", r, buf[r]);
            }
            fclose(fp);
        }
        MPI_Abort(input.comm_pkg->cart_comm, -20230120);
#endif
        // 从buf里拷出
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++)
        for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
            idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
            x.data[j * b.slice_ki_size + i * b.slice_k_size + k] = buf[(real_j * midle_dim + real_i) * inner_dim + real_k];
        }

        // x = weight*x = weight*U^{-1}*L^{-1}*b
        if (this->weight != 1.0) vec_mul_by_scalar(this->weight, output, output);
    } 
    else {
        par_structVector<idx_t, res_t> resi(input), error(output);

        // resi = A*x
        this->oper->Mult(output, resi, false);
        
        vec_add(input, -1.0, resi, resi);
        resi.update_halo();// Additive Schwartz要求右端项填充边界

        // 拷贝到buf里
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++)
        for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
            idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
            buf[(real_j * midle_dim + real_i) * inner_dim + real_k] 
                = resi.local_vector->data[j * resi.local_vector->slice_ki_size + i * resi.local_vector->slice_k_size + k];
        }

        // s = L^{-1}*resi = L^{-1}*(b - A*x)
        // struct_trsv_forward_3d (L, buf, tmp, outer_dim, midle_dim, inner_dim, (num_stencil - 1)>>1, stencil_offset                  );
        struct_sptrsv_3d_forward_frame_hardCode (L, buf, tmp, outer_dim, midle_dim, inner_dim, lnz, stencil_offset);
        // e = U^{-1}*s = U^{-1}*L^{-1}*(b - A*x)

        // struct_trsv_backward_3d(U, tmp, buf, outer_dim, midle_dim, inner_dim, (num_stencil + 1)>>1, stencil_offset + 3 * ((num_stencil - 1)>>1));
        struct_sptrsv_3d_backward_frame_hardCode(U, tmp, buf, outer_dim, midle_dim, inner_dim, rnz, stencil_offset + lnz*3);

        // 从buf里拷出
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++)
        for (idx_t k = kbeg; k < kend; k++) {// 只拷贝内部的本进程负责的区域
            idx_t real_j = j - jbeg, real_i = i - ibeg, real_k = k - kbeg;
            error.local_vector->data[j * error.local_vector->slice_ki_size + i * error.local_vector->slice_k_size + k] 
                = buf[(real_j * midle_dim + real_i) * inner_dim + real_k];
        }
        
        // x = x + w*e = x + w*U^{-1}*L^{-1}*(b - A*x)
        vec_add(output, this->weight, error, output);
    }
}

#endif