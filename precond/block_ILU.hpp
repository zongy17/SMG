#ifndef SMG_BLOCK_ILU_HPP
#define SMG_BLOCK_ILU_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"
#include "../utils/ILU.hpp"
#include "../utils/ILU_hardcode.hpp"

typedef enum {ILU_3D7, ILU_3D15, ILU_3D19, ILU_3D27} BlockILU_type;

template<typename idx_t, typename data_t, typename calc_t>
class BlockILU final : public Solver<idx_t, data_t, calc_t> {
public:
    BlockILU_type type;
    idx_t outer_dim = 0, midle_dim = 0, inner_dim = 0;
    idx_t jbeg = 0, jend = 0, ibeg = 0, iend = 0, kbeg = 0, kend = 0;
    idx_t lnz = 0, rnz = 0;
    data_t * L = nullptr, * U = nullptr;
    calc_t * tmp = nullptr, * tmp_mem = nullptr;// 中间变量
    calc_t * buf = nullptr, * buf_mem = nullptr;// 用于数据转换的
    idx_t num_stencil = 0;// 未初始化
    const idx_t * stencil_offset = nullptr;
    bool setup_called = false;

    const Operator<idx_t, calc_t, calc_t> * oper = nullptr;// operator (often as matrix-A)

    BlockILU(BlockILU_type type): Solver<idx_t, data_t, calc_t>(), type(type) {
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
    virtual void SetOperator(const Operator<idx_t, calc_t, calc_t> & op) {
#ifdef COMPRESS
        assert(((const par_structMatrix<idx_t, calc_t, calc_t>&)op).compressed == false);// 暂时先不处理BILU等的压缩情况
#endif
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
    void Mult(const par_structVector<idx_t, calc_t> & input, 
                    par_structVector<idx_t, calc_t> & output) const ;
public:
    void Mult(const par_structVector<idx_t, calc_t> & input, 
                    par_structVector<idx_t, calc_t> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }
};

template<typename idx_t, typename data_t, typename calc_t>
BlockILU<idx_t, data_t, calc_t>::~BlockILU() {
    if (L != nullptr) {delete L; L = nullptr;}
    if (U != nullptr) {delete U; U = nullptr;}
    if (buf_mem != nullptr) {delete buf_mem; buf_mem = nullptr;}
    if (tmp_mem != nullptr) {delete tmp_mem; tmp_mem = nullptr;}
}

template<typename idx_t, typename data_t, typename calc_t>
void BlockILU<idx_t, data_t, calc_t>::Setup() {
    if (setup_called) return ;

    assert(this->oper != nullptr);
    // assert matrix has updated halo to prepare data 强制类型转换
    const par_structMatrix<idx_t, calc_t, calc_t> & par_A = *((par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper));
    const seq_structMatrix<idx_t, calc_t, calc_t> & A = *(par_A.local_matrix);

    if constexpr (sizeof(calc_t) != sizeof(data_t)) {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) {
            printf("  \033[1;31mWarning\033[0m: BILU::Setup() using calc_t of %ld bytes, but data_t of %ld bytes\n",
                sizeof(calc_t), sizeof(data_t));
        }
    }

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
    calc_t * A_trc   = new calc_t [outer_dim * midle_dim * inner_dim * num_stencil];// 抽取的A矩阵
    calc_t * L_high  = new calc_t [outer_dim * midle_dim * inner_dim * lnz];// 用于高精度分解
    calc_t * U_high  = new calc_t [outer_dim * midle_dim * inner_dim * rnz];// 用于高精度分解
    if (tmp_mem != nullptr) delete tmp_mem;
    if (buf_mem != nullptr) delete buf_mem;
    const idx_t out_beg = 1, mid_beg = 1, in_beg = 1;
    tmp_mem = new calc_t  [(outer_dim + 2) * (midle_dim + 2) * (inner_dim + 2)];
    buf_mem = new calc_t  [(outer_dim + 2) * (midle_dim + 2) * (inner_dim + 2)];
    const idx_t off_mem = (out_beg * (midle_dim + 2) + mid_beg) * (inner_dim + 2) + in_beg;
    tmp = tmp_mem + off_mem;
    buf = buf_mem + off_mem;
    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < (outer_dim + 2) * (midle_dim + 2) * (inner_dim + 2); i++) {
        buf_mem[i] = 0.0;// 预先置零，免得出幺蛾子
        tmp_mem[i] = 0.0;
    }
    
    ibeg = x_lbdr ? A.halo_x : 0; iend = A.halo_x + A.local_x + (x_ubdr ? 0 : A.halo_x);
    jbeg = y_lbdr ? A.halo_y : 0; jend = A.halo_y + A.local_y + (y_ubdr ? 0 : A.halo_y);
    kbeg = z_lbdr ? A.halo_z : 0; kend = A.halo_z + A.local_z + (z_ubdr ? 0 : A.halo_z);
    // 保证拷贝范围的一致性
    assert(jend - jbeg == outer_dim); assert(iend - ibeg == midle_dim); assert(kend - kbeg == inner_dim);

    // extract vals from A_probelm
    std::vector<idx_t> extract_ids;
    if (type == ILU_3D7) {
        assert(A.num_diag >= 7);
        if (A.num_diag == 7) {
            extract_ids = {0, 1, 2, 3, 4, 5, 6};
        } else if (A.num_diag == 19) {// 从3d19矩阵里提取3d7的元素进行不完全分解
            extract_ids = {2, 6, 8, 9, 10, 12, 16};
        } else if (A.num_diag == 27) {
            extract_ids = {4, 10, 12, 13, 14, 16, 22};
        } else assert(false);
    }
    else if (type == ILU_3D15) {
        assert(A.num_diag >= 15);
        if (A.num_diag == 19) {
            extract_ids = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17};
        } else if (A.num_diag == 27) {
            extract_ids = {3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23};
        } else assert(false);
    }
    else if (type == ILU_3D19) {
        assert(A.num_diag >= 19);
        if (A.num_diag == 19) {
            extract_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        } else if (A.num_diag == 27) {
            extract_ids = {1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 25};
        } else assert(false);
    } 
    else if (type == ILU_3D27) {
        assert(A.num_diag >= 27);
        if (A.num_diag == 27) {
            extract_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21, 22, 23, 24, 25, 26};
        } else assert(false);
    } 
    else {
        MPI_Abort(par_A.comm_pkg->cart_comm, -555663);
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
    struct_ILU_3d(A_trc, L_high, U_high, outer_dim, midle_dim, inner_dim, num_stencil, stencil_offset);
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
                fprintf(fp, "%d %d %.6e\n", r, c, L_high[r * lnz + d]);
            }
            fprintf(fp, "%d %d %.6e\n", r, r, 1.0);// 自己的对角线非零元
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
                fprintf(fp, "%d %d %.6e\n", r, c, U_high[r * rnz + d]);
            }
        }
        fclose(fp);
    }
#endif
    delete A_trc;
    if (L != nullptr) delete L;
    if (U != nullptr) delete U;
    if constexpr (sizeof(data_t) == sizeof(calc_t)) {// 直接指向已开辟的内存即可，不需要释放
        L = L_high;
        U = U_high;
    } else {// 否则，拷贝截断，并做AOS=>SOA的转换，最后释放高精度的数组
        const idx_t tot_elems = outer_dim * midle_dim * inner_dim;
        L = new data_t [tot_elems * lnz];
        U = new data_t [tot_elems * rnz];
        // 类似于PGS的方式组织
        if (num_stencil == 7) {
            assert(lnz == 3 && rnz == 4);
            // L:(0,1,2), U:(0,1,2,3) 其中U(0)是主对角元
            #pragma omp parallel for schedule(static)
            for (idx_t e = 0; e < tot_elems; e++) {
                const calc_t* Lh_aos = L_high + e * lnz,
                            * Uh_aos = U_high + e * rnz;
                data_t  *L_soa_0 = L + 0 * tot_elems + e * 3,
                        *U_soa_0 = U + 0 * tot_elems + e * 4;
                L_soa_0[0] = Lh_aos[0]; L_soa_0[1] = Lh_aos[1]; L_soa_0[2] = Lh_aos[2];

                U_soa_0[0] = Uh_aos[0]; U_soa_0[1] = Uh_aos[1]; U_soa_0[2] = Uh_aos[2]; U_soa_0[3] = Uh_aos[3];
            }
        }
        else if (num_stencil == 15) {
            assert(lnz == 7 && rnz == 8);
            // L:(0,1,2) (3,4,5,6),  U:(0,1,2,3),(4,5,6,7) 其中U(0)是主对角元
            #pragma omp parallel for schedule(static)
            for (idx_t e = 0; e < tot_elems; e++) {
                const calc_t* Lh_aos = L_high + e * lnz,
                            * Uh_aos = U_high + e * rnz;
                data_t  *L_soa_0 = L + 0 * tot_elems + e * 3,
                        *L_soa_1 = L + 3 * tot_elems + e * 4,
                        *U_soa_0 = U + 0 * tot_elems + e * 4,
                        *U_soa_1 = U + 4 * tot_elems + e * 4;
                L_soa_0[0] = Lh_aos[0]; L_soa_0[1] = Lh_aos[1]; L_soa_0[2] = Lh_aos[2];
                L_soa_1[0] = Lh_aos[3]; L_soa_1[1] = Lh_aos[4]; L_soa_1[2] = Lh_aos[5]; L_soa_1[3] = Lh_aos[6];
                
                U_soa_0[0] = Uh_aos[0]; U_soa_0[1] = Uh_aos[1]; U_soa_0[2] = Uh_aos[2]; U_soa_0[3] = Uh_aos[3];
                U_soa_1[0] = Uh_aos[4]; U_soa_1[1] = Uh_aos[5]; U_soa_1[2] = Uh_aos[6]; U_soa_1[3] = Uh_aos[7];
            }
        }
        else if (num_stencil == 19) {
            assert(lnz == 9 && rnz == 10);
            // L:(0,1,2) (3,4,5) (6,7,8),  U:(0,1,2),(3,4,5),(6,7,8,9) 其中U(0)是主对角元
            #pragma omp parallel for schedule(static)
            for (idx_t e = 0; e < tot_elems; e++) {
                const calc_t* Lh_aos = L_high + e * lnz,
                            * Uh_aos = U_high + e * rnz;
                data_t  *L_soa_0 = L + 0 * tot_elems + e * 3,
                        *L_soa_1 = L + 3 * tot_elems + e * 3,
                        *L_soa_2 = L + 6 * tot_elems + e * 3,
                        *U_soa_0 = U + 0 * tot_elems + e * 3,
                        *U_soa_1 = U + 3 * tot_elems + e * 3,
                        *U_soa_2 = U + 6 * tot_elems + e * 4;
                L_soa_0[0] = Lh_aos[0]; L_soa_0[1] = Lh_aos[1]; L_soa_0[2] = Lh_aos[2];
                L_soa_1[0] = Lh_aos[3]; L_soa_1[1] = Lh_aos[4]; L_soa_1[2] = Lh_aos[5];
                L_soa_2[0] = Lh_aos[6]; L_soa_2[1] = Lh_aos[7]; L_soa_2[2] = Lh_aos[8];

                U_soa_0[0] = Uh_aos[0]; U_soa_0[1] = Uh_aos[1]; U_soa_0[2] = Uh_aos[2];
                U_soa_1[0] = Uh_aos[3]; U_soa_1[1] = Uh_aos[4]; U_soa_1[2] = Uh_aos[5];
                U_soa_2[0] = Uh_aos[6]; U_soa_2[1] = Uh_aos[7]; U_soa_2[2] = Uh_aos[8]; U_soa_2[3] = Uh_aos[9];
            }
        }
        else if (num_stencil == 27) {
            assert(lnz == 13 && rnz == 14);
            // L:(0,1,2) (3,4,5) (6,7,8) (9,10,11,12), U:(0,1,2) (3,4,5) (6,7,8,9) (10,11,12,13) 其中U(0)是主对角元
            for (idx_t e = 0; e < tot_elems; e++) {
                const calc_t* Lh_aos = L_high + e * lnz,
                            * Uh_aos = U_high + e * rnz;
                data_t  *L_soa_0 = L + 0 * tot_elems + e * 3,
                        *L_soa_1 = L + 3 * tot_elems + e * 3,
                        *L_soa_2 = L + 6 * tot_elems + e * 3,
                        *L_soa_3 = L + 9 * tot_elems + e * 4,
                        *U_soa_0 = U + 0 * tot_elems + e * 3,
                        *U_soa_1 = U + 3 * tot_elems + e * 3,
                        *U_soa_2 = U + 6 * tot_elems + e * 4,
                        *U_soa_3 = U + 10* tot_elems + e * 4;
                L_soa_0[0] = Lh_aos[0]; L_soa_0[1] = Lh_aos[1]; L_soa_0[2] = Lh_aos[2];
                L_soa_1[0] = Lh_aos[3]; L_soa_1[1] = Lh_aos[4]; L_soa_1[2] = Lh_aos[5];
                L_soa_2[0] = Lh_aos[6]; L_soa_2[1] = Lh_aos[7]; L_soa_2[2] = Lh_aos[8];
                L_soa_3[0] = Lh_aos[9]; L_soa_3[1] = Lh_aos[10];L_soa_3[2] = Lh_aos[11];L_soa_3[3] = Lh_aos[12];

                U_soa_0[0] = Uh_aos[0]; U_soa_0[1] = Uh_aos[1]; U_soa_0[2] = Uh_aos[2];
                U_soa_1[0] = Uh_aos[3]; U_soa_1[1] = Uh_aos[4]; U_soa_1[2] = Uh_aos[5];
                U_soa_2[0] = Uh_aos[6]; U_soa_2[1] = Uh_aos[7]; U_soa_2[2] = Uh_aos[8]; U_soa_2[3] = Uh_aos[9];
                U_soa_3[0] = Uh_aos[10];U_soa_3[1] = Uh_aos[11];U_soa_3[2] = Uh_aos[12];U_soa_3[3] = Uh_aos[13];
            }
        }
        // #pragma omp parallel for schedule(static)
        // for (idx_t e = 0; e < tot_elems; e++) {
        //     for (idx_t d = 0; d < lnz; d++)
        //         L[d * tot_elems + e] = L_high[e * lnz + d];
        // }
        // #pragma omp parallel for schedule(static)
        // for (idx_t e = 0; e < tot_elems; e++) {
        //     for (idx_t d = 0; d < rnz; d++)
        //         U[d * tot_elems + e] = U_high[e * rnz + d];
        // }
        delete L_high;
        delete U_high;
    }
    setup_called = true;
}

template<typename idx_t, typename data_t, typename calc_t>
void BlockILU<idx_t, data_t, calc_t>::Mult(const par_structVector<idx_t, calc_t> & input, par_structVector<idx_t, calc_t> & output) const
{
    CHECK_LOCAL_HALO(*(input.local_vector),  *(output.local_vector));// 检查相容性
    assert(outer_dim && midle_dim && inner_dim);

    int my_pid; MPI_Comm_rank(input.comm_pkg->cart_comm, &my_pid);
    
    if (this->zero_guess) {
        const seq_structVector<idx_t, calc_t> & b = *(input.local_vector);
        seq_structVector<idx_t, calc_t> & x = *(output.local_vector);
        
        input.update_halo();// Additive Schwartz要求右端项填充边界
        // s = L^{-1}*b

        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t real_j = j - jbeg, real_i = i - ibeg;
            const calc_t * src_ptr = b.data + j * b.slice_ki_size + i * b.slice_k_size + kbeg;
            calc_t * dst_ptr = buf + (real_j * midle_dim + real_i) * inner_dim;// 不需要加kbeg
            for (idx_t k = 0; k < inner_dim; k++)// 拷贝连续的一柱
                dst_ptr[k] = src_ptr[k];
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
#ifdef PROFILE
        double t, mint, maxt;
        int my_pid; MPI_Comm_rank(input.comm_pkg->cart_comm, &my_pid);
        int num_procs; MPI_Comm_size(input.comm_pkg->cart_comm, &num_procs);
        MPI_Barrier(input.comm_pkg->cart_comm);
        t = wall_time();
#endif
        // struct_trsv_forward_3d (L, buf, tmp, outer_dim, midle_dim, inner_dim, (num_stencil - 1)>>1, stencil_offset); static_assert(sizeof(data_t) == sizeof(calc_t));
        struct_sptrsv_3d_forward_frame_hardCode (L, buf, tmp, outer_dim, midle_dim, inner_dim, lnz, stencil_offset);
#ifdef PROFILE
        t = wall_time() - t;
        MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, input.comm_pkg->cart_comm);
        MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, input.comm_pkg->cart_comm);
        if (my_pid == 0) {
            double bytes = (outer_dim + 2) * (midle_dim + 2) * (inner_dim + 2) * 2 * sizeof(calc_t);// 向量
            bytes += outer_dim * midle_dim * inner_dim * lnz * sizeof(data_t);// 矩阵
            bytes = bytes * num_procs / (1024*1024*1024);// total GB
            printf("BILU-F data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), lnz, bytes, mint, maxt, bytes/maxt, bytes/mint);
        }
#endif
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
#ifdef PROFILE
        MPI_Barrier(input.comm_pkg->cart_comm);
        t = wall_time();
#endif
        // x = U^{-1}*s = U^{-1}*L^{-1}*b
        // struct_trsv_backward_3d(U, tmp, buf, outer_dim, midle_dim, inner_dim, (num_stencil + 1)>>1, stencil_offset + 3 * ((num_stencil - 1)>>1)); static_assert(sizeof(data_t) == sizeof(calc_t));
        struct_sptrsv_3d_backward_frame_hardCode(U, tmp, buf, outer_dim, midle_dim, inner_dim, rnz, stencil_offset + lnz*3);
#ifdef PROFILE
        t = wall_time() - t;
        MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, input.comm_pkg->cart_comm);
        MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, input.comm_pkg->cart_comm);
        if (my_pid == 0) {
            double bytes = (outer_dim + 2) * (midle_dim + 2) * (inner_dim + 2) * 2 * sizeof(calc_t);// 向量
            bytes += outer_dim * midle_dim * inner_dim * rnz * sizeof(data_t);// 矩阵
            bytes = bytes * num_procs / (1024*1024*1024);// total GB
            printf("BILU-B data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), rnz, bytes, mint, maxt, bytes/maxt, bytes/mint);
        }
#endif
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
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t real_j = j - jbeg, real_i = i - ibeg;
            const calc_t * src_ptr = buf + (real_j * midle_dim + real_i) * inner_dim;// 不需要加kbeg
            calc_t * dst_ptr = x.data + j * x.slice_ki_size + i * x.slice_k_size + kbeg;
            for (idx_t k = 0; k < inner_dim; k++)// 拷贝连续的一柱
                dst_ptr[k] = src_ptr[k];
        }

        // x = weight*x = weight*U^{-1}*L^{-1}*b
        if (this->weight != 1.0) vec_mul_by_scalar(this->weight, output, output);
    } 
    else {
        par_structVector<idx_t, calc_t> resi(input), error(output);

        // resi = A*x
        this->oper->Mult(output, resi, false);
        
        vec_add(input, -1.0, resi, resi);
        resi.update_halo();// Additive Schwartz要求右端项填充边界

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

        // s = L^{-1}*resi = L^{-1}*(b - A*x)
        // struct_trsv_forward_3d (L, buf, tmp, outer_dim, midle_dim, inner_dim, (num_stencil - 1)>>1, stencil_offset); static_assert(sizeof(data_t) == sizeof(calc_t));
        struct_sptrsv_3d_forward_frame_hardCode (L, buf, tmp, outer_dim, midle_dim, inner_dim, lnz, stencil_offset);
        // e = U^{-1}*s = U^{-1}*L^{-1}*(b - A*x)

        // struct_trsv_backward_3d(U, tmp, buf, outer_dim, midle_dim, inner_dim, (num_stencil + 1)>>1, stencil_offset + 3 * ((num_stencil - 1)>>1)); static_assert(sizeof(data_t) == sizeof(calc_t));
        struct_sptrsv_3d_backward_frame_hardCode(U, tmp, buf, outer_dim, midle_dim, inner_dim, rnz, stencil_offset + lnz*3);

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
        
        // x = x + w*e = x + w*U^{-1}*L^{-1}*(b - A*x)
        vec_add(output, this->weight, error, output);
    }
}

#endif