#ifndef SMG_DENSELU_HPP
#define SMG_DENSELU_HPP

#include "../utils/par_struct_mat.hpp"
#include "../utils/LU.hpp"
#define USE_DENSE
#ifndef USE_DENSE
#ifdef __x86_64__
#include "mkl_types.h"
#include "mkl_pardiso.h"
#elif defined(__aarch64__)
// #include "hpdss.h"
#include "slu_ddefs.h"
#endif
#endif

typedef enum {DenseLU_3D7, DenseLU_3D27} DenseLU_type;

#ifndef USE_DENSE
template<typename idx_t, typename data_t, typename res_t>
class CSR_sparseMat {
public:
    idx_t nrows = 0;
    idx_t * row_ptr = nullptr;
    idx_t * col_idx = nullptr;
    data_t* vals = nullptr;

#ifdef __aarch64__
    // hpdss::AnalyseConfig config;
    // hpdss::Handler handler;

    char           equed[1];
    SuperMatrix    A, L, U;
    SuperMatrix    B, X;
    NCformat       *Astore;
    NCformat       *Ustore;
    SCformat       *Lstore;
    GlobalLU_t	   Glu; // facilitate multiple factorizations with SamePattern_SameRowPerm       
    double         *a;
    int            *asub, *xa;
    int            *perm_c; // column permutation vector
    int            *perm_r; // row permutations from partial pivoting
    int            *etree;
    void           *work = NULL;
    int            info, lwork = 0, nrhs, ldx;
    double         *b_buf, *x_buf;
    double         *R, *C;
    double         *ferr, *berr;
    double         u, rpg, rcond;
    mem_usage_t    mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
#elif defined(__x86_64__)
    MKL_INT mtype = 1;
    MKL_INT nrhs = 1;
    void * pt[64];// internal solver memory pointer
    MKL_INT iparm[64];// pardiso control parameters
    MKL_INT maxfct, mnum, phase, error, msglvl;
    double ddum;// Double dummy
    MKL_INT idum;// Integer dummy.
    res_t * b_buf = nullptr;
#endif
    
    CSR_sparseMat(idx_t m, idx_t * Ai, idx_t * Aj, data_t * Av):
        nrows(m), row_ptr(Ai), col_idx(Aj), vals(Av) {  }
    ~CSR_sparseMat() { 
#ifdef __aarch64__
        // if (row_ptr) delete row_ptr;
        // if (col_idx) delete col_idx;
        // if (vals   ) delete vals;
        SUPERLU_FREE (b_buf);
        SUPERLU_FREE (x_buf);
        SUPERLU_FREE (etree);
        SUPERLU_FREE (perm_r);
        SUPERLU_FREE (perm_c);
        SUPERLU_FREE (R);
        SUPERLU_FREE (C);
        SUPERLU_FREE (ferr);
        SUPERLU_FREE (berr);
        Destroy_CompRow_Matrix(&A);
        Destroy_SuperMatrix_Store(&B);
        Destroy_SuperMatrix_Store(&X);
        if ( lwork == 0 ) {
            Destroy_SuperNode_Matrix(&L);
            Destroy_CompRow_Matrix(&U);
        } else if ( lwork > 0 ) {
            SUPERLU_FREE(work);
        }
#elif defined(__x86_64__)
        if (row_ptr) delete row_ptr;
        if (col_idx) delete col_idx;
        if (vals   ) delete vals;
        if (b_buf) delete b_buf;
        phase = -1;// Release internal memory.
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                &nrows, &ddum, row_ptr, col_idx, &idum, &nrhs,
                iparm, &msglvl, &ddum, &ddum, &error);
#endif
    }
    double decomp() {
        assert(row_ptr && col_idx && vals);
        double t = 0.0;
#ifdef __aarch64__
        // Call Wang Xinliang's package here
        // int rtn;
        // hpdss::Analyse
        // rtn = handler.AnalyseFromCSR(nrows, row_ptr, col_idx, vals, &config);

        superlu_options_t options;
        set_default_options(&options);
        dCreate_CompRow_Matrix(&A, nrows, nrows, row_ptr[nrows], 
            vals, col_idx, row_ptr, SLU_NR, SLU_D, SLU_GE);
        Astore = (NCformat*) A.Store;

        const idx_t nrhs = 1;
        b_buf = new res_t [nrows];
        x_buf = new res_t [nrows];
        dCreate_Dense_Matrix(&B, nrows, nrhs, b_buf, nrows, SLU_DN, SLU_D, SLU_GE);
        dCreate_Dense_Matrix(&X, nrows, nrhs, x_buf, nrows, SLU_DN, SLU_D, SLU_GE);
        
        etree = new idx_t [nrows];
        perm_r= new idx_t [nrows];
        perm_c= new idx_t [nrows];
        if ( !(R = (double *) SUPERLU_MALLOC(A.nrow * sizeof(double))) ) 
            ABORT("SUPERLU_MALLOC fails for R[].");
        if ( !(C = (double *) SUPERLU_MALLOC(A.ncol * sizeof(double))) )
            ABORT("SUPERLU_MALLOC fails for C[].");
        if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
            ABORT("SUPERLU_MALLOC fails for ferr[].");
        if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) ) 
            ABORT("SUPERLU_MALLOC fails for berr[].");

        StatInit(&stat);
        B.ncol = X.ncol = 0;  // Indicate not to solve the system
        dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
        
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) {
            printf("LU factorization: dgssvx() returns info %d\n", info);
            if ( info == 0 || info == nrows+1 ) {

                if ( options.PivotGrowth ) printf("Recip. pivot growth = %e\n", rpg);
                if ( options.ConditionNumber ) printf("Recip. condition number = %e\n", rcond);
                Lstore = (SCformat *) L.Store;
                Ustore = (NCformat *) U.Store;
                printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
                printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
                printf("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz - nrows);
                printf("FILL ratio = %.1f\n", (float)(Lstore->nnz + Ustore->nnz - nrows)/row_ptr[nrows]);
                printf("L\\U MB %.3f\ttotal MB needed %.3f\n",
                    mem_usage.for_lu/1e6, mem_usage.total_needed/1e6);
                fflush(stdout);
            } else if ( info > 0 && lwork == -1 ) {
                printf("** Estimated memory: %d bytes\n", info - nrows);
            }
        }
        StatFree(&stat);
        return t;
#elif defined(__x86_64__)
        for (MKL_INT i = 0; i < 64; i++) iparm[i] = 0;
        iparm[0] = 1;// no solver default
        iparm[1] = 2;// Fill-in reordering from METIS
        iparm[3] = 0;// No iterative-direct algorithm
        iparm[4] = 0;// No user fill-in reducing permutation
        iparm[5] = 0;// Write solution into x
        iparm[7] = 0;// Max numbers of iterative refinement steps
        iparm[9] = 13;// Perturb the pivot elements with 1E-13
        iparm[10] = 1;// Use nonsymmetric permutation and scaling MPS
        iparm[12] = 0;// Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy
        iparm[13] = 0;// Output: Number of perturbed pivots
        iparm[17] = -1;// Output: Number of nonzeros in the factor LU
        iparm[18] = -1;// Output: Mflops for LU factorization
        iparm[19] = 0;// Output: Numbers of CG Iterations
        iparm[34] = 1;// PARDISO use C-style indexing for ia and ja arrays */
        maxfct = 1;// Maximum number of numerical factorizations. */
        mnum = 1;// Which factorization to use.
        msglvl = 0;// Print statistical information in file */
        error = 0;// Initialize error flag
        for (MKL_INT i = 0; i < 64; i++) pt[i] = 0;

        phase = 11;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &nrows, vals, row_ptr, col_idx, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
        if (error != 0 ) {
            printf ("\nERROR during symbolic factorization: %d", error);
            MPI_Abort(MPI_COMM_WORLD, -20230113);
        }

        // 只记录数值分解的时间
        t -= wall_time();
        phase = 22;
        for (int i = 0; i < maxfct; i++)
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &nrows, vals, row_ptr, col_idx, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
        if (error != 0 ) {
            printf ("\nERROR during numerical factorization: %d", error);
            MPI_Abort(MPI_COMM_WORLD, -20230113);
        }
        t += wall_time();
        b_buf = new res_t [nrows];
        return t / maxfct;
#endif
    }
    void apply(const res_t * b, res_t * x) {
#ifdef __aarch64__
        const idx_t nrhs = 1;
        options.Fact = FACTORED; // Indicate the factored form of A is supplied.
        X.ncol = B.ncol = nrhs;  // Set the number of right-hand side

        // copy in
        double *rhs = (double*) ((DNformat*) B.Store)->nzval;
        for (idx_t i = 0; i < nrows; i++)
            rhs[i] = b[i];
        StatInit(&stat);
        dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
        StatFree(&stat);
        // copy back
        double *sol = (double*) ((DNformat*) X.Store)->nzval;
        for (idx_t i = 0; i < nrows; i++)
            x[i] = sol[i];
#elif defined(__x86_64__)
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < nrows; i++)
            b_buf[i] = b[i];
        phase = 33;
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &nrows, vals, row_ptr, col_idx, &idum, &nrhs, iparm, &msglvl, b_buf, x, &error);
        if (error != 0 ) {
            printf ("\nERROR during solution: %d", error);
            MPI_Abort(MPI_COMM_WORLD, -20230113);
        }
#endif
    }
    void fprint_COO(const char * filename) const {
        FILE * fp = fopen(filename, "w+");
        for (idx_t i = 0; i < nrows; i++) {
            idx_t pbeg = row_ptr[i], pend = row_ptr[i+1];
            for (idx_t p = pbeg; p < pend; p++)
                fprintf(fp, "%d %d %.5e\n", i, col_idx[p], vals[p]);
        }
        fclose(fp);
    }
};
#endif

// 完全LU类中的data_t类型仅做摆设，实际存储和计算都用calc_t
template<typename idx_t, typename data_t, typename setup_t, typename calc_t>
class DenseLU final : public Solver<idx_t, data_t, setup_t, calc_t> {
public:
    DenseLU_type type;
    idx_t num_stencil = 0;// 未初始化
    const idx_t * stencil_offset = nullptr;
    bool setup_called = false;
    double setup_time = 0.0;

    // operator (often as matrix-A)
    const Operator<idx_t, setup_t, setup_t> * oper = nullptr;
    
    idx_t global_dof;
    calc_t * u_data = nullptr, * l_data = nullptr;
    calc_t * dense_x = nullptr, * dense_b = nullptr;// 用于前代回代的数据
    idx_t * sendrecv_cnt = nullptr, * displs = nullptr;
    MPI_Datatype vec_recv_type = MPI_DATATYPE_NULL, mat_recv_type = MPI_DATATYPE_NULL;// 接收者（只有0号进程需要）的数据类型
    MPI_Datatype mat_send_type = MPI_DATATYPE_NULL, vec_send_type = MPI_DATATYPE_NULL;// 发送者（各个进程都需要）的数据类型

#ifndef USE_DENSE
    CSR_sparseMat<idx_t, calc_t, calc_t> * glbA_csr = nullptr;
#endif

    DenseLU(DenseLU_type type) : Solver<idx_t, data_t, setup_t, calc_t>(), type(type) {
        if (type == DenseLU_3D7) {
            num_stencil = 7;
            stencil_offset = stencil_offset_3d7;
        }
        else if (type == DenseLU_3D27) {
            num_stencil = 27;
            stencil_offset = stencil_offset_3d27;
        } 
        else {
            printf("Not supported ilu type %d! Only DenseLU_3d7 or _3d27 available!\n", type);
            MPI_Abort(MPI_COMM_WORLD, -99);
        }
    }
    ~DenseLU() {
        if (u_data != nullptr) {delete u_data; u_data = nullptr;}
        if (l_data != nullptr) {delete l_data; l_data = nullptr;}
        if (dense_b!= nullptr) {delete dense_b; dense_b = nullptr;}
        if (dense_x!= nullptr) {delete dense_x; dense_x = nullptr;}
        if (mat_recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&mat_recv_type);
        if (mat_send_type != MPI_DATATYPE_NULL) MPI_Type_free(&mat_send_type);
        if (vec_recv_type != MPI_DATATYPE_NULL) MPI_Type_free(&vec_recv_type);
        if (vec_send_type != MPI_DATATYPE_NULL) MPI_Type_free(&vec_send_type);
        if (sendrecv_cnt != nullptr) {delete sendrecv_cnt; sendrecv_cnt = nullptr;}
        if (displs != nullptr) {delete displs; displs = nullptr;}
#ifndef USE_DENSE
        if (glbA_csr != nullptr) {delete glbA_csr; glbA_csr = nullptr;}
#endif
    }
    void SetOperator(const Operator<idx_t, setup_t, setup_t> & op) {
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
        if (my_pid == 0) printf("Warning: DenseLU NOT trunc!!!\n");
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

template<typename idx_t, typename data_t, typename setup_t, typename calc_t>
void DenseLU<idx_t, data_t, setup_t, calc_t>::Setup()
{
    if (setup_called) return ;
    setup_time = -wall_time();

    assert(this->oper != nullptr);
    // assert matrix has updated halo to prepare data 强制类型转换
    const par_structMatrix<idx_t, setup_t, setup_t> & par_A = *((par_structMatrix<idx_t, setup_t, setup_t>*)(this->oper));
    const seq_structMatrix<idx_t, setup_t, setup_t> & seq_A = *(par_A.local_matrix);// 外层问题的A矩阵
    assert(seq_A.num_diag == num_stencil);

    // 先确定谁来做计算：0号进程来算
    int my_pid;
    MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
    int proc_ndim = sizeof(par_A.comm_pkg->cart_ids) / sizeof(int); assert(proc_ndim == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(par_A.comm_pkg->cart_comm, proc_ndim, num_procs, periods, coords);
    // 假定0号进程就是进程网格中的最角落的进程
    if (my_pid == 0) assert(coords[0] == 0 && coords[1] == 0 && coords[2] == 0);

    const idx_t gx = seq_A.local_x * num_procs[1],
                gy = seq_A.local_y * num_procs[0],
                gz = seq_A.local_z * num_procs[2];
    global_dof = gx * gy * gz;
    // const idx_t global_nnz = seq_A.num_diag * global_dof;
    const float sparsity = (float) seq_A.num_diag / (float) global_dof;//global_nnz / (global_dof * global_dof);

    if (global_dof >= 5000 && sparsity < 0.01) {
        if (my_pid == 0) printf("\033[1;32mWARNING! Too large matrix for direct Gauss-Elim method!\033[0m\n");
        // MPI_Abort(MPI_COMM_WORLD, -999);
    }

    dense_x = new calc_t[global_dof];
    dense_b = new calc_t[global_dof];
    setup_t * buf = new setup_t[global_dof * num_stencil];// 接收缓冲区：全局的稀疏结构化矩阵

    // 执行LU分解，并存储
    idx_t sizes[4], subsizes[4], starts[4];
    // 发送方的数据类型
    sizes[0] = seq_A.local_y + seq_A.halo_y * 2;    subsizes[0] = seq_A.local_y;    starts[0] = seq_A.halo_y;
    sizes[1] = seq_A.local_x + seq_A.halo_x * 2;    subsizes[1] = seq_A.local_x;    starts[1] = seq_A.halo_x;
    sizes[2] = seq_A.local_z + seq_A.halo_z * 2;    subsizes[2] = seq_A.local_z;    starts[2] = seq_A.halo_z;
    sizes[3] = seq_A.num_diag;                      subsizes[3] = sizes[3];         starts[3] = 0;
    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, par_A.comm_pkg->mpi_scalar_type, &mat_send_type);
    MPI_Type_commit(&mat_send_type);
    assert(sizeof(calc_t) == 8 || sizeof(calc_t) == 4);
    assert(sizeof(setup_t) == 8 || sizeof(setup_t) == 4);
    MPI_Datatype vec_scalar_type = (sizeof(calc_t) == 8) ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, vec_scalar_type, &vec_send_type);
    MPI_Type_commit(&vec_send_type);

    // 接收者（0号进程）的数据类型
    MPI_Datatype tmp_type = MPI_DATATYPE_NULL;
    sizes[0] = gy; sizes[1] = gx; sizes[2] = gz;// sizes[3] = seq_A.num_diag;
    // subsize[]不用改
    // starts[]数组改成从头开始，每次都手动指定displs
    starts[0] = starts[1] = starts[2] = 0;// starts[3] = 0
    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, par_A.comm_pkg->mpi_scalar_type, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, subsizes[2] * sizes[3] * sizeof(setup_t), &mat_recv_type);
    MPI_Type_commit(&mat_recv_type);
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, vec_scalar_type, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, subsizes[2]            * sizeof(calc_t), &vec_recv_type);
    MPI_Type_commit(&vec_recv_type);

    const idx_t tot_procs = num_procs[0] * num_procs[1] * num_procs[2];// py * px * pz
    sendrecv_cnt = new idx_t [tot_procs];
    displs       = new idx_t [tot_procs]; 

    for (idx_t p = 0; p < tot_procs; p++)
        sendrecv_cnt[p] = 1;
    // 这个位移是以resize之后的一个subarray（数组整体）的位移去记的，刚才已经将extent改为了subsizes[2] * sizes[3] * sizeof(data_t)
    for (idx_t j = 0; j < num_procs[0]; j++)
        displs[ j * num_procs[1]      * num_procs[2]    ] = j * subsizes[0] * sizes[1] * num_procs[2];// j * stride_y * gx * pz
    for (idx_t j = 0; j < num_procs[0]; j++)
    for (idx_t i = 1; i < num_procs[1]; i++)
        displs[(j * num_procs[1] + i) * num_procs[2]    ] = displs[(j * num_procs[1] + i - 1) * num_procs[2]] + subsizes[1] * num_procs[2];// += stride_x * pz
    for (idx_t j = 0; j < num_procs[0]; j++)
    for (idx_t i = 0; i < num_procs[1]; i++)
    for (idx_t k = 1; k < num_procs[2]; k++)
        displs[(j * num_procs[1] + i) * num_procs[2] + k] = displs[(j * num_procs[1] + i) * num_procs[2] + k - 1] + 1;

    MPI_Allgatherv(seq_A.data, 1, mat_send_type, 
                buf, sendrecv_cnt, displs, mat_recv_type, par_A.comm_pkg->cart_comm);

#ifdef DEBUG
        for (idx_t i = 0; i < global_dof; i++) {
            printf(" idx %4d ", i);
            for (idx_t v = 0; v < seq_A.num_diag; v++)
                printf(" %.6e", buf[i * num_stencil + v]);
            printf("\n");
        }
#endif

#ifndef USE_DENSE
    // 将结构化排布的稀疏矩阵转成CSR
    idx_t * row_ptr = new idx_t [global_dof+1];
    idx_t * col_idx = new idx_t [global_dof * num_stencil];// 按照最大的可能上限开辟
    calc_t * vals    = new calc_t[global_dof * num_stencil];
    int nnz_cnt = 0;
    row_ptr[0] = 0;// init
    for (idx_t j = 0; j < gy; j++)
    for (idx_t i = 0; i < gx; i++)
    for (idx_t k = 0; k < gz; k++) {
        idx_t row = (j * gx + i) * gz + k;
        for (idx_t d = 0; d < num_stencil; d++) {
            idx_t ngb_j = j + stencil_offset[d * 3    ];
            idx_t ngb_i = i + stencil_offset[d * 3 + 1];
            idx_t ngb_k = k + stencil_offset[d * 3 + 2];
            if (ngb_j >= 0 && ngb_j < gy && 
                ngb_i >= 0 && ngb_i < gx && 
                ngb_k >= 0 && ngb_k < gz) {
                idx_t col = (ngb_j * gx + ngb_i) * gz + ngb_k;
                col_idx[nnz_cnt] = col;
                vals   [nnz_cnt] = buf[row * num_stencil + d];
                nnz_cnt++;
            }
        }
        row_ptr[row+1] = nnz_cnt;
    }
    glbA_csr = new CSR_sparseMat<idx_t, calc_t, calc_t>(global_dof, row_ptr, col_idx, vals);
    // glbA_csr->fprint_COO("sparseA.txt");
    setup_time += wall_time();
    double decomp_time = glbA_csr->decomp();
    setup_time += decomp_time;
    setup_time -= wall_time();
#else
    setup_t * dense_A = new setup_t[global_dof * global_dof];// 用于分解的稠密A矩阵
    setup_t * L_high = new setup_t [global_dof * (global_dof - 1) / 2];
    setup_t * U_high = new setup_t [global_dof * (global_dof + 1) / 2];
        // 将结构化排布的稀疏矩阵稠密化
        #pragma omp parallel for schedule(static)
        for (idx_t p = 0; p < global_dof * global_dof; p++)
            dense_A[p] = 0.0;
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t j = 0; j < gy; j++)
        for (idx_t i = 0; i < gx; i++)
        for (idx_t k = 0; k < gz; k++) {
            idx_t row = (j * gx + i) * gz + k;
            for (idx_t d = 0; d < num_stencil; d++) {
                idx_t ngb_j = j + stencil_offset[d * 3 + 0];
                idx_t ngb_i = i + stencil_offset[d * 3 + 1];
                idx_t ngb_k = k + stencil_offset[d * 3 + 2];
                if (ngb_j >= 0 && ngb_j < gy && 
                    ngb_i >= 0 && ngb_i < gx && 
                    ngb_k >= 0 && ngb_k < gz) {
                    idx_t col = (ngb_j * gx + ngb_i) * gz + ngb_k;
                    dense_A[row * global_dof + col] = (calc_t) buf[row * num_stencil + d];
                }
            }
        }
#ifdef DEBUG
        char filename[100];
        sprintf(filename, "denseA.txt.%d", my_pid);
        FILE * fp = fopen(filename, "w+");
        for (idx_t i = 0; i < global_dof; i++) {
            for (idx_t j = 0; j < global_dof; j++)
                fprintf(fp, "%.5e ", dense_A[i * global_dof + j]);
            fprintf(fp, "\n");
        }
        fclose(fp);
#endif
        // 执行分解
        dense_LU_decomp(dense_A, L_high, U_high, global_dof, global_dof);
        delete dense_A;
    
    if constexpr (sizeof(calc_t) == sizeof(setup_t)) {
        l_data = L_high;
        u_data = U_high;
    } else {
        if (my_pid == 0) {
            printf("  \033[1;31mWarning\033[0m: LU::Setup() using setup_t of %ld bytes, but calc_t of %ld bytes\n",
                sizeof(setup_t), sizeof(calc_t));
        }
        idx_t tot_len;
        tot_len = global_dof * (global_dof - 1) / 2;
        l_data = new calc_t [tot_len];
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < tot_len; i++)
            l_data[i] = L_high[i];

        tot_len = global_dof * (global_dof + 1) / 2;
        u_data = new calc_t [tot_len];
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < tot_len; i++)
            u_data[i] = U_high[i];
        
        delete L_high;
        delete U_high;
    }
#endif
    delete buf;
    setup_called = true;
    setup_time += wall_time();
}

template<typename idx_t, typename data_t, typename setup_t, typename calc_t>
void DenseLU<idx_t, data_t, setup_t, calc_t>::Mult(const par_structVector<idx_t, calc_t> & input, par_structVector<idx_t, calc_t> & output) const
{
    CHECK_LOCAL_HALO( *(input.local_vector),  *(output.local_vector));// 检查相容性
    CHECK_OUTPUT_DIM(*this, input);// A * out = in
    CHECK_INPUT_DIM(*this, output);

    int my_pid; MPI_Comm_rank(input.comm_pkg->cart_comm, &my_pid);
    assert(input.global_size_x * input.global_size_y * input.global_size_z == global_dof);

    if (this->zero_guess) {
        const seq_structVector<idx_t, calc_t> & b = *(input.local_vector);
              seq_structVector<idx_t, calc_t> & x = *(output.local_vector);

#ifdef USE_DENSE
        MPI_Allgatherv(b.data, 1, vec_send_type, dense_x, sendrecv_cnt, displs, vec_recv_type, input.comm_pkg->cart_comm);
        dense_forward (l_data, dense_x, dense_b, global_dof, global_dof);// 前代
        dense_backward(u_data, dense_b, dense_x, global_dof, global_dof);// 回代
#else
        MPI_Allgatherv(b.data, 1, vec_send_type, dense_b, sendrecv_cnt, displs, vec_recv_type, input.comm_pkg->cart_comm);
        glbA_csr->apply(dense_b, dense_x);
#endif

        {// 从dense_x中拷回到向量中
            const idx_t ox = output.offset_x     , oy = output.offset_y     , oz = output.offset_z     ;
            const idx_t gx = output.global_size_x,                            gz = output.global_size_z;
            const idx_t lx = x.local_x           , ly = x.local_y           , lz = x.local_z           ;
            const idx_t hx = x.halo_x            , hy = x.halo_y            , hz = x.halo_z            ;
            const idx_t vec_ki_size = x.slice_ki_size, vec_k_size = x.slice_k_size;
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t j = 0; j < ly; j++)
            for (idx_t i = 0; i < lx; i++)
            for (idx_t k = 0; k < lz; k++) {
                x.data[(j + hy) * vec_ki_size + (i + hx) * vec_k_size + k + hz] = dense_x[((j + oy) * gx + (i + ox)) * gz + k + oz];
            }
        }

        if (this->weight != 1.0) vec_mul_by_scalar(this->weight, output, output);
    }
    else {
        assert(false);
        // 先计算一遍残差
        // par_structVector<idx_t, calc_t> resi(input), error(output);
        // this->oper->Mult(output, resi, false);
        // vec_add(input, -1.0, resi, resi);

        // MPI_Allgatherv(resi.local_vector->data, 1, vec_send_type, dense_x, sendrecv_cnt, displs, vec_recv_type, input.comm_pkg->cart_comm);

        //     dense_forward (l_data, dense_x, dense_b, global_dof, global_dof);// 前代
        //     dense_backward(u_data, dense_b, dense_x, global_dof, global_dof);// 回代

        // {// 从dense_x中拷回到向量中
        //     seq_structVector<idx_t, calc_t> & vec = *(error.local_vector);
        //     const idx_t ox = error.offset_x     , oy = error.offset_y, oz = error.offset_z     ;
        //     const idx_t gx = error.global_size_x,                      gz = error.global_size_z;
        //     const idx_t lx = vec.local_x        , ly = vec.local_y   , lz = vec.local_z        ;
        //     const idx_t hx = vec.halo_x         , hy = vec.halo_y    , hz = vec.halo_z         ;
        //     const idx_t vec_ki_size = vec.slice_ki_size, vec_k_size = vec.slice_k_size;
        //     for (idx_t j = 0; j < ly; j++)
        //     for (idx_t i = 0; i < lx; i++)
        //     for (idx_t k = 0; k < lz; k++) {
        //         vec.data[(j + hy) * vec_ki_size + (i + hx) * vec_k_size + k + hz] = dense_x[((j + oy) * gx + (i + ox)) * gz + k + oz];
        //     }
        // }
    
        // vec_add(output, this->weight, error, output);
    }
}


#endif