#ifndef SMG_ITER_SOLVER_HPP
#define SMG_ITER_SOLVER_HPP

#include "../utils/operator.hpp"
#include <memory>

typedef enum {PREC, OPER, AXPY, DOT, NUM_KRYLOV_RECORD} KRYLOV_RECORD_TYPES; 

// 虚基类IterativeSolver，支持两种精度
// 继承自虚基类->Solver->Operator，重写了SetOperator()，还需下一层子类重写Mult()
template<typename idx_t, typename ksp_t, typename pc_t>
class IterativeSolver : public Solver<idx_t, ksp_t, ksp_t, ksp_t> {
public:
    // oper(外迭代的算子/矩阵)，可以和prec(预条件子)采用不一样的精度
    const Operator<idx_t, ksp_t, ksp_t> *oper = nullptr;
    bool scaled = false;
    par_structMatrix<idx_t, ksp_t, ksp_t> *A_scaled = nullptr;
    seq_structVector<idx_t, ksp_t> *inv_sqrtD = nullptr;
    // 预条件子的存储采用低精度，计算采用高精度
    Solver<idx_t, pc_t, ksp_t, ksp_t> *prec = nullptr;

    int max_iter = 10, print_level = -1;
    double rel_tol = 0.0, abs_tol = 0.0;// 用高精度的收敛判断

    // stats
    mutable int final_iter = 0, converged = 0;
    mutable double final_norm;
    mutable double part_times[NUM_KRYLOV_RECORD];

    IterativeSolver() : Solver<idx_t, ksp_t, ksp_t, ksp_t>() {   }
    ~IterativeSolver() {
        if (scaled) {
            assert(A_scaled != nullptr);
            delete A_scaled; A_scaled = nullptr;
            assert(inv_sqrtD != nullptr);
            delete inv_sqrtD; inv_sqrtD = nullptr;
        }
    }

    void SetRelTol(double rtol) { rel_tol = rtol; }
    void SetAbsTol(double atol) { abs_tol = atol; }
    void SetMaxIter(int max_it) { max_iter = max_it; }
    void SetPrintLevel(int print_lvl) { print_level = print_level; }

    int GetNumIterations() const { return final_iter; }
    int GetConverged() const { return converged; }
    double GetFinalNorm() const { return final_norm; }

    void scale_oper(const ksp_t scaled_diag);
    void scale_vec(par_structVector<idx_t, ksp_t> & par_vec) const;

    /// This should be called before SetOperator
    virtual void SetPreconditioner(Solver<idx_t, pc_t, ksp_t, ksp_t> & pr) {
        prec = & pr;
        prec->zero_guess = true;// 预条件一般可以用0初值
    }

    /// Also calls SetOperator for the preconditioner if there is one
    virtual void SetOperator(const Operator<idx_t, ksp_t, ksp_t> & op) {
        oper = & op;
        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        // 注意，当ksp_t与pc_t的实例化类型不同时，此处会编译报错，需要另外改写：注意是用高精度的完全setup之后再截断
        if (prec) {
            int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
            if (my_pid == 0) printf("IterSolver: SetOperator\n");
            if (scaled) {
                if (my_pid == 0) printf("IterSolver: Scaled Operator used to Setup Preconditioner!\n");
                scale_oper(100.0);
                prec->SetOperator(*A_scaled);
            } else {
                prec->SetOperator(*oper);
            }
        }
    }

    virtual void truncate() {
        if (this->prec) {
            int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
            if (my_pid == 0) printf("IterSolver: Truncate\n");
            this->prec->truncate();
        }
    }

    // 迭代法里的点积（默认采用双精度）
    double Dot(const par_structVector<idx_t, ksp_t> & x, const par_structVector<idx_t, ksp_t> & y) const {
        return vec_dot<idx_t, ksp_t, double>(x, y);
    }
    // 迭代法里的范数（默认采用双精度）
    double Norm(const par_structVector<idx_t, ksp_t> & x) const {
        return sqrt(Dot(x, x));
    }

protected:
    virtual void Mult(const par_structVector<idx_t, ksp_t> & input, 
                            par_structVector<idx_t, ksp_t> & output) const = 0;
public:
    // 所有具体的迭代方法的唯一的公共的外部接口
    void Mult(const par_structVector<idx_t, ksp_t> & input, 
                    par_structVector<idx_t, ksp_t> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;// 迭代法的初值是否为0
        this->Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }
};

template<typename idx_t, typename ksp_t, typename pc_t>
void IterativeSolver<idx_t, ksp_t, pc_t>::scale_oper(const ksp_t scaled_diag)
{
    assert(oper != nullptr);
    assert(scaled == true);
    const par_structMatrix<idx_t, ksp_t, ksp_t> * A = (par_structMatrix<idx_t, ksp_t, ksp_t>*)oper;
    inv_sqrtD = new seq_structVector<idx_t, ksp_t>(A->local_matrix->local_x, A->local_matrix->local_y, A->local_matrix->local_z,
        A->local_matrix->halo_x, A->local_matrix->halo_y, A->local_matrix->halo_z);
    // 确定各维上是否是边界
    const bool x_lbdr = A->comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = A->comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = A->comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = A->comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = A->comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = A->comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    idx_t jbeg = (y_lbdr ? inv_sqrtD->halo_y : 0), jend = inv_sqrtD->halo_y + inv_sqrtD->local_y + (y_ubdr ? 0 : inv_sqrtD->halo_y);
    idx_t ibeg = (x_lbdr ? inv_sqrtD->halo_x : 0), iend = inv_sqrtD->halo_x + inv_sqrtD->local_x + (x_ubdr ? 0 : inv_sqrtD->halo_x);
    idx_t kbeg = (z_lbdr ? inv_sqrtD->halo_z : 0), kend = inv_sqrtD->halo_z + inv_sqrtD->local_z + (z_ubdr ? 0 : inv_sqrtD->halo_z);
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0) printf("scaled => diagonal as %.2e\n", scaled_diag);
    // printf(" proc %d [%d,%d) x [%d,%d) x [%d,%d)\n", my_pid, ibeg, iend, jbeg, jend, kbeg, kend);

    CHECK_LOCAL_HALO(*inv_sqrtD, *(A->local_matrix));
    assert(A->num_diag == 7);
    const idx_t vec_ki_size = inv_sqrtD->slice_ki_size, vec_k_size = inv_sqrtD->slice_k_size;
    const idx_t slice_dki_size = A->local_matrix->slice_dki_size, slice_dk_size = A->local_matrix->slice_dk_size;
    const ksp_t sqrt_scaled_diag = sqrt(scaled_diag);
    // 提取对角线元素，开方并取倒数
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        ksp_t tmp = A->local_matrix->data[j * slice_dki_size + i * slice_dk_size + k * A->num_diag + 3];
        assert(tmp > 0.0);
        inv_sqrtD->data[j * vec_ki_size + i * vec_k_size + k] = sqrt_scaled_diag / sqrt(tmp);
    }

    // 矩阵scaling
    A_scaled = new par_structMatrix<idx_t, ksp_t, ksp_t>(*A);
    jbeg = A->local_matrix->halo_y; jend = jbeg + A->local_matrix->local_y;
    ibeg = A->local_matrix->halo_x; iend = ibeg + A->local_matrix->local_x;
    kbeg = A->local_matrix->halo_z; kend = kbeg + A->local_matrix->local_z;
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        idx_t offset = j * slice_dki_size + i * slice_dk_size + k * A->num_diag;
        ksp_t   * src_ptr = A->local_matrix->data + offset,
                * dst_ptr = A_scaled->local_matrix->data + offset;
        for (int d = 0; d < A->num_diag; d++) {
            ksp_t tmp = src_ptr[d];
            int ngb_j = j + stencil_offset_3d7[d * 3 + 0];
            int ngb_i = i + stencil_offset_3d7[d * 3 + 1];
            int ngb_k = k + stencil_offset_3d7[d * 3 + 2];
            if      (x_lbdr && ngb_i <  ibeg) assert(tmp == 0.0);
            else if (x_ubdr && ngb_i >= iend) assert(tmp == 0.0);
            else if (y_lbdr && ngb_j <  jbeg) assert(tmp == 0.0);
            else if (y_ubdr && ngb_j >= jend) assert(tmp == 0.0);
            else if (z_lbdr && ngb_k <  kbeg) assert(tmp == 0.0);
            else if (z_ubdr && ngb_k >= kend) assert(tmp == 0.0);
            else {
                ksp_t my_inv_sqrtD_val = inv_sqrtD->data[    j * vec_ki_size +     i * vec_k_size +     k];
                ksp_t ngb_inv_sqrtD_val= inv_sqrtD->data[ngb_j * vec_ki_size + ngb_i * vec_k_size + ngb_k];
                // if (my_inv_sqrtD_val <= 0.0) {
                //     printf(" (%d,%d,%d) my_inv_sqrtD_val %.5e\n", i, j, k, my_inv_sqrtD_val);
                // }
                assert(my_inv_sqrtD_val > 0.0);
                assert(ngb_inv_sqrtD_val > 0.0);
                tmp *= my_inv_sqrtD_val * ngb_inv_sqrtD_val;// 除以自己的对角线 和 邻居的对角线
            }
            // 赋回
            dst_ptr[d] = tmp;
        }
    }
    A_scaled->update_halo();
    assert(A_scaled->check_Dirichlet());
    assert(A_scaled->check_scaling(scaled_diag));
}

template<typename idx_t, typename ksp_t, typename pc_t>
void IterativeSolver<idx_t, ksp_t, pc_t>::scale_vec(par_structVector<idx_t, ksp_t> & par_vec) const
{
    assert(scaled == true);
    seq_vec_elemwise_mul(*(par_vec.local_vector), *inv_sqrtD);
}

#endif