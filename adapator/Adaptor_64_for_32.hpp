#ifndef SMG_ADAPTOR_64_32_HPP
#define SMG_ADAPTOR_64_32_HPP

#include "../utils/par_struct_mat.hpp"
#include "../utils/operator.hpp"
#include "../Solver_ls.hpp"
/*  外部表现为calc_t=fp64, data_t=fp32的预条件子
    内部实现为calc_t=fp32, data_t=fp16的预条件子

    接受一个fp64的向量，在此截断为fp32的向量，再传给真正的“预条件”
    
*/

class Adaptor_64_for_32 : public Solver<int, float, double>
{
private:
    std::string prc_name;
    bool scale_before_setup = false;
    par_structMatrix<int, float, float> * A_trc = nullptr;// 将fp64截断成fp32的矩阵，用于给预条件setup
    par_structVector<int, float> * tmp_b = nullptr, * tmp_x = nullptr;
    Solver<int, __fp16, float> * real_prec = nullptr;
public:
    void SetOperator(const Operator<int, double, double> & op);
    Adaptor_64_for_32(char * namelist[], const int list_length, bool need_to_scale=false);
    ~Adaptor_64_for_32() {
        delete A_trc;
        delete tmp_b; delete tmp_x;
        delete real_prec;
    }
    void Mult(const par_structVector<int, double> & input, 
                    par_structVector<int, double> & output, bool use_zero_guess) const ;
    void truncate() {
        real_prec->truncate();
    }
};

Adaptor_64_for_32::Adaptor_64_for_32(char * namelist[], const int list_length, bool need_to_scale)
    : Solver<int, float, double>(), scale_before_setup(need_to_scale) {
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int cnt = 0;
    assert(list_length >= 1);
    prc_name = namelist[cnt++];
    if (my_pid == 0) {
        printf("\033[1;35m  ADAPTOR: truncate problem matrix from %ld to %ld\033[0m\n", sizeof(double), sizeof(float));
        for (int i = 0; i < list_length; i++) printf("%s ", namelist[i]);
        printf("\n");
    }
    if (strstr(prc_name.c_str(), "PGS")) {
        SCAN_TYPE type = SYMMETRIC;
        if      (strstr(prc_name.c_str(), "F")) type = FORWARD;
        else if (strstr(prc_name.c_str(), "B")) type = BACKWARD;
        else if (strstr(prc_name.c_str(), "S")) type = SYMMETRIC;
        if (my_pid == 0) printf("Adaptor:  using \033[1;35mpointwise-GS %d\033[0m as preconditioner\n", type);
        real_prec = new PointGS<int, __fp16, float>(type);
    } else if (prc_name == "GMG") {
        int num_discrete = atoi(namelist[cnt++]);
        int num_Galerkin = atoi(namelist[cnt++]);
        std::unordered_map<std::string, RELAX_TYPE> trans_smth;
        trans_smth["PGS"]= PGS;
        std::vector<RELAX_TYPE> rel_types;
        for (IDX_TYPE i = 0; i < num_discrete + num_Galerkin + 1; i++) {
            rel_types.push_back(trans_smth[namelist[cnt++]]);
            // if (my_pid == 0) printf("i %d type %d\n", i, rel_types[i]);
        }
        real_prec = new GeometricMultiGrid<int, __fp16, float>(num_discrete, num_Galerkin, {}, rel_types);
    } else {
        if (my_pid == 0) printf("Adaptor: NO preconditioner was set.\n");
    }
}

void Adaptor_64_for_32::SetOperator(const Operator<int, double, double> & op) {
    this->input_dim[0] = op.input_dim[0];
    this->input_dim[1] = op.input_dim[1];
    this->input_dim[2] = op.input_dim[2];

    this->output_dim[0] = op.output_dim[0];
    this->output_dim[1] = op.output_dim[1];
    this->output_dim[2] = op.output_dim[2];
    // 在此将外迭代的问题矩阵截断，并传给真正的预条件子
    const par_structMatrix<int, double, double> & A_problem = 
        (const par_structMatrix<int, double, double>&)(op);
    int gx = A_problem.input_dim[0],
        gy = A_problem.input_dim[1],
        gz = A_problem.input_dim[2];
    int px = gx / A_problem.local_matrix->local_x,
        py = gy / A_problem.local_matrix->local_y,
        pz = gz / A_problem.local_matrix->local_z;
    A_trc = new par_structMatrix<int, float, float>
            (MPI_COMM_WORLD, A_problem.num_diag, gx, gy, gz, px, py, pz);
    {// 截断
        const seq_structMatrix<int, double, double> & seq_A = *(A_problem.local_matrix);
        seq_structMatrix<int, float, float> & seq_A_trc = *(A_trc->local_matrix);
        CHECK_LOCAL_HALO(seq_A, seq_A_trc);
        const int tot_len = (seq_A.local_x + seq_A.halo_x * 2) * (seq_A.local_y + seq_A.halo_y * 2)
                        *   (seq_A.local_z + seq_A.halo_z * 2) * seq_A.num_diag;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < tot_len; i++)
            seq_A_trc.data[i] = seq_A.data[i];
        // 非规则点
        if (A_problem.num_irrgPts > 0) {
            const int num_irrgPts = A_problem.num_irrgPts;
            A_trc->num_irrgPts = num_irrgPts;
            A_trc->irrgPts = new irrgPts_mat<int> [num_irrgPts];
            for (int ir = 0; ir < num_irrgPts; ir++) 
                A_trc->irrgPts[ir] = A_problem.irrgPts[ir];

            const int tot_nnz = A_trc->irrgPts[num_irrgPts-1].beg + A_trc->irrgPts[num_irrgPts-1].nnz;
            A_trc->irrgPts_ngb_ijk = new int [tot_nnz * 3];// i j k
            for (int j = 0; j < tot_nnz * 3; j++)
                A_trc->irrgPts_ngb_ijk[j] = A_problem.irrgPts_ngb_ijk[j];

            A_trc->irrgPts_A_vals = new float [tot_nnz * 2];// two-sided effect
            for (int j = 0; j < tot_nnz * 2; j++)
                A_trc->irrgPts_A_vals[j] = A_problem.irrgPts_A_vals[j];
        }
    }
    if (real_prec) {// 有预条件
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("prec need to scale\n");
        real_prec->SetOperator(*A_trc);
    }
    tmp_b = new par_structVector<int, float>(MPI_COMM_WORLD, gx, gy, gz, px, py, pz, A_problem.num_diag!=7);

    int irrgPts_gids[A_trc->num_irrgPts];
    for (int ir = 0; ir < A_trc->num_irrgPts; ir++)
        irrgPts_gids[ir] = A_trc->irrgPts[ir].gid;
    tmp_b->init_irrgPts(A_trc->num_irrgPts, irrgPts_gids);
    tmp_x = new par_structVector<int, float>(*tmp_b);
}

void static inline truncate_f64_to_f32(const int num, const double * src, float * dst)
{
    int k = 0;
    const int max_6k = num & (~5);
    const int max_4k = num & (~3);
    for ( ; k < max_6k; k += 6) {// 每次处理6个双精度数
        float64x2x3_t high = vld1q_f64_x3(src);
        float32x2_t low0 = vcvt_f32_f64(high.val[0]),
                    low1 = vcvt_f32_f64(high.val[1]),
                    low2 = vcvt_f32_f64(high.val[2]);
        vst1_f32(dst, low0); dst += 2;
        vst1_f32(dst, low1); dst += 2;
        vst1_f32(dst, low2); dst += 2;
        src += 6;
    }
    for ( ; k < max_4k; k += 4) {
        float64x2x2_t high = vld1q_f64_x2(src);
        float32x2_t low0 = vcvt_f32_f64(high.val[0]),
                    low1 = vcvt_f32_f64(high.val[1]);
        vst1_f32(dst, low0); dst += 2;
        vst1_f32(dst, low1); dst += 2;
        src += 4;
    }
    for (k = 0; k < num - max_4k; k++)
        dst[k] = src[k];
}

void static inline recover_f32_to_f64(const int num, const float * src, double * dst)
{
    int k = 0;
    const int max_12k = num & (~11);
    const int max_8k = num & (~7);
    const int max_4k = num & (~3);
    for ( ; k < max_12k; k += 12) {// 每次处理12个单精度数
        float32x4x3_t low = vld1q_f32_x3(src);
        float64x2_t high00 = vcvt_f64_f32(vget_low_f32(low.val[0])), high01 = vcvt_high_f64_f32(low.val[0]),
                    high10 = vcvt_f64_f32(vget_low_f32(low.val[1])), high11 = vcvt_high_f64_f32(low.val[1]),
                    high20 = vcvt_f64_f32(vget_low_f32(low.val[2])), high21 = vcvt_high_f64_f32(low.val[2]);
        vst1q_f64(dst, high00); dst += 2; vst1q_f64(dst, high01); dst += 2;
        vst1q_f64(dst, high10); dst += 2; vst1q_f64(dst, high11); dst += 2;
        vst1q_f64(dst, high20); dst += 2; vst1q_f64(dst, high21); dst += 2;
        src += 12;
    }
    for ( ; k < max_8k; k += 8) {
        float32x4x2_t low = vld1q_f32_x2(src);
        float64x2_t high00 = vcvt_f64_f32(vget_low_f32(low.val[0])), high01 = vcvt_high_f64_f32(low.val[0]),
                    high10 = vcvt_f64_f32(vget_low_f32(low.val[1])), high11 = vcvt_high_f64_f32(low.val[1]);
        vst1q_f64(dst, high00); dst += 2; vst1q_f64(dst, high01); dst += 2;
        vst1q_f64(dst, high10); dst += 2; vst1q_f64(dst, high11); dst += 2;
        src += 8;
    }
    for ( ; k < max_4k; k += 4) {
        float32x4_t low = vld1q_f32(src);
        float64x2_t high00 = vcvt_f64_f32(vget_low_f32(low)), high01 = vcvt_high_f64_f32(low);
        vst1q_f64(dst, high00); dst += 2; vst1q_f64(dst, high01); dst += 2;
        src += 4;
    }
    for (k = 0; k < num - max_4k; k++)
        dst[k] = src[k];
}

void Adaptor_64_for_32::Mult(const par_structVector<int, double> & input,
        par_structVector<int, double> & output, bool use_zero_guess) const
{
    this->zero_guess = use_zero_guess;
    const int   jbeg = input.local_vector->halo_y, jend = jbeg + input.local_vector->local_y,
                ibeg = input.local_vector->halo_x, iend = ibeg + input.local_vector->local_x,
                kbeg = input.local_vector->halo_z, kend = kbeg + input.local_vector->local_z;
    const int col_height = kend - kbeg;
    const int vec_ki_size = input.local_vector->slice_ki_size, vec_k_size = input.local_vector->slice_k_size;
    {// copy in
        const seq_structVector<int, double> & src = *(input.local_vector);
        seq_structVector<int, float> & dst = *(tmp_b->local_vector);
        CHECK_LOCAL_HALO(src, dst);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = jbeg; j < jend; j++)
        for (int i = ibeg; i < iend; i++) {
            const double* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
            float       * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
            for (int k = 0; k < col_height; k++)
                dst_ptr[k] = src_ptr[k];
        }
        assert(input.num_irrgPts == tmp_b->num_irrgPts);
        for (int ir = 0; ir < input.num_irrgPts; ir++) {
            assert(tmp_b->irrgPts[ir].gid == input.irrgPts[ir].gid);
            tmp_b->irrgPts[ir].val = input.irrgPts[ir].val;
        }
    }
    if (this->zero_guess == false) {// 有非零初始值
        const seq_structVector<int, double> & src = *(output.local_vector);
        seq_structVector<int, float> & dst = *(tmp_x->local_vector);
        CHECK_LOCAL_HALO(src, dst);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = jbeg; j < jend; j++)
        for (int i = ibeg; i < iend; i++) {
            const double* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
            float       * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
            for (int k = 0; k < col_height; k++)
                dst_ptr[k] = src_ptr[k];
        }
        assert(output.num_irrgPts == tmp_x->num_irrgPts);
        for (int ir = 0; ir < output.num_irrgPts; ir++) {
            assert(tmp_x->irrgPts[ir].gid == output.irrgPts[ir].gid);
            tmp_x->irrgPts[ir].val = output.irrgPts[ir].val;
        }
    }

    // double dot = 0.0;
    // dot = vec_dot<int, float, double>(*tmp_b, *tmp_b);
    // printf("before real_prec (b,b) = %.10e\n", dot);

    if (real_prec)
        real_prec->Mult(*tmp_b, *tmp_x, this->zero_guess);
    else
        vec_copy(*tmp_b, *tmp_x);

    // dot = vec_dot<int, float, double>(*tmp_x, *tmp_x);
    // printf("after  real_prec (x,x) = %.10e\n", dot);

    
    {// copy out
        const seq_structVector<int, float> & src = *(tmp_x->local_vector);
        seq_structVector<int, double> & dst = *(output.local_vector);
        CHECK_LOCAL_HALO(src, dst);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = jbeg; j < jend; j++)
        for (int i = ibeg; i < iend; i++) {
            const float* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
            double     * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
            for (int k = 0; k < col_height; k++)
                dst_ptr[k] = src_ptr[k];
        }
        assert(output.num_irrgPts == tmp_x->num_irrgPts);
        for (int ir = 0; ir < output.num_irrgPts; ir++) {
            assert(tmp_x->irrgPts[ir].gid == output.irrgPts[ir].gid);
            output.irrgPts[ir].val = tmp_x->irrgPts[ir].val;
        }
    }
    this->zero_guess = false;
}

#endif