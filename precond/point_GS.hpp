#ifndef SMG_POINT_GS_HPP
#define SMG_POINT_GS_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
class PointGS : public Solver<idx_t, data_t, oper_t, res_t> {
public:
    // 对称GS：0 for sym, 1 for forward, -1 backward
    SCAN_TYPE scan_type = SYMMETRIC;
    mutable bool last_time_forward = false;

    // operator (often as matrix-A)
    const Operator<idx_t, oper_t, res_t> * oper = nullptr;

    // separate diagonal values if for efficiency concern is needed
    // should only be used when separation is cheap
    bool LU_separated = false;
    seq_structMatrix<idx_t, data_t, res_t> * L = nullptr;
    seq_structMatrix<idx_t, data_t, res_t> * U = nullptr;
    void separate_LU();

    bool Diags_separated = false;
    idx_t Diags_cnt = 0;
    seq_structVector<idx_t, data_t> ** Diags = nullptr;
    // AOS => SOA
    void separate_Diags();

    void (*AOS_forward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t*) = nullptr;
    void (*AOS_forward_zero_scaled)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t*) = nullptr;
    void (*AOS_forward_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t*) = nullptr;
    void (*AOS_forward_ALL_scaled)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t *) = nullptr;
    void (*AOS_backward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t *) = nullptr;
    void (*AOS_backward_zero_scaled)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t *) = nullptr;
    void (*AOS_backward_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t *) = nullptr;
    void (*AOS_backward_ALL_scaled)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t *, const data_t *, data_t *, const data_t *) = nullptr;

    PointGS() : Solver<idx_t, data_t, oper_t, res_t>() {  }
    PointGS(SCAN_TYPE type) : Solver<idx_t, data_t, oper_t, res_t>(), scan_type(type) {
        if (type == FORW_BACK)      last_time_forward = false;
        else if (type == BACK_FORW) last_time_forward = true;
    }

    virtual void SetOperator(const Operator<idx_t, oper_t, res_t> & op) {
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        if (sizeof(oper_t) != sizeof(data_t)) {
            assert(sizeof(data_t) < sizeof(oper_t));
            MPI_Abort(MPI_COMM_WORLD, -10200);
            // separate_Diags();
        }
        else {
            separate_LU();
            const idx_t num_diag = ((const par_structMatrix<idx_t, oper_t, res_t>&)op).num_diag;
            switch(num_diag)
            {
            case 7:
                AOS_forward_zero        = AOS_point_forward_zero_3d7<idx_t, data_t>;
                AOS_forward_zero_scaled = AOS_point_forward_zero_3d7_scaled<idx_t, data_t>;
                AOS_forward_ALL         = AOS_point_forward_ALL_3d7<idx_t, data_t>;
                AOS_forward_ALL_scaled  = AOS_point_forward_ALL_3d7_scaled<idx_t, data_t>;
                AOS_backward_zero       = AOS_point_backward_zero_3d7<idx_t, data_t>;
                AOS_backward_zero_scaled= AOS_point_backward_zero_3d7_scaled<idx_t, data_t>;
                AOS_backward_ALL        = AOS_point_backward_ALL_3d7<idx_t, data_t>;
                AOS_backward_ALL_scaled = AOS_point_backward_ALL_3d7_scaled<idx_t, data_t>;
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10200);
            }
        }
    }

    virtual void SetScanType(SCAN_TYPE type) {scan_type = type;}

    void truncate() {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: PGS truncated\n");
        assert(LU_separated || Diags_separated);
        if (LU_separated) {
            assert(L != nullptr && U != nullptr);// 在截断之前务必已经做完分解了
	    	L->truncate();
		    U->truncate();
        }
        if (Diags_separated) {
#ifdef __aarch64__
            for (idx_t id = 0; id < Diags_cnt; id++) {
                assert(Diags[id] != nullptr);
                idx_t tot_len =     (Diags[id]->halo_y * 2 + Diags[id]->local_y) 
                                *   (Diags[id]->halo_x * 2 + Diags[id]->local_x) * (Diags[id]->halo_z * 2 + Diags[id]->local_z);
                for (idx_t i = 0; i < tot_len; i++) {
                    __fp16 tmp = (__fp16) Diags[id]->data[i];
                    // if (i == 1235) printf("PGS::Diags truncate %.20e to", Diags[id]->data[i]);
                    Diags[id]->data[i] = (data_t) tmp;
                    // if (i == 1235) printf("%.20e\n", Diags[id]->data[i]);
                }
            }
#else
            printf("architecture not support truncated to fp16\n");
#endif
        }
    }

protected:
    // 近似求解一个（残差）方程，以b为右端向量，返回x为近似解
    void Mult(const par_structVector<idx_t, res_t> & b, 
                    par_structVector<idx_t, res_t> & x) const;
public:
    void Mult(const par_structVector<idx_t, res_t> & b,
                    par_structVector<idx_t, res_t> & x, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(b, x);
        this->zero_guess = false;// reset for safety concern
    }

    void ForwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
    void BackwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
    void BackwardPass_FFW0(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;

    void ForwardPass_neon_prft(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
    void BackwardPass_neon_prft(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
    void BackwardPass_FFW0_neon_prft(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;

    virtual ~PointGS();
};

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
PointGS<idx_t, data_t, oper_t, res_t>::~PointGS() {
    if (LU_separated) {
        if (L != nullptr) {delete L; L = nullptr;}
        if (U != nullptr) {delete U; U = nullptr;}
    }
    if (Diags_separated) {
        for (idx_t i = 0; i < Diags_cnt; i++) {
            delete Diags[i];
            Diags[i] = nullptr;
        }
        delete [] Diags;
    }
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void PointGS<idx_t, data_t, oper_t, res_t>::Mult(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const {
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double t = 0.0, bytes;
#endif

    if (sizeof(data_t) == sizeof(oper_t)) {
        switch (scan_type)
        {
        // 零初值优化，注意在对称GS中的后扫和单纯后扫的零初值优化是不一样的，小心！！！
        // 需要注意这个x传进来时可能halo区并不是0，如果不设置0并通信更新halo区，会导致使用错误的值，从而收敛性变慢
        // 还需要注意，这个x传进来时可能数据区不是0，如果不采用0初值的优化代码，而直接采用naive版本但是又没有数据区清零，则会用到旧值，从而收敛性变慢
        case SYMMETRIC:
            if (this->zero_guess) 
                x.set_val(0.0, true);
            else
                x.update_halo();
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            ForwardPass(b, x);
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time() - t;
            if (my_pid == 0) {
                int num_diag;
                if (this->zero_guess)
                    num_diag = (LDU_separated) ? (L->num_diag + 1) : ( (Diags_cnt+1)/2 );
                else
                    num_diag = (LDU_separated) ? (L->num_diag + 1 + U->num_diag) : Diags_cnt;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("Forward data_t %d oper_t %d diag %d total %.2f GB time %.5f s BW %.2f GB/s\n",
                     sizeof(data_t), sizeof(oper_t), num_diag, bytes, t, bytes/t);
            }
#endif
            
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            // 是否要注释掉这行决定后扫零初值的，取决于迭代次数是否会被显著影响
            // 一般在前扫和后扫中间加一次通信，有利于减少迭代数，次数和访存量的减少需要权衡
            this->zero_guess = false;
            // if (this->zero_guess) {
            //     // 不通信！！！
            //     // 这个函数一定要与“halo区不填充”同时使用！！！如果前扫完了对x进行通信，而仍使用BackwardPass_FFW0，会导致压根不收敛！！！
            //     BackwardPass_FFW0(b, x);// following forward with 0, 
            // } else {
                x.update_halo();// 通信完之后halo区是非零的，用普通版本的后扫
                BackwardPass(b, x);
            // }
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time() - t;
            if (my_pid == 0) {
                int num_diag;
                if (this->zero_guess)
                    num_diag = (LDU_separated) ? (U->num_diag + 1) : ( (Diags_cnt+1)/2 );
                else
                    num_diag = (LDU_separated) ? (L->num_diag + 1 + U->num_diag) : Diags_cnt;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("Backwad data_t %d oper_t %d diag %d total %.2f GB time %.5f s BW %.2f GB/s\n",
                     sizeof(data_t), sizeof(oper_t), num_diag, bytes, t, bytes/t);
            }
#endif
            break;
        case FORWARD:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            ForwardPass(b, x);
            break;
        case BACKWARD:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            BackwardPass(b, x);
            break; 
        case FORW_BACK:
        case BACK_FORW:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            if (last_time_forward) {
                BackwardPass(b, x);
                last_time_forward = false;
            } else {
                ForwardPass(b, x);
                last_time_forward = false;
            }
            break;
        default:// do nothing, just act as an identity operator
            vec_copy(b, x);
            break;
        }
    } else {
        // 务必从上面正常精度的抄，有重要修改！！！
        /*
        assert(sizeof(data_t) == 2);
        assert(scan_type == SYMMETRIC);
        switch (scan_type)
        {
        case SYMMETRIC:
            if (this->zero_guess) 
                x.set_val(0.0, true);
            else
                x.update_halo();
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            ForwardPass_neon_prft(b, x);
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time() - t;
            if (my_pid == 0) {
                int num_diag;
                if (this->zero_guess)
                    num_diag = (LDU_separated) ? (L->num_diag + 1) : ( (Diags_cnt+1)/2 );
                else
                    num_diag = (LDU_separated) ? (L->num_diag + 1 + U->num_diag) : Diags_cnt;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("Forward data_t %d oper_t %d diag %d total %.2f GB time %.5f s BW %.2f GB/s\n",
                     sizeof(data_t), sizeof(oper_t), num_diag, bytes, t, bytes/t);
            }
#endif
            x.update_halo();
            // x.set_halo(0.0);
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
            if (this->zero_guess)
                BackwardPass_FFW0_neon_prft(b, x);
            else
                BackwardPass_neon_prft(b, x);
#ifdef PROFILE
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time() - t;
            if (my_pid == 0) {
                int num_diag;
                if (this->zero_guess)
                    num_diag = (LDU_separated) ? (U->num_diag + 1) : ( (Diags_cnt+1)/2 );
                else
                    num_diag = (LDU_separated) ? (L->num_diag + 1 + U->num_diag) : Diags_cnt;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("Backwad data_t %d oper_t %d diag %d total %.2f GB time %.5f s BW %.2f GB/s\n",
                     sizeof(data_t), sizeof(oper_t), num_diag, bytes, t, bytes/t);
            }
#endif
            break;
        default:// do nothing, just act as an identity operator
            vec_copy(b, x);
            break;
        }
        */
       MPI_Abort(MPI_COMM_WORLD, -10220);
    }
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void PointGS<idx_t, data_t, oper_t, res_t>::ForwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const
{
    const seq_structVector<idx_t, res_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, res_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, oper_t, res_t> * par_A = (par_structMatrix<idx_t, oper_t, res_t>*)(this->oper);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    assert(LU_separated);

    const res_t * b_data = b_vec.data;
          res_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;

    const res_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const data_t * mat_data, * aux_data;
    idx_t slice_dki_size, slice_dk_size, num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t, 
                    const data_t*, const data_t*, data_t*, const data_t*) = nullptr;
    
    const bool & scaled = par_A->scaled;
    aux_data = scaled ? par_A->sqrt_D->data : nullptr;
    if (this->zero_guess) {
        mat_data = L->data;
        slice_dki_size = L->slice_dki_size;
        slice_dk_size = L->slice_dk_size;
        num_diag = L->num_diag;
        kernel = scaled ? AOS_forward_zero_scaled : AOS_forward_zero;
    } else {
        mat_data = par_A->local_matrix->data;
        slice_dki_size = par_A->local_matrix->slice_dki_size;
        slice_dk_size = par_A->local_matrix->slice_dk_size;
        num_diag = par_A->num_diag;
        kernel = scaled ? AOS_forward_ALL_scaled : AOS_forward_ALL;
    }

    if (num_threads > 1) {// level-based的多线程并行
        // level是等值线 slope * j + i = Const, 对于3d7和3d15 斜率为1, 对于3d19和3d27 斜率为2
        const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[0] = dim_1 - 1;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j + 1] = -1;// 初始化为-1
        const idx_t wait_offi = slope - 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = 0; lid < nlevs; lid++) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_end - 1; it >= t_beg; it--) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    idx_t i_to_wait = (i == iend - 1) ? i_lev : (i_lev + wait_offi);
                    const data_t * mat_jik = mat_data + j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
                    const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
                    const data_t * aux_jik = scaled ? (aux_data + vec_off) : nullptr;
                    res_t * x_jik = x_data + vec_off;
                    const res_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }
                    
                    // 中间的不需等待
                    kernel(col_height, vec_k_size, vec_ki_size, weight, mat_jik, b_jik, x_jik, aux_jik);

                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev+1] = i_lev;
                }
            }
        }
    }
    else {
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            const data_t * mat_jik = mat_data + j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
            const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
            const data_t * aux_jik = scaled ? (aux_data + vec_off) : nullptr;
            res_t * x_jik = x_data + vec_off;
            const res_t * b_jik = b_data + vec_off;
            kernel(col_height, vec_k_size, vec_ki_size, weight, mat_jik, b_jik, x_jik, aux_jik);
        }
    }
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void PointGS<idx_t, data_t, oper_t, res_t>::BackwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const
{
    const seq_structVector<idx_t, res_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, res_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, oper_t, res_t> * par_A = (par_structMatrix<idx_t, oper_t, res_t> *)(this->oper);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    assert(LU_separated);

    const res_t * b_data = b_vec.data;
          res_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;
    
    const res_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const data_t * mat_data, * aux_data;
    idx_t slice_dki_size, slice_dk_size, num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t, 
                    const data_t*, const data_t*, data_t*, const data_t*) = nullptr;
    
    const bool & scaled = par_A->scaled;
    aux_data = scaled ? par_A->sqrt_D->data : nullptr;
    if (this->zero_guess) {
        mat_data = U->data;
        slice_dki_size = U->slice_dki_size;
        slice_dk_size = U->slice_dk_size;
        num_diag = U->num_diag;
        kernel = scaled ? AOS_backward_zero_scaled : AOS_backward_zero;
    } else {
        mat_data = par_A->local_matrix->data;
        slice_dki_size = par_A->local_matrix->slice_dki_size;
        slice_dk_size = par_A->local_matrix->slice_dk_size;
        num_diag = par_A->num_diag;
        kernel = scaled ? AOS_backward_ALL_scaled : AOS_backward_ALL;
    }

    if (num_threads > 1) {// level-based的多线程并行
        const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j] = dim_1;// 初始化
        const idx_t wait_offi = - (slope - 1);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = nlevs - 1; lid >= 0; lid--) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_beg; it < t_end; it++) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    idx_t i_to_wait = (i == ibeg) ? i_lev : (i_lev + wait_offi);
                    const data_t * mat_jik = mat_data + j * slice_dki_size + i * slice_dk_size + (kend - 1) * num_diag;// 注意从尾部开始
                    const idx_t vec_off = j * vec_ki_size + i * vec_k_size + (kend - 1);
                    const data_t * aux_jik = scaled ? (aux_data + vec_off) : nullptr;
                    res_t * x_jik = x_data + vec_off;
                    const res_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
                    // 中间的不需等待
                    kernel(col_height, vec_k_size, vec_ki_size, weight, mat_jik, b_jik, x_jik, aux_jik);
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev] = i_lev;
                }
            }
        }
    }
    else {
        for (idx_t j = jend - 1; j >= jbeg; j--)
        for (idx_t i = iend - 1; i >= ibeg; i--) {
            const data_t * mat_jik = mat_data + j * slice_dki_size + i * slice_dk_size + (kend - 1) * num_diag;// 注意从尾部开始
            const idx_t vec_off = j * vec_ki_size + i * vec_k_size + (kend - 1);
            const data_t * aux_jik = scaled ? (aux_data + vec_off) : nullptr;
            res_t * x_jik = x_data + vec_off;
            const res_t * b_jik = b_data + vec_off;
            kernel(col_height, vec_k_size, vec_ki_size, weight, mat_jik, b_jik, x_jik, aux_jik);
        }
    }
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void PointGS<idx_t, data_t, oper_t, res_t>::separate_LU() {
    assert(this->oper != nullptr);
    assert(!LU_separated);
    // 提取矩阵对角元到向量，提取L和U到另一个矩阵
    assert(this->oper->input_dim[0] == this->oper->output_dim[0] &&
           this->oper->input_dim[1] == this->oper->output_dim[1] &&
           this->oper->input_dim[2] == this->oper->output_dim[2] );

    const seq_structMatrix<idx_t, oper_t, res_t> & mat = *(((par_structMatrix<idx_t, oper_t, res_t> *) oper)->local_matrix);
    const idx_t diag_block_width = 1;
    assert((mat.num_diag - diag_block_width) % 2 ==0);

    L = new seq_structMatrix<idx_t, data_t, res_t>( (mat.num_diag + diag_block_width) / 2, // 包含对角线
                                            mat.local_x, mat.local_y, mat.local_z, mat.halo_x, mat.halo_y, mat.halo_z);
    U = new seq_structMatrix<idx_t, data_t, res_t>(*L);

    const idx_t jbeg = 0, jend = mat.local_y + 2 * mat.halo_y,
                ibeg = 0, iend = mat.local_x + 2 * mat.halo_x,
                kbeg = 0, kend = mat.local_z + 2 * mat.halo_z;

    if (mat.num_diag == 7) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
        /*  
                  /--------------/                    
                 / |    4       / |                   
                /--|-----------/  |
               |   |           |  |
               |   |------6----|--|
   z   y       | 1 |    3      | 5|
   ^  ^        |---|-- 0 ------|  |
   | /         |   |           |  |
   |/          |   /-----------|--/ 
   O-------> x | /      2      | / 
               |/--------------|/
                            
                    D : 3
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,
                    U : 4 ,  5,  6
                */
                L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
                L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];
                L->data[L_jik_loc + 3] = mat.data[A_jik_loc + 3];// 最后一个位置放对角元

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 4];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 5];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc + 6];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc + 3];// 最后一个位置放对角元

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
    }
    else if (mat.num_diag == 19) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
                /*
                    D : 9
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,  3,  4,  5,  6,  7,  8
                    U : 10, 11, 12, 13, 14, 15, 16, 17, 18
                */
                L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
                L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];
                L->data[L_jik_loc + 3] = mat.data[A_jik_loc + 3];
                L->data[L_jik_loc + 4] = mat.data[A_jik_loc + 4];
                L->data[L_jik_loc + 5] = mat.data[A_jik_loc + 5];
                L->data[L_jik_loc + 6] = mat.data[A_jik_loc + 6];
                L->data[L_jik_loc + 7] = mat.data[A_jik_loc + 7];
                L->data[L_jik_loc + 8] = mat.data[A_jik_loc + 8];
                L->data[L_jik_loc + 9] = mat.data[A_jik_loc + 9];// 最后一个位置放对角元

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 10];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 11];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc + 12];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc + 13];
                U->data[U_jik_loc + 4] = mat.data[A_jik_loc + 14];
                U->data[U_jik_loc + 5] = mat.data[A_jik_loc + 15];
                U->data[U_jik_loc + 6] = mat.data[A_jik_loc + 16];
                U->data[U_jik_loc + 7] = mat.data[A_jik_loc + 17];
                U->data[U_jik_loc + 8] = mat.data[A_jik_loc + 18];
                U->data[U_jik_loc + 9] = mat.data[A_jik_loc + 9];// 最后一个位置放对角元

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
    }
    else if (mat.num_diag == 27) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
                /*
                    D : 13
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
                    U : 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
                */
                L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
                L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];
                L->data[L_jik_loc + 3] = mat.data[A_jik_loc + 3];
                L->data[L_jik_loc + 4] = mat.data[A_jik_loc + 4];
                L->data[L_jik_loc + 5] = mat.data[A_jik_loc + 5];
                L->data[L_jik_loc + 6] = mat.data[A_jik_loc + 6];
                L->data[L_jik_loc + 7] = mat.data[A_jik_loc + 7];
                L->data[L_jik_loc + 8] = mat.data[A_jik_loc + 8];
                L->data[L_jik_loc + 9] = mat.data[A_jik_loc + 9];
                L->data[L_jik_loc +10] = mat.data[A_jik_loc +10];
                L->data[L_jik_loc +11] = mat.data[A_jik_loc +11];
                L->data[L_jik_loc +12] = mat.data[A_jik_loc +12];
                L->data[L_jik_loc +13] = mat.data[A_jik_loc +13];

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc +14];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc +15];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc +16];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc +17];
                U->data[U_jik_loc + 4] = mat.data[A_jik_loc +18];
                U->data[U_jik_loc + 5] = mat.data[A_jik_loc +19];
                U->data[U_jik_loc + 6] = mat.data[A_jik_loc +20];
                U->data[U_jik_loc + 7] = mat.data[A_jik_loc +21];
                U->data[U_jik_loc + 8] = mat.data[A_jik_loc +22];
                U->data[U_jik_loc + 9] = mat.data[A_jik_loc +23];
                U->data[U_jik_loc +10] = mat.data[A_jik_loc +24];
                U->data[U_jik_loc +11] = mat.data[A_jik_loc +25];
                U->data[U_jik_loc +12] = mat.data[A_jik_loc +26];
                U->data[U_jik_loc +13] = mat.data[A_jik_loc +13];

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
    }
    else {
        printf("PointGS::separate_LDU: num_diag of %d not yet supported\n", mat.num_diag);
        MPI_Abort(MPI_COMM_WORLD, -4000);
    }
    LU_separated = true;
}

#endif