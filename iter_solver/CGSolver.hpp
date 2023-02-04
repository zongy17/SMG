#ifndef SMG_CG_HPP
#define SMG_CG_HPP

#include "iter_solver.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
class CGSolver : public IterativeSolver<idx_t, pc_data_t, pc_calc_t, ksp_t> {
public:

    CGSolver() {  };
    void Pipelined          (const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const;
    void Chronopoulos_Gear  (const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const;
    void Standard           (const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const;
    
    // 默认采用Chronopoulos-Gear CG（每迭代步只需一次AllReduce）
    virtual void Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const {
        // Pipelined(b, x);
        // Chronopoulos_Gear(b, x);
        Standard(b, x);
    }
};

template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
void CGSolver<idx_t, pc_data_t, pc_calc_t, ksp_t>::Pipelined(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    CHECK_VEC_GLOBAL(x, b);// 只有方阵才能求逆
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);

    // 初始化辅助向量
    par_structVector<idx_t, ksp_t> r(x), u(x), w(x), m(x), n(x), p(x), s(x), q(x), z(x);
    r.set_halo(0.0); u.set_halo(0.0); w.set_halo(0.0);
    m.set_halo(0.0); n.set_halo(0.0); p.set_halo(0.0);
    s.set_halo(0.0); q.set_halo(0.0); z.set_halo(0.0);
    double gamma, delta, eta, alpha, beta; 
    double tmp_loc[4], tmp_glb[4];// 用作局部点积的存储，发送缓冲区和接受缓冲区

    this->oper->Mult(x, r, false);
    vec_add(b, -1.0, r, r);// r0 <- b-A*x0
    if (this->prec) {
        this->prec->Mult(r, u, true);// u0 <- M^{−1}*r0
    } else {
        vec_copy(r, u);
    }
    this->oper->Mult(u, w, false);// w0 <- A*u0

    MPI_Request dot_req = MPI_REQUEST_NULL;
    tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(r.local_vector));// γ0
    tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(w.local_vector));// δ0
    tmp_loc[2] = seq_vec_dot<idx_t, ksp_t, double>(*(b.local_vector), *(b.local_vector));
    tmp_loc[3] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
    MPI_Iallreduce(tmp_loc, tmp_glb, 4, MPI_DOUBLE, MPI_SUM, comm, &dot_req);

    if (this->prec) {
        this->prec->Mult(w, m, true);// m0 <- M^{−1}*w0
    } else {
        vec_copy(w, m);
    }
    this->oper->Mult(m, n, false);// n0 <- A*m0

    vec_copy(u, p);
    vec_copy(w, s);
    vec_copy(m, q);
    vec_copy(n, z);

    MPI_Wait(&dot_req, MPI_STATUS_IGNORE);
    gamma = tmp_glb[0];
    delta = tmp_glb[1];
    double norm_b = sqrt(tmp_glb[2]);
    double norm_r = sqrt(tmp_glb[3]);
    eta = delta;
    alpha = gamma / eta;

    int & iter = this->final_iter;
    if (my_pid == 0) printf("iter %4d   ||b|| = %.16e ||r||/||b|| = %.16e\n", iter, norm_b, norm_r / norm_b);

    iter++;// 初始化也算一步
    for ( ; iter < this->max_iter; iter++) {
        vec_add(x,  alpha, p, x);// x(i) <- x(i-1) + α(i-1)*p(i-1) 
        vec_add(r, -alpha, s, r);// r(i) <- r(i-1) - α(i-1)*s(i-1)
        vec_add(u, -alpha, q, u);// u(i) <- u(i-1) - α(i-1)*q(i-1)
        vec_add(w, -alpha, z, w);// w(i) <- w(i-1) - α(i-1)*z(i-1)

        tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(r.local_vector));// γ(i)的局部点积
        tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(w.local_vector));// δ(i)的局部点积
        tmp_loc[2] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
        MPI_Iallreduce(tmp_loc, tmp_glb, 3, MPI_DOUBLE, MPI_SUM, comm, &dot_req);

        if (this->prec) {
            this->prec->Mult(w, m, true);// m(i) <- M^{-1}*w(i)
        } else {
            vec_copy(w, m);
        }        
        this->oper->Mult(m, n, false);// n(i) <- A*m(i)

        MPI_Wait(&dot_req, MPI_STATUS_IGNORE);
        double old_gamma = gamma;
        gamma = tmp_glb[0];
        delta = tmp_glb[1];
        norm_r = sqrt(tmp_glb[2]);

        double _rel = norm_r / norm_b;
        if (my_pid == 0) printf("iter %4d   alpha %.16e   ||r||/||b|| = %.16e\n", iter, alpha, _rel);
        if (_rel < this->rel_tol || norm_r < this->abs_tol) {// 判敛
            this->converged = 1;
            break;
        }

        beta = gamma / old_gamma;// β(i) <- γ(i) / γ(i-1)
        eta  = delta - beta*beta * eta;// η(i) <- δ(i) - |β(i)|^2 * η(i-1)
        alpha = gamma / eta;// α(i) <- γ(i) / η(i)
        
        vec_add(u, beta, p, p);// p(i) <- u(i) + β(i) * p(i−1)
        vec_add(w, beta, s, s);// s(i) <- w(i) + β(i) * s(i-1)
        vec_add(m, beta, q, q);// q(i) <- m(i) + β(i) * q(i-1)
        vec_add(n, beta, z, z);// z(i) <- n(i) + β(i) * z(i-1)
    }
}

template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
void CGSolver<idx_t, pc_data_t, pc_calc_t, ksp_t>::Chronopoulos_Gear(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const 
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    CHECK_VEC_GLOBAL(x, b);// 只有方阵才能求逆
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);

    // 初始化辅助向量
    par_structVector<idx_t, ksp_t> r(x), u(x), w(x), p(x), s(x);
    r.set_halo(0.0); u.set_halo(0.0); w.set_halo(0.0); p.set_halo(0.0); s.set_halo(0.0);
    double gamma, delta, eta, alpha, beta;
    double tmp_loc[4], tmp_glb[4];// 用作局部点积的存储，发送缓冲区和接受缓冲区

    this->oper->Mult(x, r, false);

    vec_add(b, -1.0, r, r);// r0 <- b-A*x0

    if (this->prec) {
        this->prec->Mult(r, u, true);// u0 <- M^{-1}*r0
    } else {
        vec_copy(r, u);
    }
    this->oper->Mult(u, w, false);// w0 <- A*u0

#ifdef DOT_FISSON
    tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(r.local_vector));// γ的局部点积
    tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(w.local_vector));// δ的局部点积
    tmp_loc[3] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
#else
    {
        const seq_structVector<idx_t, ksp_t>& u_loc = *(u.local_vector),
                                            & w_loc = *(w.local_vector),
                                            & r_loc = *(r.local_vector);
        CHECK_LOCAL_HALO(u_loc, r_loc);
        CHECK_LOCAL_HALO(u_loc, w_loc);
        CHECK_LOCAL_HALO(u_loc, r_loc);
        const idx_t xbeg = u_loc.halo_x, xend = xbeg + u_loc.local_x,
                    ybeg = u_loc.halo_y, yend = ybeg + u_loc.local_y,
                    zbeg = u_loc.halo_z, zend = zbeg + u_loc.local_z;
        const idx_t slice_k_size = u_loc.slice_k_size, slice_ki_size = u_loc.slice_ki_size;
        double dot_ur = 0.0, dot_uw = 0.0, dot_rr = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:dot_ur,dot_uw,dot_rr) schedule(static)
        for (idx_t j = ybeg; j < yend; j++)
        for (idx_t i = xbeg; i < xend; i++) {
            idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
            const ksp_t * u_data = u_loc.data + ji_loc, 
                        * w_data = w_loc.data + ji_loc,
                        * r_data = r_loc.data + ji_loc;
            for (idx_t k = zbeg; k < zend; k++) {
                dot_ur += (double) u_data[k] * (double) r_data[k];
                dot_uw += (double) u_data[k] * (double) w_data[k];
                dot_rr += (double) r_data[k] * (double) r_data[k];
            }
        }
        tmp_loc[0] = dot_ur;
        tmp_loc[1] = dot_uw;
        tmp_loc[3] = dot_rr;
    }
#endif
    tmp_loc[2] = seq_vec_dot<idx_t, ksp_t, double>(*(b.local_vector), *(b.local_vector));
    MPI_Allreduce(tmp_loc, tmp_glb, 4, MPI_DOUBLE, MPI_SUM, comm);

    gamma = tmp_glb[0];
    delta = tmp_glb[1];
    double norm_b = sqrt(tmp_glb[2]);
    double norm_r = sqrt(tmp_glb[3]);
    eta = delta;// η0 <- δ0
    vec_copy(u, p);// p0 <- u0
    vec_copy(w, s);// s0 <- w0
    alpha = gamma / delta;

    int & iter = this->final_iter;
    if (my_pid == 0) printf("iter %4d   ||b|| = %.16e ||r||/||b|| = %.16e\n", iter, norm_b, norm_r / norm_b);

    iter++;// 初始化也算一步
    for ( ; iter < this->max_iter; iter++) {
        vec_add(x,  alpha, p, x);// x(i) <- x(i-1) + α(i-1)*p(i-1) 
        vec_add(r, -alpha, s, r);// r(i) <- r(i-1) - α(i-1)*s(i-1)

        if (this->prec) {
            this->prec->Mult(r, u, true);// u(i) <- M^{-1}*r(i)
        } else {
            vec_copy(r, u);
        }

        this->oper->Mult(u, w, false);// w(i) <- A*u(i)

#ifdef DOT_FISSON
        tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(r.local_vector));// γ(i)的局部点积
        tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(w.local_vector));// δ(i)的局部点积
        tmp_loc[2] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
#else
        {
            const seq_structVector<idx_t, ksp_t>& u_loc = *(u.local_vector),
                                                & w_loc = *(w.local_vector),
                                                & r_loc = *(r.local_vector);
            CHECK_LOCAL_HALO(u_loc, r_loc);
            CHECK_LOCAL_HALO(u_loc, w_loc);
            CHECK_LOCAL_HALO(u_loc, r_loc);
            const idx_t xbeg = u_loc.halo_x, xend = xbeg + u_loc.local_x,
                        ybeg = u_loc.halo_y, yend = ybeg + u_loc.local_y,
                        zbeg = u_loc.halo_z, zend = zbeg + u_loc.local_z;
            const idx_t slice_k_size = u_loc.slice_k_size, slice_ki_size = u_loc.slice_ki_size;
            double dot_ur = 0.0, dot_uw = 0.0, dot_rr = 0.0;
            #pragma omp parallel for collapse(2) reduction(+:dot_ur,dot_uw,dot_rr) schedule(static)
            for (idx_t j = ybeg; j < yend; j++)
            for (idx_t i = xbeg; i < xend; i++) {
                idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
                const ksp_t * u_data = u_loc.data + ji_loc, 
                            * w_data = w_loc.data + ji_loc,
                            * r_data = r_loc.data + ji_loc;
                for (idx_t k = zbeg; k < zend; k++) {
                    dot_ur += (double) u_data[k] * (double) r_data[k];
                    dot_uw += (double) u_data[k] * (double) w_data[k];
                    dot_rr += (double) r_data[k] * (double) r_data[k];
                }
            }
            tmp_loc[0] = dot_ur;
            tmp_loc[1] = dot_uw;
            tmp_loc[2] = dot_rr;
        }
#endif
        MPI_Allreduce(tmp_loc, tmp_glb, 3, MPI_DOUBLE, MPI_SUM, comm);

        double old_gamma = gamma;
        gamma = tmp_glb[0];
        delta = tmp_glb[1];
        norm_r = sqrt(tmp_glb[2]);

        double _rel = norm_r / norm_b;
        if (my_pid == 0) printf("iter %4d   alpha %.16e   ||r||/||b|| = %.16e\n", iter, alpha, _rel);
        if (_rel < this->rel_tol || norm_r < this->abs_tol) {// 判敛
            this->converged = 1;
            break;
        }

        beta = gamma / old_gamma;// β(i) <- γ(i) / γ(i-1)
        eta  = delta - beta*beta * eta;// η(i) <- δ(i) - |β(i)|^2 * η(i-1)
        alpha = gamma / eta;// α(i) <- γ(i) / η(i)
        
        vec_add(u, beta, p, p);// p(i) <- u(i) + β(i) * p(i−1)
        vec_add(w, beta, s, s);// s(i) <- w(i) + β(i) * s(i-1)
    }
}

template<typename idx_t, typename pc_data_t, typename pc_calc_t, typename ksp_t>
void CGSolver<idx_t, pc_data_t, pc_calc_t, ksp_t>::Standard(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const 
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);
    double * record = this->part_times;
    memset(record, 0.0, sizeof(double) * NUM_KRYLOV_RECORD);

    // 初始化辅助向量：r残差，p搜索方向，Ap为A乘以搜索方向
    par_structVector<idx_t, ksp_t> r(x), u(x), p(x), s(x);
    par_structVector<idx_t, pc_calc_t> * pc_buf_b = nullptr, * pc_buf_x = nullptr;
    if constexpr (sizeof(pc_calc_t) != sizeof(ksp_t)) {
        const idx_t num_diag = ((const par_structMatrix<idx_t, ksp_t, ksp_t>*)this->oper)->num_diag;
        pc_buf_b = new par_structVector<idx_t, pc_calc_t>(r.comm_pkg->cart_comm,
            r.global_size_x, r.global_size_y, r.global_size_z,
            r.global_size_x/r.local_vector->local_x,
            r.global_size_y/r.local_vector->local_y,
            r.global_size_z/r.local_vector->local_z, num_diag != 7);
        pc_buf_x = new par_structVector<idx_t, pc_calc_t>(*pc_buf_b);
        pc_buf_b->set_halo(0.0);
        pc_buf_x->set_halo(0.0);
    }
    const idx_t jbeg = r.local_vector->halo_y, jend = jbeg + r.local_vector->local_y,
                ibeg = r.local_vector->halo_x, iend = ibeg + r.local_vector->local_x,
                kbeg = r.local_vector->halo_z, kend = kbeg + r.local_vector->local_z;
    const idx_t col_height = kend - kbeg;
    const idx_t vec_ki_size = r.local_vector->slice_ki_size, vec_k_size = r.local_vector->slice_k_size;
    r.set_halo(0.0); u.set_halo(0.0); p.set_halo(0.0); s.set_halo(0.0);
    double gamma, eta, alpha, beta;
    double tmp_loc[4], tmp_glb[4];// 用作局部点积的存储，发送缓冲区和接受缓冲区

    //record[OPER] -= wall_time();
    this->oper->Mult(x, r, false);
    //record[OPER] += wall_time();
    //record[AXPY] -= wall_time();
    vec_add(b, -1.0, r, r);
    //record[AXPY] += wall_time();
    
    // double ck_dot = 0.0;
    // ck_dot = vec_dot<idx_t, ksp_t, double>(r, r);
    // printf("before prec (b,b) = %.10e\n", ck_dot);
    if (this->prec) {// 有预条件子，则以预条件后的残差M^{-1}*r作为搜索方向 p = M^{-1}*r
        bool zero_guess = true;
        if constexpr (sizeof(pc_calc_t) != sizeof(ksp_t)) {
            {// copy in
                const seq_structVector<idx_t, ksp_t> & src = *(r.local_vector);
                seq_structVector<idx_t, pc_calc_t> & dst = *(pc_buf_b->local_vector);
                CHECK_LOCAL_HALO(src, dst);
                #pragma omp parallel for collapse(2) schedule(static)
                for (int j = jbeg; j < jend; j++)
                for (int i = ibeg; i < iend; i++) {
                    const double* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
                    float       * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
                    for (int k = 0; k < col_height; k++)
                        dst_ptr[k] = src_ptr[k];
                }
            }
            if (zero_guess) {
                pc_buf_x->set_val(0.0);
            } else {// copy in
                const seq_structVector<idx_t, ksp_t> & src = *(u.local_vector);
                seq_structVector<idx_t, pc_calc_t> & dst = *(pc_buf_x->local_vector);
                CHECK_LOCAL_HALO(src, dst);
                #pragma omp parallel for collapse(2) schedule(static)
                for (int j = jbeg; j < jend; j++)
                for (int i = ibeg; i < iend; i++) {
                    const double* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
                    float       * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
                    for (int k = 0; k < col_height; k++)
                        dst_ptr[k] = src_ptr[k];
                }
            }
            record[PREC] -= wall_time();
            this->prec->Mult(*pc_buf_b, *pc_buf_x, true);
            record[PREC] += wall_time();
            {// copy out
                const seq_structVector<idx_t, pc_calc_t> & src = *(pc_buf_x->local_vector);
                seq_structVector<idx_t, ksp_t> & dst = *(u.local_vector);
                CHECK_LOCAL_HALO(src, dst);
                #pragma omp parallel for collapse(2) schedule(static)
                for (int j = jbeg; j < jend; j++)
                for (int i = ibeg; i < iend; i++) {
                    const float* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
                    double     * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
                    for (int k = 0; k < col_height; k++)
                        dst_ptr[k] = src_ptr[k];
                }
            }
        } else {
            if (zero_guess) u.set_val(0.0);
            record[PREC] -= wall_time();
            this->prec->Mult(r, u, true);
            record[PREC] += wall_time();
        }
    } else {// 没有预条件则直接以残差r作为搜索方向
        //record[AXPY] -= wall_time();
        vec_copy(r, u);
        //record[AXPY] += wall_time();
    }
    // ck_dot = vec_dot<idx_t, ksp_t, double>(u, u);
    // printf("after  prec (x,x) = %.10e\n", ck_dot);

    //record[AXPY] -= wall_time();
    vec_copy(u, p);
    //record[AXPY] += wall_time();
    //record[OPER] -= wall_time();
    this->oper->Mult(p, s, false);
    //record[OPER] += wall_time();

    //record[DOT] -= wall_time();
    tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(r.local_vector));// γ的局部点积
    tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(s.local_vector), *(p.local_vector));// η的局部点积
    tmp_loc[2] = seq_vec_dot<idx_t, ksp_t, double>(*(b.local_vector), *(b.local_vector));
    tmp_loc[3] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
    MPI_Allreduce(tmp_loc, tmp_glb, 4, MPI_DOUBLE, MPI_SUM, comm);
    gamma = tmp_glb[0];
    eta   = tmp_glb[1];
    double norm_b = sqrt(tmp_glb[2]);
    double norm_r = sqrt(tmp_glb[3]);
    alpha = gamma / eta;
    //record[DOT] += wall_time();

    int & iter = this->final_iter;
    if (my_pid == 0) printf("iter %4d   ||b|| = %.16e ||r||/||b|| = %.16e\n", iter, norm_b, norm_r / norm_b);

    iter++;// 执行一次预条件子就算一次迭代
    for ( ; iter < this->max_iter; iter++) {
        //record[AXPY] -= wall_time();
        vec_add(x,  alpha, p, x);
        vec_add(r, -alpha, s, r);
        //record[AXPY] += wall_time();


        // double ck_dot = 0.0;
        // ck_dot = vec_dot<idx_t, ksp_t, double>(r, r);
        // printf("before prec (b,b) = %.10e\n", ck_dot);
        if (this->prec) {
            bool zero_guess = true;
            if constexpr (sizeof(pc_calc_t) != sizeof(ksp_t)) {
                {// copy in
                    const seq_structVector<idx_t, ksp_t> & src = *(r.local_vector);
                    seq_structVector<idx_t, pc_calc_t> & dst = *(pc_buf_b->local_vector);
                    CHECK_LOCAL_HALO(src, dst);
                    #pragma omp parallel for collapse(2) schedule(static)
                    for (int j = jbeg; j < jend; j++)
                    for (int i = ibeg; i < iend; i++) {
                        const double* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
                        float       * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
                        for (int k = 0; k < col_height; k++)
                            dst_ptr[k] = src_ptr[k];
                    }
                }
                if (zero_guess) {
                    pc_buf_x->set_val(0.0);
                } else {// copy in
                    const seq_structVector<idx_t, ksp_t> & src = *(u.local_vector);
                    seq_structVector<idx_t, pc_calc_t> & dst = *(pc_buf_x->local_vector);
                    CHECK_LOCAL_HALO(src, dst);
                    #pragma omp parallel for collapse(2) schedule(static)
                    for (int j = jbeg; j < jend; j++)
                    for (int i = ibeg; i < iend; i++) {
                        const double* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
                        float       * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
                        for (int k = 0; k < col_height; k++)
                            dst_ptr[k] = src_ptr[k];
                    }
                }
                record[PREC] -= wall_time();
                this->prec->Mult(*pc_buf_b, *pc_buf_x, true);
                record[PREC] += wall_time();
                {// copy out
                    const seq_structVector<idx_t, pc_calc_t> & src = *(pc_buf_x->local_vector);
                    seq_structVector<idx_t, ksp_t> & dst = *(u.local_vector);
                    CHECK_LOCAL_HALO(src, dst);
                    #pragma omp parallel for collapse(2) schedule(static)
                    for (int j = jbeg; j < jend; j++)
                    for (int i = ibeg; i < iend; i++) {
                        const float* src_ptr = src.data + j * vec_ki_size + i * vec_k_size + kbeg;
                        double     * dst_ptr = dst.data + j * vec_ki_size + i * vec_k_size + kbeg;
                        for (int k = 0; k < col_height; k++)
                            dst_ptr[k] = src_ptr[k];
                    }
                }
            } else {
                if (zero_guess) u.set_val(0.0);
                record[PREC] -= wall_time();
                this->prec->Mult(r, u, true);
                record[PREC] += wall_time();
            }
        } else {
            //record[AXPY] -= wall_time();
            vec_copy(r, u);
            //record[AXPY] += wall_time();
        }
        // ck_dot = vec_dot<idx_t, ksp_t, double>(u, u);
        // printf("after  prec (x,x) = %.10e\n", ck_dot);

        //record[DOT] -= wall_time();
        double old_gamma = gamma;
        tmp_loc[0] = seq_vec_dot<idx_t, ksp_t, double>(*(u.local_vector), *(r.local_vector));// γ的局部点积
        tmp_loc[1] = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));
        MPI_Allreduce(tmp_loc, tmp_glb, 2, MPI_DOUBLE, MPI_SUM, comm);
        gamma = tmp_glb[0];
        norm_r = sqrt(tmp_glb[1]);
        //record[DOT] += wall_time();

        if (my_pid == 0) printf("iter %4d   alpha %.16e   ||r||/||b|| = %.16e\n", iter, alpha, norm_r / norm_b);
        if (norm_r / norm_b <= this->rel_tol || norm_r <= this->abs_tol) {
            this->converged = 1;
            break;
        }

        beta = gamma / old_gamma;

        //record[AXPY] -= wall_time();
        vec_add(u, beta, p, p);
        //record[AXPY] += wall_time();
        //record[OPER] -= wall_time();
        this->oper->Mult(p, s, false);
        //record[OPER] += wall_time();

        //record[DOT] -= wall_time();
        eta = this->Dot(p, s);
        if (eta <= 0.0) {
            double dd = this->Dot(p, p);
            if (dd > 0.0) if (my_pid == 0) printf("WARNING: PCG: The operator is not positive definite. (Ad, d) = %.5e\n", (double)eta);
            if (eta == 0.0) break;
        }
        alpha = gamma / eta;
        //record[DOT] += wall_time();
    }

    if (pc_buf_x) delete pc_buf_x;
    if (pc_buf_b) delete pc_buf_b;
}


#endif