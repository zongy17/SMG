#ifndef SMG_TRUNCATE_COPY_HPP
#define SMG_TRUNCATE_COPY_HPP

#include "Solver_ls.hpp"


template<typename idx_t, typename hp_t, typename lp_t>
void copy_w_trunc_LineSolver(const LineSolver<idx_t, hp_t, hp_t, hp_t> & src_h, LineSolver<idx_t, lp_t, hp_t, hp_t> & dst_l)
{
    // 之前应该两者都已经Setop好
    CHECK_INOUT_DIM(dst_l, src_h);
    dst_l.weight = src_h.weight;

    // dst_l.cart_comm = src_h.cart_comm;
    // assert(dst_l.my_pid == src_h.my_pid && dst_l.next_id == src_h.next_id && dst_l.prev_id == src_h.prev_id);
    // assert(dst_l.line_dir == src_h.line_dir);

    // 注意先按高精度setup了三对角系数，再行截断
    assert(src_h.tri_solver[0]->Get_n_size() == dst_l.tri_solver[0]->Get_n_size());
    const idx_t n = src_h.tri_solver[0]->Get_n_size();
    assert(dst_l.num_solvers == src_h.num_solvers);

    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0) printf("\033[1;31mcopy_w_trunc_LineSolver\033[0m\n");

    for (idx_t i = 0; i < dst_l.num_solvers; i++) {
        assert(dst_l.tri_solver[i]->periodic == src_h.tri_solver[i]->periodic);
        lp_t * a_l = dst_l.tri_solver[i]->Get_a(); hp_t * a_h = src_h.tri_solver[i]->Get_a();
        lp_t * b_l = dst_l.tri_solver[i]->Get_b(); hp_t * b_h = src_h.tri_solver[i]->Get_b();
        lp_t * c_l = dst_l.tri_solver[i]->Get_c(); hp_t * c_h = src_h.tri_solver[i]->Get_c();
        for (idx_t j = 0; j < n; j++) {
            a_l[j] = (lp_t) a_h[j];
            b_l[j] = (lp_t) b_h[j];
            c_l[j] = (lp_t) c_h[j];
        }
    }
}

template<typename idx_t, typename hp_t, typename lp_t>
void copy_w_trunc_PILU(const PlaneILU<idx_t, hp_t, hp_t, hp_t> & src_h, PlaneILU<idx_t, lp_t, hp_t, hp_t> & dst_l)
{
    // 之前应该两者都已经Setop好
    CHECK_INOUT_DIM(dst_l, src_h);
    dst_l.weight = src_h.weight;

    assert(dst_l.plane_dir == src_h.plane_dir);

    assert(dst_l.type == src_h.type);
    assert(dst_l.num_stencil == src_h.num_stencil);
    assert(dst_l.cw_type == src_h.cw_type);
    assert(dst_l.setup_called && src_h.setup_called);
    assert(dst_l.outer_dim == src_h.outer_dim && dst_l.midle_dim == src_h.midle_dim && dst_l.inner_dim == src_h.inner_dim);
    assert(dst_l.lnz == src_h.lnz && dst_l.rnz == src_h.rnz);

    // 注意先按高精度setup了三对角系数，再行截断
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0) printf("\033[1;31mcopy_w_trunc_PILU\033[0m\n");
    idx_t tot_len;
    // L
    tot_len = dst_l.outer_dim * dst_l.midle_dim * dst_l.inner_dim * dst_l.lnz;
    for (idx_t i = 0; i < tot_len; i++)
        dst_l.L[i] = src_h.L[i];
    // U
    tot_len = dst_l.outer_dim * dst_l.midle_dim * dst_l.inner_dim * dst_l.rnz;
    for (idx_t i = 0; i < tot_len; i++)
        dst_l.U[i] = src_h.U[i];
}

template<typename idx_t, typename hp_t, typename lp_t>
void copy_w_trunc_BILU(const BlockILU<idx_t, hp_t, hp_t, hp_t> & src_h, BlockILU<idx_t, lp_t, hp_t, hp_t> & dst_l)
{
    // 之前应该两者都已经Setop好
    CHECK_INOUT_DIM(dst_l, src_h);
    dst_l.weight = src_h.weight;

    assert(dst_l.type == src_h.type);
    assert(dst_l.num_stencil == src_h.num_stencil);
    assert(dst_l.setup_called && src_h.setup_called);
    assert(dst_l.outer_dim == src_h.outer_dim && dst_l.midle_dim == src_h.midle_dim && dst_l.inner_dim == src_h.inner_dim);
    assert(dst_l.lnz == src_h.lnz && dst_l.rnz == src_h.rnz);

    // 注意先按高精度setup了，再行截断
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0) printf("\033[1;31mcopy_w_trunc_PILU\033[0m\n");
    idx_t tot_len;
    // L
    tot_len = dst_l.outer_dim * dst_l.midle_dim * dst_l.inner_dim * dst_l.lnz;
    for (idx_t i = 0; i < tot_len; i++)
        dst_l.L[i] = src_h.L[i];
    // U
    tot_len = dst_l.outer_dim * dst_l.midle_dim * dst_l.inner_dim * dst_l.rnz;
    for (idx_t i = 0; i < tot_len; i++)
        dst_l.U[i] = src_h.U[i];
}

template<typename idx_t, typename hp_t, typename lp_t>
void copy_w_trunc_LU(const DenseLU<idx_t, hp_t, hp_t, hp_t> & src_h, DenseLU<idx_t, lp_t, hp_t, hp_t> & dst_l)
{
    CHECK_INOUT_DIM(dst_l, src_h);
    dst_l.weight = src_h.weight;

    assert(dst_l.type == src_h.type);
    assert(dst_l.num_stencil == src_h.num_stencil);
    assert(dst_l.setup_called && src_h.setup_called);

    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0) printf("\033[1;31mcopy_w_trunc_LU\033[0m\n");

    // 拷贝分解后的L和U
    assert(dst_l.global_dof == src_h.global_dof);
    idx_t lnz = (dst_l.num_stencil - 1) >> 1;
    idx_t rnz = (dst_l.num_stencil + 1) >> 1;
    for (idx_t i = 0; i < dst_l.global_dof; i++) {
        for (idx_t d = 0; d < lnz; d++)
            dst_l.l_data[i * lnz + d] = src_h.l_data[i * lnz + d];
        for (idx_t d = 0; d < rnz; d++)
            dst_l.u_data[i * rnz + d] = src_h.u_data[i * rnz + d];
    }
}


#endif