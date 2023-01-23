#ifndef SMG_GCR_HPP
#define SMG_GCR_HPP

#include "iter_solver.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename ksp_t, typename pc_t>
class GCRSolver : public IterativeSolver<idx_t, ksp_t, pc_t> {
public:
    // iter_max+1就是重启长度，闭区间[0, iter_max]
    int inner_iter_max = 10;

    GCRSolver() {  }
    virtual void SetInnerIterMax(int num) { inner_iter_max = num; }

    // 求解以b为右端向量的方程组，x为返回的近似解
    virtual void Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const ;
};

template<typename idx_t, typename ksp_t, typename pc_t>
void GCRSolver<idx_t, ksp_t, pc_t>::Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const 
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(this->inner_iter_max > 0);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);// 如果不满足，可能会有问题？从之前的向量构造的规范性上应该是满足的

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);
    if (my_pid == 0) printf("GCR with inner_iter_max = %d\n", inner_iter_max);
    double * record = this->part_times;
    memset(record, 0.0, sizeof(double) * NUM_KRYLOV_RECORD);

    // 初始化辅助向量
    std::allocator<par_structVector<idx_t, ksp_t> > alloc;
    par_structVector<idx_t, ksp_t> * p = alloc.allocate(inner_iter_max), 
                                   *ap = alloc.allocate(inner_iter_max);

    for (int i = 0; i < inner_iter_max; i++) {
        alloc.construct( p + i, x); // vec_set_val( p[i], 0.0);// 不需要清零
        alloc.construct(ap + i, x); // vec_set_val(ap[i], 0.0);
        p[i].set_halo(0.0);
        ap[i].set_halo(0.0);
    }

    par_structVector<idx_t, ksp_t> r(x), ar(x), gcr(x);
    r.set_halo(0.0);
    ar.set_halo(0.0);
    gcr.set_halo(0.0);

    record[DOT] -= wall_time();
    double b_dot = this->Dot(b, b);
    record[DOT] += wall_time();

    // 初始化辅助标量
    double res = 0.0;
    double aps[inner_iter_max], 
            c1[inner_iter_max], c2[inner_iter_max], beta[inner_iter_max];

    // 开始计算
    record[OPER] -= wall_time();
    this->oper->Mult(x, r, false);
    record[OPER] += wall_time();
    record[AXPY] -= wall_time();
    vec_add(b, -1.0, r, r);
    record[AXPY] += wall_time();
    record[DOT] -= wall_time();
    res = this->Dot(r, r);
    record[DOT] += wall_time();

    if (my_pid == 0) printf("begin of gcr %20.16e\n", res);

    // 在这里做预条件，r => M^{-1}*r，然后用M^{-1}*r去跟矩阵A乘
    if (this->prec) {
        record[PREC] -= wall_time();
        this->prec->Mult(r, gcr, true);
        record[PREC] += wall_time();
    } else {
        record[AXPY] -= wall_time();
        vec_copy(r, gcr);
        record[AXPY] += wall_time();
    }
    record[OPER] -= wall_time();
    this->oper->Mult(gcr, ar, false);// ar = A*r
    record[OPER] += wall_time();

    // p[0] = r, ap[0] = ar
    record[AXPY] -= wall_time();
    vec_copy(gcr,  p[0]);
    vec_copy( ar, ap[0]);
    record[AXPY] += wall_time();

    // 主循环
    for (int mm = 0; mm < this->max_iter; mm++) {
        int m = mm % (inner_iter_max - 1);
        record[AXPY] -= wall_time();
        if (m == 0 && mm > 0) {// 内迭代归零时，开始下一轮内迭代，拷贝上一轮的最后一个值
            vec_copy(  p[inner_iter_max - 1],  p[0] );
            vec_copy( ap[inner_iter_max - 1], ap[0] );
        }
        record[AXPY] += wall_time();

        record[DOT] -= wall_time();
        c1[0] = seq_vec_dot<idx_t, ksp_t, double>(*(    r.local_vector), *(ap[m].local_vector));
        c1[1] = seq_vec_dot<idx_t, ksp_t, double>(*(ap[m].local_vector), *(ap[m].local_vector));
        MPI_Allreduce(c1, c2, 2, MPI_DOUBLE, MPI_SUM, comm);
        double ac = c2[0] / c2[1];
        aps[m]    = c2[1];
        record[DOT] += wall_time();

        record[AXPY] -= wall_time();
        vec_add(x,  ac,  p[m], x);// 更新解向量
        vec_add(r, -ac, ap[m], r);// 更新残差向量
        record[AXPY] += wall_time();
        record[DOT] -= wall_time();
        double loc_err, glb_err;
        loc_err = seq_vec_dot<idx_t, ksp_t, double>(*(r.local_vector), *(r.local_vector));// 判敛的全局通信可以放到后面的和c1、c2一起通
        record[DOT] += wall_time();

        // 这里做预条件，r => M^{-1}*r，然后用M^{-1}*r去跟矩阵A乘，得到ar
        if (this->prec) {
            record[PREC] -= wall_time();
            this->prec->Mult(r, gcr, true);
            record[PREC] += wall_time();
        } else {
            record[AXPY] -= wall_time();
            vec_copy(r, gcr);
            record[AXPY] += wall_time();
        }
        record[OPER] -= wall_time();
        this->oper->Mult(gcr, ar, false);
        record[OPER] += wall_time();
        record[DOT] -= wall_time();
        for (int l = 0; l <= m; l++)
            c1[l] = seq_vec_dot<idx_t, ksp_t, double>(*(ar.local_vector), *((ap[l]).local_vector));
        c1[m + 1] = loc_err;
        MPI_Allreduce(c1, c2, m + 2, MPI_DOUBLE, MPI_SUM, comm);// 通信更新[0,1,...,m]共计m+1个数

        glb_err = c2[m + 1];
        double _rel = glb_err / b_dot;
        record[DOT] += wall_time();

        if (my_pid == 0) printf("  res of gcr %20.16e at %3d iter (r,r)/(b,b) = %.16e\n", glb_err, mm, _rel);
        if (glb_err <= this->abs_tol || _rel <= this->rel_tol || mm == this->max_iter) goto finish;

        record[AXPY] -= wall_time();
        // 计算beta并更新下一次内迭代的向量
        for (int l = 0; l <= m; l++)
            beta[l] = - c2[l] / aps[l];
        vec_copy(gcr,  p[m+1]);
        vec_copy( ar, ap[m+1]);
        for (int l = 0; l <= m; l++) {// 根据[0,1,...,m]共计m+1个向量更新下一次内迭代要使用的序号为m+1的
            vec_add(  p[m+1], beta[l],  p[l],  p[m+1]);
            vec_add( ap[m+1], beta[l], ap[l], ap[m+1]);
        }
        record[AXPY] += wall_time();
    }

finish:
    // 清理数据
    for (int i = 0; i < inner_iter_max; i++) {
        alloc.destroy( p + i);
        alloc.destroy(ap + i);
    }
    alloc.deallocate( p, inner_iter_max);
    alloc.deallocate(ap, inner_iter_max);
}


#endif