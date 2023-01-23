#ifndef SMG_GMRES_HPP
#define SMG_GMRES_HPP

#include "iter_solver.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename ksp_t, typename pc_t>
class GMRESSolver : public IterativeSolver<idx_t, ksp_t, pc_t> {
public:
    int restart_len = 10;

    GMRESSolver() {  }
    virtual void SetRestartlen(int num) { restart_len = num; }
    virtual void Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const ;
};

// 抄的hypre的GMRES，比较麻烦，精简了一点
template<typename idx_t, typename ksp_t, typename pc_t>
void GMRESSolver<idx_t, ksp_t, pc_t>::Mult(const par_structVector<idx_t, ksp_t> & b, par_structVector<idx_t, ksp_t> & x) const 
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(this->restart_len > 0);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);// 如果不满足，可能会有问题？从之前的向量构造的规范性上应该是满足的

    MPI_Comm comm = x.comm_pkg->cart_comm;
    int my_pid; MPI_Comm_rank(comm, &my_pid);
    if (my_pid == 0) printf("GMRES with restart = %d\n", restart_len);
    double * record = this->part_times;
    memset(record, 0.0, sizeof(double) * NUM_KRYLOV_RECORD);

    double * rs = new double [restart_len + 1];
    double *  c = new double [restart_len];
    double *  s = new double [restart_len];
    double * rs2= new double [restart_len + 1];
    double ** hh = new double * [restart_len + 1];
    for (int i = 0; i <= restart_len; i++)
        hh[i] = new double [restart_len];

    // 初始化辅助向量
    par_structVector<idx_t, ksp_t> r(x), w(x);
    r.set_halo(0.0);
    w.set_halo(0.0);
    std::allocator<par_structVector<idx_t, ksp_t> > alloc;
    par_structVector<idx_t, ksp_t> * p = alloc.allocate(restart_len + 1);
    for (int i = 0; i <= restart_len; i++) {
        alloc.construct(p + i, x);
        p[i].set_halo(0.0);
    }

    // compute initial residual
    record[OPER] -= wall_time();
    this->oper->Mult(x, p[0], false);
    record[OPER] += wall_time();
    record[AXPY] -= wall_time();
    vec_add(b, -1.0, p[0], p[0]);
    record[AXPY] += wall_time();
    record[DOT ] -= wall_time();
    double b_norm = this->Norm(b);
    double real_r_norm_old = b_norm, real_r_norm_new;
    double r_norm = this->Norm(p[0]);
    // double r_norm_0 = r_norm;
    record[DOT ] += wall_time();

    if (my_pid == 0) {
        printf("L2 norm of b: %20.16e\n", b_norm);
        printf("Initial L2 norm of residual: %20.16e\n", r_norm);
    }

    int & iter = this->final_iter = 0;
    double den_norm;
    if (b_norm > 0.0)   den_norm = b_norm;// convergence criterion |r_i|/|b| <= accuracy if |b| > 0
    else                den_norm = r_norm;// convergence criterion |r_i|/|r0| <= accuracy if |b| = 0

    /*  convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
        den_norm = |r_0| or |b|
        note: default for a_tol is 0.0, so relative residual criteria is used unless
        user specifies a_tol, or sets r_tol = 0.0, which means absolute
        tol only is checked  */
    double epsilon;
    epsilon = std::max(this->abs_tol, this->rel_tol * den_norm);

    // so now our stop criteria is |r_i| <= epsilon
    if (my_pid == 0) {
        if (b_norm > 0.0) {
            printf("=========================================================\n\n");
            printf("Iters             resid.norm               rel.res.norm  \n");
            printf("------       -------------------      --------------------\n");
        } else {
            printf("=============================================\n\n");
            printf("Iters     resid.norm     \n");
            printf("------    ------------    \n");
        }
    }

    /* once the rel. change check has passed, we do not want to check it again */
    // bool rel_change_passed = false;
    while (iter < this->max_iter) {
        /* initialize first term of hessenberg system */
        rs[0] = r_norm;
        if (r_norm == 0.0) {
            this->converged = 1;
            goto finish;
        }

        /* see if we are already converged and 
           should print the final norm and exit */
        if (r_norm <= epsilon) {
            record[OPER] -= wall_time();
            this->oper->Mult(x, r, false);
            record[OPER] += wall_time();
            record[AXPY] -= wall_time();
            vec_add(b, -1.0, r, r);
            record[AXPY] += wall_time();
            record[DOT ] -= wall_time();
            r_norm = this->Norm(r);
            record[DOT ] += wall_time();
            if (r_norm <= epsilon) {
                if (my_pid == 0)
                    printf("\nFinal L2 norm of residual: %20.16e\n\n", r_norm);
                break;
            } else {
                if (my_pid == 0) 
                    printf("false convergence\n");
            }
        }

        record[AXPY] -= wall_time();
        vec_scale(1.0 / r_norm, p[0]);
        record[AXPY] += wall_time();
        int i = 0;
        while (i < restart_len && iter < this->max_iter) {
            i++;
            iter++;

            if (this->prec) {
                record[AXPY] -= wall_time();
                r.set_val(0.0);
                record[AXPY] += wall_time();
                record[PREC] -= wall_time();
                this->prec->Mult(p[i-1], r, true);
                record[PREC] += wall_time();
            }
            else {
                record[AXPY] -= wall_time();
                vec_copy(p[i-1], r);
                record[AXPY] += wall_time();
            }
            record[OPER] -= wall_time();
            this->oper->Mult(r, p[i], false);
            record[OPER] += wall_time();

            /* modified Gram_Schmidt */
            for (int j = 0; j < i; j++) {
                record[DOT ] -= wall_time();
                hh[j][i-1] = this->Dot(p[j], p[i]);
                record[DOT ] += wall_time();
                record[AXPY] -= wall_time();
                vec_add(p[i], -hh[j][i-1], p[j], p[i]);
                record[AXPY] += wall_time();
            }
            record[DOT ] -= wall_time();
            double t = this->Norm(p[i]);
            hh[i][i-1] = t;
            record[DOT ] += wall_time();
            record[AXPY] -= wall_time();
            if (t != 0.0) vec_scale(1.0 / t, p[i]);
            record[AXPY] += wall_time();
            /* done with modified Gram_schmidt and Arnoldi step.
                update factorization of hh */
            for (int j = 1; j < i; j++) {
                double t = hh[j-1][i-1];
                hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1] * t;
                hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
            }
            t = hh[i][i-1] * hh[i][i-1];
            t += hh[i-1][i-1] * hh[i-1][i-1];
            double gamma = sqrt(t);
            if (gamma == 0.0) gamma = 1.0e-16;
            c[i-1] = hh[i-1][i-1]/gamma;
            s[i-1] = hh[i][i-1]/gamma;
            rs[i] = -hh[i][i-1]*rs[i-1];
            rs[i]/=  gamma;
            rs[i-1] = c[i-1]*rs[i-1];
            /* determine residual norm */
            hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
            r_norm = fabs(rs[i]);

            if (my_pid == 0) {
                if (b_norm > 0.0)
                    printf("%5d    %.16e   %.16e\n", iter, r_norm, r_norm/b_norm);
                else
                    printf("%5d    %.16e\n", iter, r_norm);
            }

            if (r_norm <= epsilon) break;
        }/*** end of restart cycle ***/

        /* now compute solution, first solve upper triangular system */
        rs[i-1] = rs[i-1]/hh[i-1][i-1];
        for (int k = i-2; k >= 0; k--)
        {
            double t = 0.0;
            for (int j = k+1; j < i; j++) {
                t -= hh[k][j]*rs[j];
            }
            t += rs[k];
            rs[k] = t/hh[k][k];
        }
        record[AXPY] -= wall_time();
        vec_mul_by_scalar(rs[i-1], p[i-1], w);
        for (int j = i-2; j >= 0; j--)
            vec_add(w, rs[j], p[j], w);
        record[AXPY] += wall_time();
        if (this->prec) {
            record[AXPY] -= wall_time();
            r.set_val(0.0);
            record[AXPY] += wall_time();
            record[PREC] -= wall_time();
            this->prec->Mult(w, r, true);
            record[PREC] += wall_time();
        }
        else {
            record[AXPY] -= wall_time();
            vec_copy(w, r);
            record[AXPY] += wall_time();
        }
        /* update current solution x (in x) */
        record[AXPY] -= wall_time();
        vec_add(x, 1.0, r, x);
        record[AXPY] += wall_time();

        /* check for convergence by evaluating the actual residual */
        if (r_norm <= epsilon) {
            record[OPER] -= wall_time();
            this->oper->Mult(x, r, false);
            record[OPER] += wall_time();
            record[AXPY] -= wall_time();
            vec_add(b, -1.0, r, r);
            record[AXPY] += wall_time();
            record[DOT ] -= wall_time();
            real_r_norm_new = r_norm = this->Norm(r);
            record[DOT ] += wall_time();

            if (r_norm <= epsilon) {
                if (my_pid == 0)
                    printf("\nFinal L2 norm of residual: %20.16e\n\n", r_norm);
                goto finish;
            } else {/* conv. has not occurred, according to true residual */
                /* exit if the real residual norm has not decreased */
                if (real_r_norm_new >= real_r_norm_old) {
                    if (my_pid == 0)
                        printf("\nFinal L2 norm of residual: %20.16e\n\n", r_norm);
                    this->converged = 1;
                    break;
                }

                /* report discrepancy between real/GMRES residuals and restart */
                if (my_pid == 0)
                    printf("false convergence 2, L2 norm of residual: %20.16e\n", r_norm);

                record[AXPY] -= wall_time();
                vec_copy(r, p[0]);
                record[AXPY] += wall_time();
                i = 0;
                real_r_norm_old = real_r_norm_new;
            }
        }/* end of convergence check */

        /* compute residual vector and continue loop */
	    for (int j=i ; j > 0; j--) {
            rs[j-1] = -s[j-1]*rs[j];
            rs[j] = c[j-1]*rs[j];
    	}
        
        record[AXPY] -= wall_time();
        if (i) vec_add(p[i], rs[i] - 1.0, p[i], p[i]);

        for (int j=i-1 ; j > 0; j--)
            vec_add(p[i], rs[j], p[j], p[i]);
        
        if (i) {
            vec_add(p[0], rs[0] - 1.0, p[0], p[0]);
            vec_add(p[0], 1.0, p[i], p[0]);
        }
        record[AXPY] += wall_time();

        if (my_pid == 0) printf("Restarting...\n");
    }/* END of iteration while loop */

finish:
    delete c; delete s; delete rs; delete rs2;
    c = s = rs = rs2 = nullptr;
    for (int i = 0; i <= restart_len; i++) {
        delete hh[i]; hh[i] = nullptr;
    }
    delete hh; hh = nullptr;
    for (int i = 0; i <= restart_len; i++) alloc.destroy(p + i);
    alloc.deallocate(p, restart_len + 1);
}

#endif