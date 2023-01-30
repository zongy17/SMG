#ifndef SMG_GMG_PROLONG_HPP
#define SMG_GMG_PROLONG_HPP

#include "GMG_types.hpp"

template<typename idx_t, typename data_t>
class Interpolator {
private:
    // 插值算子不需要根据方程是否为标准化形式而改变
    const PROLONG_TYPE type;
    data_t a0, a1, a2, a3;
public:
    Interpolator(PROLONG_TYPE type): type(type) { 
        setup_weights();
    }
    virtual void setup_weights();
    virtual void apply(const par_structVector<idx_t, data_t> & par_coar_vec,
        par_structVector<idx_t, data_t> & par_fine_vec, const COAR_TO_FINE_INFO<idx_t> & info);
    virtual ~Interpolator() {  }
};

template<typename idx_t, typename data_t>
void Interpolator<idx_t, data_t>::setup_weights() {
    switch (type)
    {
    case Plg_linear_4cell:
        a0 = 1.0;
        break;
    case Plg_linear_8cell:
        //    a0----a0
        //    /|     /|
        //  a0----a0|
        //   | a0--|a0
        //   |/     |/
        //  a0----a0
        a0 = 1.0;
        break;
    case Plg_linear_64cell:
        /*      a3----a2----a2----a3
                /     /    /     /
              a2----a1----a1----a2
              /     /     /     /
            a2----a1----a1----a2
            /     /     /     /
          a3----a2----a2----a3
                 
                a2----a1----a1----a2
                /     /    /     /
              a1----a0----a0----a1
              /     /     /     /
            a1----a0----a0----a1
            /     /     /     /
          a2----a1----a1----a2

                 a2----a1----a1----a2
                /     /    /     /
              a1----a0----a0----a1
              /     /     /     /
            a1----a0----a0----a1
            /     /     /     /
          a2----a1----a1----a2

                 a3----a2----a2----a3
                /     /    /     /
              a2----a1----a1----a2
              /     /     /     /
            a2----a1----a1----a2
            /     /     /     /
          a3----a2----a2----a3
        */
        a0 = 27.0 / 64.0;
        a1 =  9.0 / 64.0;
        a2 =  3.0 / 64.0;
        a3 =  1.0 / 64.0;
        break;
    default:
        printf("Invalid interpolator type!\n");
        MPI_Abort(MPI_COMM_WORLD, -20221105);
    }
}

#define C_VECIDX(k, i, j) (k) + (i) * c_k_size + (j) * c_ki_size
#define F_VECIDX(k, i, j) (k) + (i) * f_k_size + (j) * f_ki_size

template<typename idx_t, typename data_t>
void Interpolator<idx_t, data_t>::apply(const par_structVector<idx_t, data_t> & par_coar_vec,
        par_structVector<idx_t, data_t> & par_fine_vec, const COAR_TO_FINE_INFO<idx_t> & info)
{
    const seq_structVector<idx_t, data_t> & c_vec = *(par_coar_vec.local_vector);
          seq_structVector<idx_t, data_t> & f_vec = *(par_fine_vec.local_vector);
    CHECK_HALO(f_vec, c_vec);
    /* 插值计算的基本单位：与一个粗cell对应的8个细cell
            F----F
          F----F |
          | |C | |
          | F--|-F
          F----F
    */
    const idx_t hx = f_vec.halo_x         , hy = f_vec.halo_y         , hz = f_vec.halo_z         ;
    const idx_t bx = info.fine_base_idx[0], by = info.fine_base_idx[1], bz = info.fine_base_idx[2];
    const idx_t sx = info.stride[0]       , sy = info.stride[1]       , sz = info.stride[2]       ;

    const idx_t cibeg = c_vec.halo_x, ciend = cibeg + c_vec.local_x,
                cjbeg = c_vec.halo_y, cjend = cjbeg + c_vec.local_y,
                ckbeg = c_vec.halo_z, ckend = ckbeg + c_vec.local_z;
    const idx_t c_k_size = c_vec.slice_k_size, c_ki_size = c_vec.slice_ki_size;
    const idx_t f_k_size = f_vec.slice_k_size, f_ki_size = f_vec.slice_ki_size;
    data_t * f_data = f_vec.data;
    const data_t * c_data = c_vec.data;

    if (type == Plg_linear_64cell) {
        par_coar_vec.update_halo();
        
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = hz + bz + (ck - ckbeg) * sz;
            f_data[F_VECIDX(fk  , fi  , fj  )] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck-1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci-1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj-1)])
                + a2 * (c_data[C_VECIDX(ck-1, ci-1, cj  )] + c_data[C_VECIDX(ck  , ci-1, cj-1)] + c_data[C_VECIDX(ck-1, ci  , cj-1)])
                + a3 *  c_data[C_VECIDX(ck-1, ci-1, cj-1)];
            f_data[F_VECIDX(fk+1, fi  , fj  )] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck+1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci-1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj-1)])
                + a2 * (c_data[C_VECIDX(ck+1, ci-1, cj  )] + c_data[C_VECIDX(ck  , ci-1, cj-1)] + c_data[C_VECIDX(ck+1, ci  , cj-1)])
                + a3 *  c_data[C_VECIDX(ck+1, ci-1, cj-1)];
            f_data[F_VECIDX(fk  , fi+1, fj  )] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck-1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci+1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj-1)])
                + a2 * (c_data[C_VECIDX(ck-1, ci+1, cj  )] + c_data[C_VECIDX(ck  , ci+1, cj-1)] + c_data[C_VECIDX(ck-1, ci  , cj-1)])
                + a3 *  c_data[C_VECIDX(ck-1, ci+1, cj-1)];
            f_data[F_VECIDX(fk+1, fi+1, fj  )] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck+1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci+1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj-1)])
                + a2 * (c_data[C_VECIDX(ck+1, ci+1, cj  )] + c_data[C_VECIDX(ck  , ci+1, cj-1)] + c_data[C_VECIDX(ck+1, ci  , cj-1)])
                + a3 *  c_data[C_VECIDX(ck+1, ci+1, cj-1)];
            // ------------------------------------------------------------ //
            f_data[F_VECIDX(fk  , fi  , fj+1)] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck-1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci-1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj+1)])
                + a2 * (c_data[C_VECIDX(ck-1, ci-1, cj  )] + c_data[C_VECIDX(ck  , ci-1, cj+1)] + c_data[C_VECIDX(ck-1, ci  , cj+1)])
                + a3 *  c_data[C_VECIDX(ck-1, ci-1, cj+1)];
            f_data[F_VECIDX(fk+1, fi  , fj+1)] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck+1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci-1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj+1)])
                + a2 * (c_data[C_VECIDX(ck+1, ci-1, cj  )] + c_data[C_VECIDX(ck  , ci-1, cj+1)] + c_data[C_VECIDX(ck+1, ci  , cj+1)])
                + a3 *  c_data[C_VECIDX(ck+1, ci-1, cj+1)];
            f_data[F_VECIDX(fk  , fi+1, fj+1)] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck-1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci+1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj+1)])
                + a2 * (c_data[C_VECIDX(ck-1, ci+1, cj  )] + c_data[C_VECIDX(ck  , ci+1, cj+1)] + c_data[C_VECIDX(ck-1, ci  , cj+1)])
                + a3 *  c_data[C_VECIDX(ck-1, ci+1, cj+1)];
            f_data[F_VECIDX(fk+1, fi+1, fj+1)] = 
                + a0 *  c_data[C_VECIDX(ck  , ci  , cj  )] 
                + a1 * (c_data[C_VECIDX(ck+1, ci  , cj  )] + c_data[C_VECIDX(ck  , ci+1, cj  )] + c_data[C_VECIDX(ck  , ci  , cj+1)])
                + a2 * (c_data[C_VECIDX(ck+1, ci+1, cj  )] + c_data[C_VECIDX(ck  , ci+1, cj+1)] + c_data[C_VECIDX(ck+1, ci  , cj+1)])
                + a3 *  c_data[C_VECIDX(ck+1, ci+1, cj+1)];
        }
    }
    else if (type == Plg_linear_8cell) {
#ifdef PROFILE
        double t, mint, maxt;
        int test_cnt = 10;
        int my_pid; MPI_Comm_rank(par_fine_vec.comm_pkg->cart_comm, &my_pid);
        int num_procs; MPI_Comm_size(par_fine_vec.comm_pkg->cart_comm, &num_procs);
        MPI_Barrier(par_fine_vec.comm_pkg->cart_comm);
        t = wall_time();
        for (int te = 0; te < test_cnt; te++) {
#endif
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = hz + bz + (ck - ckbeg) * sz;
            data_t coar_val = c_data[C_VECIDX(ck  , ci  , cj  )];
            f_data[F_VECIDX(fk  , fi  , fj  )] = coar_val;
            f_data[F_VECIDX(fk+1, fi  , fj  )] = coar_val;
            f_data[F_VECIDX(fk  , fi+1, fj  )] = coar_val;
            f_data[F_VECIDX(fk+1, fi+1, fj  )] = coar_val;
            // ------------------------------------------------------------ //
            f_data[F_VECIDX(fk  , fi  , fj+1)] = coar_val;
            f_data[F_VECIDX(fk+1, fi  , fj+1)] = coar_val;
            f_data[F_VECIDX(fk  , fi+1, fj+1)] = coar_val;
            f_data[F_VECIDX(fk+1, fi+1, fj+1)] = coar_val; 
        }

        // // 这两种写法没差别，但对64cell的插值可能会有差别
        // #pragma omp parallel for collapse(3) schedule(static)
        // for (idx_t cj = cjbeg; cj < cjend; cj++)
        // for (idx_t ci = cibeg; ci < ciend; ci++)
        // for (idx_t ck = ckbeg; ck < ckend; ck++) {
        //     const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
        //     const idx_t fi = hx + bx + (ci - cibeg) * sx;
        //     const idx_t fk = hz + bz + (ck - ckbeg) * sz;
        //     data_t coar_val = c_data[C_VECIDX(ck  , ci  , cj  )];
        //     idx_t d00 = F_VECIDX(fk, fi, fj), d10 = F_VECIDX(fk, fi+1, fj), d01 = F_VECIDX(fk, fi, fj+1), d11 = F_VECIDX(fk, fi+1, fj+1);
        //     f_data[d00  ] = coar_val;
        //     f_data[d00+1] = coar_val;
        //     f_data[d10  ] = coar_val;
        //     f_data[d10+1] = coar_val;
        //     // ------------------------------------------------------------ //
        //     f_data[d01  ] = coar_val;
        //     f_data[d01+1] = coar_val;
        //     f_data[d11  ] = coar_val;
        //     f_data[d11+1] = coar_val; 
        // }
#ifdef PROFILE
        }
        t = wall_time() - t; t /= test_cnt;
        MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, par_fine_vec.comm_pkg->cart_comm);
        MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, par_fine_vec.comm_pkg->cart_comm);
        if (my_pid == 0) {
            double bytes = (f_vec.local_x) * (f_vec.local_y) * (f_vec.local_z) * sizeof(data_t);// 细网格向量
            bytes       += (c_vec.local_x) * (c_vec.local_y) * (c_vec.local_z) * sizeof(data_t);// 粗网格向量
            bytes = bytes * num_procs / (1024*1024*1024);// total GB
            printf("Prlg data %ld total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), bytes, mint, maxt, bytes/maxt, bytes/mint);
        }
#endif
    }
    else if (type == Plg_linear_4cell) {
        assert(f_k_size == c_k_size);
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = ck;// same
            data_t coar_val = c_data[C_VECIDX(ck  , ci  , cj  )];
            f_data[F_VECIDX(fk  , fi  , fj  )] = coar_val;
            f_data[F_VECIDX(fk  , fi+1, fj  )] = coar_val;
            // ------------------------------------------------------------ //
            f_data[F_VECIDX(fk  , fi  , fj+1)] = coar_val;
            f_data[F_VECIDX(fk  , fi+1, fj+1)] = coar_val;
        }
    }
    else {
        assert(false);
    }
}

#undef C_VECIDX
#undef F_VECIDX

#endif