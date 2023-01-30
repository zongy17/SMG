#ifndef SMG_GMG_RESTRICT_HPP
#define SMG_GMG_RESTRICT_HPP

#include "GMG_types.hpp"

template<typename idx_t, typename data_t>
class Restrictor {
protected:
    data_t a0, a1, a2, a3;
    const RESTRICT_TYPE type;
public:
    Restrictor(RESTRICT_TYPE type): type(type)  { 
        setup_weights();
    }
    virtual void setup_weights();
    virtual void apply(const par_structVector<idx_t, data_t> & par_fine_vec, 
        par_structVector<idx_t, data_t> & par_coar_vec, const COAR_TO_FINE_INFO<idx_t> & info);
    virtual ~Restrictor() {  }
};

template<typename idx_t, typename data_t>
void Restrictor<idx_t, data_t>::setup_weights() {
    switch (type)
    {
    case Rst_4cell:
        a0 = 0.25;
        break;
    case Rst_8cell:
        //    a00----a00
        //    /|     /|
        //  a00----a00|
        //   | a00--|a00
        //   |/     |/
        //  a00----a00
        a0 = 0.125;
        break;
    case Rst_64cell:
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
        a0 = 0.052734375;
        a1 = 0.017578125;
        a2 = 0.005859375;
        a3 = 0.001953125;
        break;
    default:
        printf("Invalid restrictor type!\n");
        MPI_Abort(MPI_COMM_WORLD, -20221105);
    }
}

template<typename idx_t, typename data_t>
void Restrictor<idx_t, data_t>::apply(const par_structVector<idx_t, data_t> & par_fine_vec, 
        par_structVector<idx_t, data_t> & par_coar_vec, const COAR_TO_FINE_INFO<idx_t> & info) 
{
    const seq_structVector<idx_t, data_t> & f_vec = *(par_fine_vec.local_vector);
          seq_structVector<idx_t, data_t> & c_vec = *(par_coar_vec.local_vector);
    CHECK_HALO(f_vec, c_vec);
    const idx_t hx = f_vec.halo_x         , hy = f_vec.halo_y         , hz = f_vec.halo_z         ;
    const idx_t bx = info.fine_base_idx[0], by = info.fine_base_idx[1], bz = info.fine_base_idx[2];
    const idx_t sx = info.stride[0]       , sy = info.stride[1]       , sz = info.stride[2]       ;

    const idx_t cibeg = c_vec.halo_x, ciend = cibeg + c_vec.local_x,
                cjbeg = c_vec.halo_y, cjend = cjbeg + c_vec.local_y,
                ckbeg = c_vec.halo_z, ckend = ckbeg + c_vec.local_z;
    const idx_t c_k_size = c_vec.slice_k_size, c_ki_size = c_vec.slice_ki_size;
    const idx_t f_k_size = f_vec.slice_k_size, f_ki_size = f_vec.slice_ki_size;
    const data_t * f_data = f_vec.data;
    data_t * c_data = c_vec.data;

    

#define C_VECIDX(k, i, j) (k) + (i) * c_k_size + (j) * c_ki_size
#define F_VECIDX(k, i, j) (k) + (i) * f_k_size + (j) * f_ki_size

    if (type == Rst_8cell) {
        if (bx < 0 || by < 0 || bz < 0)// 此时需要引用到不在自己进程负责范围内的数据
            par_fine_vec.update_halo();

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
            c_data[C_VECIDX(ck, ci, cj)] = a0 * (
                f_data[F_VECIDX(fk  , fi  , fj  )]
            +   f_data[F_VECIDX(fk+1, fi  , fj  )]
            +   f_data[F_VECIDX(fk  , fi+1, fj  )]
            +   f_data[F_VECIDX(fk+1, fi+1, fj  )]
            +   f_data[F_VECIDX(fk  , fi  , fj+1)]
            +   f_data[F_VECIDX(fk+1, fi  , fj+1)]
            +   f_data[F_VECIDX(fk  , fi+1, fj+1)]
            +   f_data[F_VECIDX(fk+1, fi+1, fj+1)]
            );
        }
#ifdef PROFILE
        }
        t = wall_time() - t; t /= test_cnt;
        MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, par_fine_vec.comm_pkg->cart_comm);
        MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, par_fine_vec.comm_pkg->cart_comm);
        if (my_pid == 0) {
            double bytes = (f_vec.local_x) * (f_vec.local_y) * (f_vec.local_z) * sizeof(data_t);// 细网格向量
            bytes       += (c_vec.local_x) * (c_vec.local_y) * (c_vec.local_z) * sizeof(data_t);// 粗网格向量
            bytes = bytes * num_procs / (1024*1024*1024);// total GB
            printf("Rstr data %ld total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), bytes, mint, maxt, bytes/maxt, bytes/mint);
        }
#endif
    }
    else if (type == Rst_4cell) {
        if (bx < 0 || by < 0)
            par_fine_vec.update_halo();
        
        assert(c_k_size == f_k_size);
        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {// same as fk's range
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = ck;
            c_data[C_VECIDX(ck, ci, cj)] = a0 * (
                f_data[F_VECIDX(fk  , fi  , fj  )]
            +   f_data[F_VECIDX(fk  , fi+1, fj  )]
            +   f_data[F_VECIDX(fk  , fi  , fj+1)]
            +   f_data[F_VECIDX(fk  , fi+1, fj+1)]
            );
        }
    }
    else if (type == Rst_64cell) {
        par_fine_vec.update_halo();

        #pragma omp parallel for collapse(3) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++)
        for (idx_t ck = ckbeg; ck < ckend; ck++) {
            const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
            const idx_t fi = hx + bx + (ci - cibeg) * sx;
            const idx_t fk = hz + bz + (ck - ckbeg) * sz;
            c_data[C_VECIDX(ck, ci, cj)] = 
            +   a0 * (
                    f_data[F_VECIDX(fk  , fi  , fj  )]// 最邻近的左下前方的subcell
                +   f_data[F_VECIDX(fk+1, fi  , fj  )]
                +   f_data[F_VECIDX(fk  , fi+1, fj  )]
                +   f_data[F_VECIDX(fk+1, fi+1, fj  )]
                +   f_data[F_VECIDX(fk  , fi  , fj+1)]
                +   f_data[F_VECIDX(fk+1, fi  , fj+1)]
                +   f_data[F_VECIDX(fk  , fi+1, fj+1)]
                +   f_data[F_VECIDX(fk+1, fi+1, fj+1)] )
            +   a1 * (
                    f_data[F_VECIDX(fk-1, fi  , fj  )] + f_data[F_VECIDX(fk+2, fi  , fj  )]
                +   f_data[F_VECIDX(fk-1, fi+1, fj  )] + f_data[F_VECIDX(fk+2, fi+1, fj  )]
                +   f_data[F_VECIDX(fk-1, fi  , fj+1)] + f_data[F_VECIDX(fk+2, fi  , fj+1)]
                +   f_data[F_VECIDX(fk-1, fi+1, fj+1)] + f_data[F_VECIDX(fk+2, fi+1, fj+1)]

                +   f_data[F_VECIDX(fk  , fi-1, fj  )] + f_data[F_VECIDX(fk  , fi+2, fj  )]
                +   f_data[F_VECIDX(fk  , fi  , fj-1)] + f_data[F_VECIDX(fk  , fi  , fj+2)]
                +   f_data[F_VECIDX(fk  , fi+2, fj+1)] + f_data[F_VECIDX(fk  , fi+1, fj+2)]
                +   f_data[F_VECIDX(fk  , fi-1, fj+1)] + f_data[F_VECIDX(fk  , fi+1, fj-1)]

                +   f_data[F_VECIDX(fk+1, fi-1, fj  )] + f_data[F_VECIDX(fk+1, fi+2, fj  )]
                +   f_data[F_VECIDX(fk+1, fi  , fj-1)] + f_data[F_VECIDX(fk+1, fi  , fj+2)]
                +   f_data[F_VECIDX(fk+1, fi+2, fj+1)] + f_data[F_VECIDX(fk+1, fi+1, fj+2)]
                +   f_data[F_VECIDX(fk+1, fi-1, fj+1)] + f_data[F_VECIDX(fk+1, fi+1, fj-1)] )
            + a2 * (
                    f_data[F_VECIDX(fk  , fi-1, fj-1)] + f_data[F_VECIDX(fk+1, fi-1, fj-1)]
                +   f_data[F_VECIDX(fk  , fi+2, fj-1)] + f_data[F_VECIDX(fk+1, fi+2, fj-1)]
                +   f_data[F_VECIDX(fk  , fi-1, fj+2)] + f_data[F_VECIDX(fk+1, fi-1, fj+2)]
                +   f_data[F_VECIDX(fk  , fi+2, fj+2)] + f_data[F_VECIDX(fk+1, fi+2, fj+2)]

                +   f_data[F_VECIDX(fk-1, fi-1, fj  )] + f_data[F_VECIDX(fk-1, fi  , fj-1)]
                +   f_data[F_VECIDX(fk-1, fi+2, fj  )] + f_data[F_VECIDX(fk-1, fi+1, fj-1)]
                +   f_data[F_VECIDX(fk-1, fi-1, fj+1)] + f_data[F_VECIDX(fk-1, fi  , fj+2)] 
                +   f_data[F_VECIDX(fk-1, fi+2, fj+1)] + f_data[F_VECIDX(fk-1, fi+1, fj+2)] 

                +   f_data[F_VECIDX(fk+2, fi-1, fj  )] + f_data[F_VECIDX(fk+2, fi  , fj-1)]
                +   f_data[F_VECIDX(fk+2, fi+2, fj  )] + f_data[F_VECIDX(fk+2, fi+1, fj-1)]
                +   f_data[F_VECIDX(fk+2, fi-1, fj+1)] + f_data[F_VECIDX(fk+2, fi  , fj+2)] 
                +   f_data[F_VECIDX(fk+2, fi+2, fj+1)] + f_data[F_VECIDX(fk+2, fi+1, fj+2)] )
            + a3 * (
                    f_data[F_VECIDX(fk-1, fi-1, fj-1)] + f_data[F_VECIDX(fk+2, fi-1, fj-1)]
                +   f_data[F_VECIDX(fk-1, fi+2, fj-1)] + f_data[F_VECIDX(fk+2, fi+2, fj-1)]
                +   f_data[F_VECIDX(fk-1, fi-1, fj+2)] + f_data[F_VECIDX(fk+2, fi-1, fj+2)]
                +   f_data[F_VECIDX(fk-1, fi+2, fj+2)] + f_data[F_VECIDX(fk+2, fi+2, fj+2)] );
        }
    }
    else {
        assert(false);
    }
#undef C_VECIDX
#undef F_VECIDX
}

#endif