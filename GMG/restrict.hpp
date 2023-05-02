#ifndef SMG_GMG_RESTRICT_HPP
#define SMG_GMG_RESTRICT_HPP

#include "GMG_types.hpp"

template<typename idx_t, typename data_t>
class Restrictor {
public:
    data_t a0, a1, a2, a3;
    const bool eqn_normalized;
    const RSTR_PRLG_TYPE type;
    par_structMatrix<idx_t, data_t, data_t> * Rmat = nullptr;

    Restrictor(RSTR_PRLG_TYPE type, bool normalized): eqn_normalized(normalized), type(type)  { 
        setup_weights();
    }
    virtual void setup_weights();
    virtual void setup_operator(const par_structMatrix<idx_t, data_t, data_t> & A, 
        const par_structVector<idx_t, data_t> & par_fine_vec, par_structVector<idx_t, data_t> & par_coar_vec,
        const COAR_TO_FINE_INFO<idx_t> & info);
    virtual void apply(const par_structVector<idx_t, data_t> & par_fine_vec, 
        par_structVector<idx_t, data_t> & par_coar_vec, const COAR_TO_FINE_INFO<idx_t> & info);
    virtual ~Restrictor() {
        if (Rmat) {delete Rmat; Rmat = nullptr;}
    }
};

template<typename idx_t, typename data_t>
void Restrictor<idx_t, data_t>::setup_weights() {
    switch (type)
    {
    case Vtx_2d9:
    case Vtx_2d9_OpDep:
        if (eqn_normalized) {
            a0 = 1.0000;
            a1 = 0.5000;
            a2 = 0.2500;
        } else {
            a0 = 0.2500;
            a1 = 0.1250;
            a2 = 0.0625;
        }
        break;
    case Vtx_2d5:
        if (eqn_normalized) {
            a0 = 2.0000;
            a1 = 0.5000;
        } else {
            a0 = 0.500;
            a1 = 0.125;
        }
        break;
    case Vtx_inject:
        if (eqn_normalized) {
            a0 = 4.0;
        } else {
            a0 = 1.0;
        }
        break;
    case Cell_2d4:
        if (eqn_normalized) {
            a0 = 1.0;   
        } else {
            a0 = 0.25;
        }
        break;
    case Cell_2d16:
        if (eqn_normalized) {
            a0 = 0.5625;
            a1 = 0.1875;
            a2 = 0.0625;
        } else {
            a0 = 0.140625;// (9/16) / 4
            a1 = 0.046875;// (3/16) / 4
            a2 = 0.015625;// (1/16) / 4
        }
        break;
    case Cell_3d8:
        //    a00----a00
        //    /|     /|
        //  a00----a00|
        //   | a00--|a00
        //   |/     |/
        //  a00----a00
        a0 = 0.125;
        break;
    case Cell_3d64:
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
void Restrictor<idx_t, data_t>::setup_operator(const par_structMatrix<idx_t, data_t, data_t> & A, 
        const par_structVector<idx_t, data_t> & par_fine_vec, par_structVector<idx_t, data_t> & par_coar_vec,
        const COAR_TO_FINE_INFO<idx_t> & info)
{
    assert(type == Vtx_2d9_OpDep);

    const seq_structMatrix<idx_t, data_t, data_t> & Ah = *(A.local_matrix);
    const seq_structVector<idx_t, data_t> & f_vec = *(par_fine_vec.local_vector);
    const seq_structVector<idx_t, data_t> & c_vec = *(par_coar_vec.local_vector);
    
    CHECK_LOCAL_HALO(Ah, f_vec);
    // 按照粗点为簇来组织限制矩阵，Rmat应该是跟par_coar_vec相同的规格
    int procs_dims = sizeof(A.comm_pkg->cart_ids) / sizeof(int); assert(procs_dims == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(A.comm_pkg->cart_comm, procs_dims, num_procs, periods, coords);

    Rmat = new par_structMatrix<idx_t, data_t, data_t>(A.comm_pkg->cart_comm, 9, 
            par_coar_vec.global_size_x, par_coar_vec.global_size_y, par_coar_vec.global_size_z,
            num_procs[1], num_procs[0], num_procs[2]);
    Rmat->set_val(0.0, true);// 需要将halo区设为0
    seq_structMatrix<idx_t, data_t, data_t> * R_local = Rmat->local_matrix;

    if (info.type == DECP_XZ) {
        const idx_t sx = info.stride[0], sz = info.stride[2];
        assert(sx == 2 && info.stride[1] == 1 && sz == 2);
        const idx_t vec_k_size = f_vec.slice_k_size;
        const idx_t cibeg = c_vec.halo_x, ciend = cibeg + c_vec.local_x,
                     jbeg = c_vec.halo_y,  jend =  jbeg + c_vec.local_y,
                    ckbeg = c_vec.halo_z, ckend = ckbeg + c_vec.local_z;
        assert(cibeg == f_vec.halo_x && jbeg == f_vec.halo_y && ckbeg == f_vec.halo_z);

        const idx_t buf_len = (f_vec.halo_x * 2 + f_vec.local_x) * vec_k_size;// 逐面做buf即可
        data_t * se_scy = new data_t [buf_len], * sw_scy = new data_t [buf_len];// se/scy and sw/scy 
        data_t * ss_scx = new data_t [buf_len], * sn_scx = new data_t [buf_len];// ss/scx and sn/scx

        assert(Ah.num_diag == 9);
        const idx_t mat_dk_size = Ah.slice_dk_size;
#define FVEC_IDX(i,k) (i) * vec_k_size + (k)
#define FMAT_IDX(i,k) (i) * mat_dk_size + (k) * 9
        for (idx_t j = jbeg; j < jend; j++) {
            const data_t * Adat = Ah.data + j * Ah.slice_dki_size;
            // 准备该面的数据
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t fi = 0; fi < 2*f_vec.halo_x + f_vec.local_x; fi++)// 注意这里需要将halo区内的也一并计算，因为下面Rmat的计算会引用到
            for (idx_t fk = 0; fk < vec_k_size; fk++) {
                const idx_t id = FVEC_IDX(fi, fk);
                const data_t * src = Adat + fi * Ah.slice_dk_size + fk * Ah.num_diag;
                data_t sw = src[0] + src[1] + src[2], scy = src[3] + src[4] + src[5], se = src[6] + src[7] + src[8];
                data_t ss = src[0] + src[3] + src[6], scx = src[1] + src[4] + src[7], sn = src[2] + src[5] + src[8];
                se_scy[id] = se / scy;
                sw_scy[id] = sw / scy;
                ss_scx[id] = ss / scx;
                sn_scx[id] = sn / scx;
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t ci = cibeg; ci < ciend; ci++)
            for (idx_t ck = ckbeg; ck < ckend; ck++) {// 遍历每个粗点
                data_t * ptr = R_local->data + j * R_local->slice_dki_size + ci * R_local->slice_dk_size + ck * R_local->num_diag;
                const idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx,
                            fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;// 与该粗点位置重合的细点的坐标
                const data_t * Av = nullptr;
                // 对v(i-1,k-1)的影响
                Av = Adat + FMAT_IDX(fi-1, fk-1);
                ptr[0] = (- Av[8] + Av[5]*se_scy[FVEC_IDX(fi-1, fk)] + Av[7]*sn_scx[FVEC_IDX(fi, fk-1)]) / Av[4];
                // 对v(i-1,k  )的影响
                ptr[1] = - se_scy[FVEC_IDX(fi-1, fk)];
                // 对v(i-1,k+1)的影响
                Av = Adat + FMAT_IDX(fi-1, fk+1);
                ptr[2] = (- Av[6] + Av[3]*se_scy[FVEC_IDX(fi-1, fk)] + Av[7]*ss_scx[FVEC_IDX(fi, fk+1)]) / Av[4];
                // 对v(i  ,k-1)的影响
                ptr[3] = - sn_scx[FVEC_IDX(fi, fk-1)];
                // 对v(i  ,k  )的影响，与自己重合的细点，永远是1.0
                ptr[4] = 1.0;
                // 对v(i  ,k+1)的影响
                ptr[5] = - ss_scx[FVEC_IDX(fi, fk+1)];
                // 对v(i+1,k-1)的影响
                Av = Adat + FMAT_IDX(fi+1, fk-1);
                ptr[6] = (- Av[2] + Av[1]*sn_scx[FVEC_IDX(fi, fk-1)] + Av[5]*sw_scy[FVEC_IDX(fi+1, fk)]) / Av[4];
                // 对v(i+1,k  )的影响
                ptr[7] = - sw_scy[FVEC_IDX(fi+1, fk)];
                // 对v(i+1,k+1)的影响
                Av = Adat + FMAT_IDX(fi+1, fk+1);
                ptr[8] = (- Av[0] + Av[1]*ss_scx[FVEC_IDX(fi, fk+1)] + Av[3]*sw_scy[FVEC_IDX(fi+1, fk)]) / Av[4];

                for (idx_t d = 0; d < R_local->num_diag; d++)
                    ptr[d] *= 0.25;// 限制算子是插值除以2^(dim)

                // For DEBUG purpose
                // ptr[2] = 0.0625; ptr[5] = 0.125; ptr[8] = 0.0625;
                // ptr[1] = 0.125;  ptr[4] = 0.25 ; ptr[7] = 0.125;
                // ptr[0] = 0.0625; ptr[3] = 0.125; ptr[6] = 0.0625;
            }
        }
#undef FVEC_IDX
#undef FMAT_IDX
        delete se_scy; delete sw_scy;
        delete ss_scx; delete sn_scx;   
    }
    else if (info.type == SEMI_XY) {
        // 当作二维半粗化的ODI时，并非解耦，而是将每个点上的3dx的数据压缩成XY面上的2d9
        //                 East
        //  ^ J(外)      6  7  8
        //  |     South  3  4  5  North    
        //  ----> I(内)  0  1  2
        //                 West 
        const idx_t sx = info.stride[0], sy = info.stride[1];
        assert(sx == 2 && sy == 2 && info.stride[2] == 1);
        const idx_t vec_k_size = f_vec.slice_k_size, vec_ki_size = f_vec.slice_ki_size;
        const idx_t cibeg = c_vec.halo_x, ciend = cibeg + c_vec.local_x,
                    cjbeg = c_vec.halo_y, cjend = cjbeg + c_vec.local_y,
                     kbeg = c_vec.halo_z,  kend =  kbeg + c_vec.local_z;
        assert(cibeg == f_vec.halo_x && cjbeg == f_vec.halo_y && kbeg == f_vec.halo_z);
        // 因为压缩沿着内存最内维的z向，所以三维的buf
        seq_structMatrix<idx_t, data_t, data_t> Ah_avgZ(9, Ah.local_x, Ah.local_y, Ah.local_z, Ah.halo_x, Ah.halo_y, Ah.halo_z);
        const idx_t mat_dk_size = Ah_avgZ.slice_dk_size, mat_dki_size = Ah_avgZ.slice_dki_size;

        const idx_t buf_len = (f_vec.halo_y * 2 + f_vec.local_y) * vec_ki_size;
        data_t * se_sc_in  = new data_t [buf_len], * sw_sc_in  = new data_t [buf_len];
        data_t * ss_sc_out = new data_t [buf_len], * sn_sc_out = new data_t [buf_len];
#define FVEC_IDX(j,i,k) (j)*vec_ki_size + (i)*vec_k_size + (k)
#define FMAT_IDX(j,i,k) (j)*mat_dki_size + (i)*mat_dk_size + (k)*9
        if (Ah.num_diag == 19) {
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t fj = 0; fj < 2*f_vec.halo_y + f_vec.local_y; fj++)
            for (idx_t fi = 0; fi < 2*f_vec.halo_x + f_vec.local_x; fi++)
            for (idx_t fk = 0; fk < vec_k_size; fk++) {// 注意这里需要将halo区内的也一并计算，因为下面Rmat的计算会引用到
                const data_t * src = Ah.data + fj * Ah.slice_dki_size + fi * Ah.slice_dk_size + fk * Ah.num_diag;
                data_t * dst = Ah_avgZ.data + fj * Ah_avgZ.slice_dki_size + fi * Ah_avgZ.slice_dk_size + fk * Ah_avgZ.num_diag;
                idx_t id = FVEC_IDX(fj, fi, fk);
                dst[0] = src[0]                  ; dst[1] = src[1] + src[2] + src[3] ; dst[2] = src[4];
                dst[3] = src[5] + src[6] + src[7]; dst[4] = src[8] + src[9] + src[10]; dst[5] = src[11]+ src[12]+ src[13];
                dst[6] = src[14]                 ; dst[7] = src[15]+ src[16]+ src[17]; dst[8] = src[18];
                data_t ss = dst[0] + dst[3] + dst[6], sc_out = dst[1] + dst[4] + dst[7], sn = dst[2] + dst[5] + dst[8];
                data_t sw = dst[0] + dst[1] + dst[2], sc_in  = dst[3] + dst[4] + dst[5], se = dst[6] + dst[7] + dst[8];
                se_sc_in[id] = se / sc_in;
                sw_sc_in[id] = sw / sc_in;
                ss_sc_out[id]= ss / sc_out;
                sn_sc_out[id]= sn / sc_out;

                id ++ ;
                src += Ah.num_diag; dst += Ah_avgZ.num_diag;         
            }
        } else if (Ah.num_diag == 27) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t fj = 0; fj < 2*f_vec.halo_y + f_vec.local_y; fj++)
            for (idx_t fi = 0; fi < 2*f_vec.halo_x + f_vec.local_x; fi++)
            for (idx_t fk = 0; fk < vec_k_size; fk++) {// 注意这里需要将halo区内的也一并计算，因为下面Rmat的计算会引用到
                const data_t * src = Ah.data + fj * Ah.slice_dki_size + fi * Ah.slice_dk_size + fk * Ah.num_diag;
                data_t * dst = Ah_avgZ.data + fj * Ah_avgZ.slice_dki_size + fi * Ah_avgZ.slice_dk_size + fk * Ah_avgZ.num_diag;
                idx_t id = FVEC_IDX(fj, fi, fk);
                dst[0] = src[0] + src[1] + src[2] ; dst[1] = src[3] + src[4] + src[5] ; dst[2] = src[6] + src[7] + src[8];
                dst[3] = src[9] + src[10]+ src[11]; dst[4] = src[12]+ src[13]+ src[14]; dst[5] = src[15]+ src[16]+ src[17];
                dst[6] = src[18]+ src[19]+ src[20]; dst[7] = src[21]+ src[22]+ src[23]; dst[8] = src[24]+ src[25]+ src[26];
                data_t ss = dst[0] + dst[3] + dst[6], sc_out = dst[1] + dst[4] + dst[7], sn = dst[2] + dst[5] + dst[8];
                data_t sw = dst[0] + dst[1] + dst[2], sc_in  = dst[3] + dst[4] + dst[5], se = dst[6] + dst[7] + dst[8];
                se_sc_in[id] = se / sc_in;
                sw_sc_in[id] = sw / sc_in;
                ss_sc_out[id]= ss / sc_out;
                sn_sc_out[id]= sn / sc_out;

                id ++ ;
                src += Ah.num_diag; dst += Ah_avgZ.num_diag;            
            }
        } else MPI_Abort(MPI_COMM_WORLD, -606);

        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++) {
            const idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx,
                        fj = f_vec.halo_y + info.fine_base_idx[1] + (cj - cjbeg) * sy;// 与该粗点位置重合的细点的坐标
            const data_t * Av = nullptr;
            for (idx_t k = kbeg; k < kend; k++) {
                data_t * ptr = R_local->data + cj * R_local->slice_dki_size + ci * R_local->slice_dk_size + k * R_local->num_diag;

                // 对v(j-1,i-1, k)的影响
                Av = Ah_avgZ.data + FMAT_IDX(fj-1, fi-1, k);
                ptr[0] = (- Av[8] + Av[5]*se_sc_in[FVEC_IDX(fj-1, fi, k)] + Av[7]*sn_sc_out[FVEC_IDX(fj, fi-1, k)]) / Av[4];

                // if (ptr[0] != ptr[0]) {
                //     printf(" ptr[0] !!! cj %d ci %d fj %d fi %d k %d Av[8] %.3e Av[5] %.3e se_sc_in %.3e Av[7] %.3e sn_sc_out %.3e\n",
                //         cj, ci, fj, fi, k, Av[8], Av[5], se_sc_in[FVEC_IDX(fj-1, fi, k)], Av[7], sn_sc_out[FVEC_IDX(fj, fi-1, k)]);
                //     assert(false);
                // }

                // 对v(j-1,i  , k)的影响
                ptr[1] = - se_sc_in[FVEC_IDX(fj-1, fi, k)];
                // 对v(j-1,i+1, k)的影响
                Av = Ah_avgZ.data + FMAT_IDX(fj-1, fi+1, k);
                ptr[2] = (- Av[6] + Av[3]*se_sc_in[FVEC_IDX(fj-1, fi, k)] + Av[7]*ss_sc_out[FVEC_IDX(fj, fi+1, k)]) / Av[4];
                // 对v(j  ,i-1, k)的影响
                ptr[3] = - sn_sc_out[FVEC_IDX(fj, fi-1, k)];
                // 对v(j  ,i  , k)的影响，与自己重合的细点，永远是1.0
                ptr[4] = 1.0;
                // 对v(j  ,i+1, k)的影响
                ptr[5] = - ss_sc_out[FVEC_IDX(fj, fi+1, k)];
                // 对v(j+1,i-1, k)的影响
                Av = Ah_avgZ.data + FMAT_IDX(fj+1, fi-1, k);
                ptr[6] = (- Av[2] + Av[1]*sn_sc_out[FVEC_IDX(fj, fi-1, k)] + Av[5]*sw_sc_in[FVEC_IDX(fj+1, fi, k)]) / Av[4];
                // 对v(j+1,i  , k)的影响
                ptr[7] = - sw_sc_in[FVEC_IDX(fj+1, fi, k)];
                // 对v(j+1,i+1, k)的影响
                Av = Ah_avgZ.data + FMAT_IDX(fj+1, fi+1, k);
                ptr[8] = (- Av[0] + Av[1]*ss_sc_out[FVEC_IDX(fj, fi+1, k)] + Av[3]*sw_sc_in[FVEC_IDX(fj+1, fi, k)]) / Av[4];

                for (idx_t d = 0; d < R_local->num_diag; d++) 
                    ptr[d] *= 0.25;// 限制算子是插值除以2^(dim)

                // For DEBUG purpose
                // ptr[2] = 0.0625; ptr[5] = 0.125; ptr[8] = 0.0625;
                // ptr[1] = 0.125;  ptr[4] = 0.25 ; ptr[7] = 0.125;
                // ptr[0] = 0.0625; ptr[3] = 0.125; ptr[6] = 0.0625;
            }
        }
#undef FVEC_IDX
#undef FMAT_IDX
        delete se_sc_in; delete sw_sc_in;
        delete ss_sc_out; delete sn_sc_out;   
    }
    else {
        MPI_Abort(MPI_COMM_WORLD, -20230429);
    }

    Rmat->update_halo();
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

    const idx_t c_k_size = c_vec.slice_k_size, c_ki_size = c_vec.slice_ki_size;
    const idx_t f_k_size = f_vec.slice_k_size, f_ki_size = f_vec.slice_ki_size;
    const data_t * f_data = f_vec.data;
    data_t * c_data = c_vec.data;

#define C_VECIDX(k, i, j) (k) + (i) * c_k_size + (j) * c_ki_size
#define F_VECIDX(k, i, j) (k) + (i) * f_k_size + (j) * f_ki_size

    if (type == Cell_3d8) {
        const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                    cjbeg = hy, cjend = cjbeg + c_vec.local_y,
                    ckbeg = hz, ckend = ckbeg + c_vec.local_z;
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
    }
    else if (type == Cell_2d4) {
        assert(info.fine_base_idx[2] == 0 && (f_vec.local_z & 0x1) == 0);// 细网格的cell数一定是偶数
        if (info.type == SEMI_XY) {
            const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                        cjbeg = hy, cjend = cjbeg + c_vec.local_y,
                         kbeg = hz,  kend =  kbeg + c_vec.local_z;
            assert(c_k_size == f_k_size);
            #pragma omp parallel for collapse(3) schedule(static)
            for (idx_t cj = cjbeg; cj < cjend; cj++)
            for (idx_t ci = cibeg; ci < ciend; ci++)
            for (idx_t  k =  kbeg;  k <  kend;  k++) {// same as fk's range
                const idx_t fj = hy + by + (cj - cjbeg) * sy;// fj - fjbeg == (cj - cjbeg) * sy + by，而其中fjbeg就等于hy
                const idx_t fi = hx + bx + (ci - cibeg) * sx;
                c_data[C_VECIDX(k, ci, cj)] = a0 * (
                    f_data[F_VECIDX(k  , fi  , fj  )]
                +   f_data[F_VECIDX(k  , fi+1, fj  )]
                +   f_data[F_VECIDX(k  , fi  , fj+1)]
                +   f_data[F_VECIDX(k  , fi+1, fj+1)]
                );
            }
        }
        else if (info.type == DECP_XZ) {
            const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                         jbeg = hy,  jend =  jbeg + c_vec.local_y,
                        ckbeg = hz, ckend = ckbeg + c_vec.local_z;
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                data_t * c_col = c_data + j * c_ki_size + ci * c_k_size;
                const idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 左半的i序号
                const data_t* f_col_iN = f_data + j * f_ki_size +  fi   * f_k_size,
                            * f_col_iP = f_data + j * f_ki_size + (fi+1)* f_k_size;
                for (idx_t ck = ckbeg; ck < ckend; ck++) {
                    idx_t fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;// 下半的k序号
                    c_col[ck] = a0 * (f_col_iN[fk] + f_col_iN[fk+1] + f_col_iP[fk] + f_col_iP[fk+1]);
                }
            }
        }
        else {
            MPI_Abort(MPI_COMM_WORLD, -809);
        }
    }
    else if (type == Cell_3d64) {
        par_fine_vec.update_halo();

        const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                    cjbeg = hy, cjend = cjbeg + c_vec.local_y,
                    ckbeg = hz, ckend = ckbeg + c_vec.local_z;
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
    else if (type == Vtx_2d9) {
        par_fine_vec.update_halo();

        if (info.type == SEMI_XY) {
            const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                        cjbeg = hy, cjend = cjbeg + c_vec.local_y,
                         kbeg = hz,  kend =  kbeg + c_vec.local_z;
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t cj = cjbeg; cj < cjend; cj++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                data_t* c_col = c_data + C_VECIDX(0, ci, cj);
                idx_t fj = f_vec.halo_y + info.fine_base_idx[1] + (cj - cjbeg) * sy;
                idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 该粗位置在细网格中的对应索引
                const data_t* f2 = f_data + F_VECIDX(0, fi-1, fj+1), * f5 = f_data + F_VECIDX(0, fi, fj+1), * f8 = f_data + F_VECIDX(0, fi+1, fj+1),
                            * f1 = f_data + F_VECIDX(0, fi-1, fj  ), * f4 = f_data + F_VECIDX(0, fi, fj  ), * f7 = f_data + F_VECIDX(0, fi+1, fj  ),
                            * f0 = f_data + F_VECIDX(0, fi-1, fj-1), * f3 = f_data + F_VECIDX(0, fi, fj-1), * f6 = f_data + F_VECIDX(0, fi+1, fj-1);
                for (idx_t k = kbeg; k < kend; k++) {
                    c_col[k] = a0 *  f4[k]
                            +  a1 * (f1[k] + f3[k] + f5[k] + f7[k])
                            +  a2 * (f0[k] + f2[k] + f6[k] + f8[k]);
                }
            }
        }
        else if (info.type == DECP_XZ) {
            const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                         jbeg = hy,  jend =  jbeg + c_vec.local_y,
                        ckbeg = hz, ckend = ckbeg + c_vec.local_z;
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                // 优化版：差不多，单节点带宽~200GB/s
                idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;
                idx_t ck = ckbeg;
                idx_t fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;
                data_t       * c_col    = c_data + j * c_ki_size + ci * c_k_size;
                const data_t * f_col_iZ = f_data + j * f_ki_size + fi * f_k_size;
                const data_t * f_col_iP = f_col_iZ + f_k_size;
                const data_t * f_col_iN = f_col_iZ - f_k_size;
                for ( ; ck < ckend; ck ++, fk += sz) {// fixde 2 for sz?
                    c_col[ck] = a0 *  f_col_iZ[fk]
                            +   a1 * (f_col_iZ[fk+1] + f_col_iZ[fk-1] + f_col_iN[fk  ] + f_col_iP[fk  ])
                            +   a2 * (f_col_iP[fk+1] + f_col_iN[fk+1] + f_col_iP[fk-1] + f_col_iN[fk-1]);
                }
            }// j loop
        }
        else {
            MPI_Abort(MPI_COMM_WORLD, -908);
        }
    }
    else if (type == Vtx_2d9_OpDep) {
        par_fine_vec.update_halo();

        const seq_structMatrix<idx_t, data_t, data_t> * R_local = Rmat->local_matrix;
        CHECK_LOCAL_HALO(*R_local, c_vec);
        assert(R_local->num_diag == 9);
        if (info.type == SEMI_XY) {
            //                 East
            //  ^ J(外)      6  7  8
            //  |     South  3  4  5  North    
            //  ----> I(内)  0  1  2
            //                 West 
            const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                        cjbeg = hy, cjend = cjbeg + c_vec.local_y,
                         kbeg = hz,  kend =  kbeg + c_vec.local_z;
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t cj = cjbeg; cj < cjend; cj++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                data_t* c_col = c_data + C_VECIDX(0, ci, cj);
                idx_t fj = f_vec.halo_y + info.fine_base_idx[1] + (cj - cjbeg) * sy;
                idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 该粗位置在细网格中的对应索引
                const data_t* f0 = f_data + F_VECIDX(0, fi-1, fj-1), * f1 = f_data + F_VECIDX(0, fi  , fj-1), * f2 = f_data + F_VECIDX(0, fi+1, fj-1),
                            * f3 = f_data + F_VECIDX(0, fi-1, fj  ), * f4 = f_data + F_VECIDX(0, fi  , fj  ), * f5 = f_data + F_VECIDX(0, fi+1, fj  ),
                            * f6 = f_data + F_VECIDX(0, fi-1, fj+1), * f7 = f_data + F_VECIDX(0, fi  , fj+1), * f8 = f_data + F_VECIDX(0, fi+1, fj+1);
                const data_t* Rv = R_local->data + cj * R_local->slice_dki_size + ci * R_local->slice_dk_size + kbeg * R_local->num_diag;
                for (idx_t k = kbeg; k < kend; k++) {
                    c_col[k] = Rv[0] * f0[k] + Rv[1] * f1[k] + Rv[2] * f2[k]
                            +  Rv[3] * f3[k] + Rv[4] * f4[k] + Rv[5] * f5[k]
                            +  Rv[6] * f6[k] + Rv[7] * f7[k] + Rv[8] * f8[k];
                    Rv += R_local->num_diag;
                }
            }
        }
        else if (info.type == DECP_XZ) {
            const idx_t cibeg = hx, ciend = cibeg + c_vec.local_x,
                         jbeg = hy,  jend =  jbeg + c_vec.local_y,
                        ckbeg = hz, ckend = ckbeg + c_vec.local_z;
            #pragma omp parallel for collapse(2) schedule(static)
            for (idx_t j = jbeg; j < jend; j++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                const idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;
                const data_t * Rv = R_local->data + j * R_local->slice_dki_size + ci * R_local->slice_dk_size + ckbeg * R_local->num_diag;
                data_t* c_col = c_data + j * c_ki_size + ci * c_k_size;
                const data_t* f_col_iZ = f_data + j * f_ki_size + fi * f_k_size,
                            * f_col_iP = f_col_iZ + f_k_size,
                            * f_col_iN = f_col_iZ - f_k_size;
                idx_t ck = ckbeg, fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;
                for ( ; ck < ckend; ck ++, fk += sz) {
                    c_col[ck] =   Rv[0] * f_col_iN[fk-1] + Rv[1] * f_col_iN[fk] + Rv[2] * f_col_iN[fk+1]
                                + Rv[3] * f_col_iZ[fk-1] + Rv[4] * f_col_iZ[fk] + Rv[5] * f_col_iZ[fk+1]
                                + Rv[6] * f_col_iP[fk-1] + Rv[7] * f_col_iP[fk] + Rv[8] * f_col_iP[fk+1];
                    Rv += R_local->num_diag;
                }
            }
        }
        else {
            MPI_Abort(MPI_COMM_WORLD, -902);
        }
    }
    else {
        assert(false);
    }
#undef C_VECIDX
#undef F_VECIDX
}

#endif