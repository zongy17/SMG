#ifndef SMG_GMG_PROLONG_HPP
#define SMG_GMG_PROLONG_HPP

#include "GMG_types.hpp"

template<typename idx_t, typename data_t>
class Interpolator {
public:
    // 插值算子不需要根据方程是否为标准化形式而改变
    const RSTR_PRLG_TYPE type;
    par_structMatrix<idx_t, data_t, data_t> * Pmat = nullptr; 

    Interpolator(RSTR_PRLG_TYPE type): type(type) {  }
    virtual void setup_operator(const par_structMatrix<idx_t, data_t, data_t> & Rmat);
    virtual void apply(const par_structVector<idx_t, data_t> & par_coar_vec,
        par_structVector<idx_t, data_t> & par_fine_vec, const COAR_TO_FINE_INFO<idx_t> & info);
    virtual ~Interpolator() {
        if (Pmat) {delete Pmat; Pmat = nullptr;}
    }
};

template<typename idx_t, typename data_t>
void Interpolator<idx_t, data_t>::setup_operator(const par_structMatrix<idx_t, data_t, data_t> & Rmat)
{
    assert(type == Vtx_2d9_OpDep);
    Pmat = new par_structMatrix<idx_t, data_t, data_t>(Rmat);
    const   seq_structMatrix<idx_t, data_t, data_t> * R_local = Rmat.local_matrix;
            seq_structMatrix<idx_t, data_t, data_t> * P_local = Pmat->local_matrix;
    // 需要乘以系数2^(dim)
    assert(Rmat.num_diag == 9);
    const idx_t tot_len = R_local->num_diag * (R_local->local_x + 2 * R_local->halo_x)
        * (R_local->local_y + 2 * R_local->halo_y) * (R_local->local_z + 2 * R_local->halo_z);
    #pragma omp parallel for schedule(static)
    for (idx_t t = 0; t < tot_len; t++)
        P_local->data[t] = R_local->data[t] * 4.0;// 二维，2^2=4

    Pmat->update_halo();
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

    if (type == Cell_3d64) {
        const data_t a0 = 27.0 / 64.0, a1 =  9.0 / 64.0, a2 =  3.0 / 64.0, a3 =  1.0 / 64.0;
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
    else if (type == Cell_3d8) {
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
    }
    else if (type == Cell_2d4) {
        if (info.type == SEMI_XY){
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
        else if (info.type == DECP_XZ) {
            assert(sy == 1 && info.fine_base_idx[0] == 0 && info.fine_base_idx[2] == 0);
            #pragma omp parallel for collapse(2) schedule(static) 
            for (idx_t  j = cjbeg;  j < cjend; j++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                const data_t * c_col = c_data + j * c_ki_size + ci * c_k_size;
                const idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 左半位置的i序号
                data_t* f_col_iN = f_data + j * f_ki_size +  fi   * f_k_size;
                data_t* f_col_iP = f_data + j * f_ki_size + (fi+1)* f_k_size;
                for (idx_t ck = ckbeg; ck < ckend; ck++) {
                    const idx_t fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;// 下半位置的k序号
                    f_col_iN[fk] = c_col[ck]; f_col_iN[fk+1] = c_col[ck];
                    f_col_iP[fk] = c_col[ck]; f_col_iP[fk+1] = c_col[ck];
                }
            }
        }
    }
    else if (type == Cell_2d16) {
         par_coar_vec.update_halo();
         
        assert(f_k_size == c_k_size);
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t cj = cjbeg; cj < cjend; cj++)
        for (idx_t ci = cibeg; ci < ciend; ci++) {
            idx_t fj_group = f_vec.halo_y + info.fine_base_idx[1] + (cj - cjbeg) * sy;
            idx_t fi_group = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;
            const data_t * c_jZiZ = c_data + cj * c_ki_size + ci * c_k_size;
            const data_t * c_jZiP = c_jZiZ                        +      c_k_size ;
            const data_t * c_jZiN = c_jZiZ                        -      c_k_size ;
            const data_t * c_jPiZ = c_jZiZ +      c_ki_size                       ;
            const data_t * c_jNiZ = c_jZiZ -      c_ki_size                       ;

            data_t * f_left_down = f_data      + fj_group * f_ki_size + fi_group * f_k_size;
            data_t * f_left_up   = f_left_down +            f_ki_size                      ;
            data_t * f_right_down= f_left_down                        +            f_k_size;
            data_t * f_right_up  = f_right_down+            f_ki_size                      ;
            for (idx_t k = ckbeg; k < ckend; k++) {
                data_t tmp0 = 0.5625 * c_jZiZ[k];
                data_t tmp1 = 0.1875 * c_jZiN[k];
                data_t tmp2 = 0.1875 * c_jNiZ[k];
                data_t tmp3 = 0.1875 * c_jPiZ[k];
                data_t tmp4 = 0.1875 * c_jZiP[k];
                f_left_down [k] = tmp0 + tmp1 + tmp2 + 0.0625 * c_jNiZ[k - c_k_size];
                f_left_up   [k] = tmp0 + tmp1 + tmp3 + 0.0625 * c_jPiZ[k - c_k_size];
                f_right_down[k] = tmp0 + tmp4 + tmp2 + 0.0625 * c_jNiZ[k + c_k_size];
                f_right_up  [k] = tmp0 + tmp4 + tmp3 + 0.0625 * c_jPiZ[k + c_k_size];
            }       
        }
    }
    else if (type == Vtx_2d9) {
        par_coar_vec.update_halo();
        if (info.type == SEMI_XY) {
            assert(info.stride[2] == 1 && info.fine_base_idx[2] == 0);
            const idx_t ofi = 1 - sx * info.fine_base_idx[0];
            const idx_t ofj = 1 - sy * info.fine_base_idx[1];
                /* 以sx*sy个F点为一个Group
                *
                *   --C----F----C----F----C----F----C----F--       基本单位    --F----F---
                *     |    |    |    |    |    |    |    |        =========>    |    |
                *   --F----F----F----F----F----F----F----F--        group     --C----F---
                *     |    |    |    |    |    |    |    |
                *   --C----F----C----F----C----F----C----F--
                *     |    | ===|====|====|====|=== |    |    进程在细网格上的负责区域：==== 所围
                *   --F----F-||-F----F----F----F-||-F----F--  细网格和粗网格周围都有一圈halo区
                *     |    | || |    |    |    | || |    |
                *   --C----F-||-C----F----C----F-||-C----F--
                *     |    | || |    |    |    | || |    |
                *   --F----F-||-F----F----F----F-||-F----F--
                *     |    | || |    |    |    | || |    |
                *   --C----F-||-C----F----C----F-||-C----F--
                *     |    | ===|====|====|====|=== |    |
                *   --F----F----F----F----F----F----F----F--
                *
                *  group中左下角的点，在粗网格、细网格中是重合的
                */
            #pragma omp parallel for collapse(2) schedule(static) 
            for (idx_t cj = cjbeg; cj < cjend; cj++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 该粗位置在细网格中的对应索引
                idx_t fj = f_vec.halo_y + info.fine_base_idx[1] + (cj - cjbeg) * sy;
                const data_t* c_iZjZ = c_data + C_VECIDX(0, ci, cj);
                      data_t* f_iZjZ = f_data + F_VECIDX(0, fi, fj);
                const data_t* c_iOjZ = c_iZjZ + ofi * c_k_size, * c_iZjO = c_iZjZ + ofj * c_ki_size,
                            * c_iOjO = c_iOjZ + ofj * c_ki_size;
                      data_t* f_iOjZ = f_iZjZ + ofi * f_k_size, * f_iZjO = f_iZjZ + ofj * f_ki_size,
                            * f_iOjO = f_iOjZ + ofj * f_ki_size;
                for (idx_t k = ckbeg; k < ckend; k++) {
                    f_iZjZ[k] = c_iZjZ[k];
                    f_iOjZ[k] = 0.5  * (c_iZjZ[k] + c_iOjZ[k]);
                    f_iZjO[k] = 0.5  * (c_iZjZ[k] + c_iZjO[k]);
                    f_iOjO[k] = 0.25 * (c_iZjZ[k] + c_iOjZ[k] + c_iZjO[k] + c_iOjO[k]);
                }
                // for (idx_t k = ckbeg; k < ckend; k++) {
                //     // C与F重合的一柱：(fi_group, fj_group) 即group中的左下角点
                //     f_data[F_VECIDX(k, fi, fj)] = c_data[C_VECIDX(k, ci, cj)];
                //     // 位于两个C点连线正中的两柱：(fi_group, fj_group + 1) 即group中的左上角点
                //     f_data[F_VECIDX(k, fi, fj + ofj)] = 0.5 * (c_data[C_VECIDX(k, ci, cj)] + c_data[C_VECIDX(k, ci, cj + ofj)]
                //     );
                //     // 位于两个C点连线正中的两柱：(fi_group + 1, fj_group) 即group中的右下角点
                //     f_data[F_VECIDX(k, fi + ofi, fj)] = 0.5 * (c_data[C_VECIDX(k, ci, cj)] + c_data[C_VECIDX(k, ci + ofi, cj)]
                //     );
                //     // 位于四个C点对角线相交位置的一柱：(fi_group + 1, fj_group + 1) 即group中的右上角点
                //     f_data[F_VECIDX(k, fi + ofi, fj + ofj)] = 0.25 * (
                //             c_data[C_VECIDX(k, ci      , cj)] + c_data[C_VECIDX(k, ci      , cj + ofj)]
                //         +   c_data[C_VECIDX(k, ci + ofi, cj)] + c_data[C_VECIDX(k, ci + ofi, cj + ofj)]
                //     );
                // }
            }
        }
        else if (info.type == DECP_XZ) {
            assert(info.fine_base_idx[2] >= 1);// 非周期方向z不能挨在边上
            assert(info.stride[1] == 1);
            const idx_t di = (info.fine_base_idx[0] == 0) ? 1 : -1;// 根据起始点的对应关系设置group内偏移
        
        /* 以sx*sz个F点为一个Group  
        当info.fine_base_idx[0] == 0时
        *   --F----F-||-F----F----F----F----F----F-||-F
        *     |    | || |    |    |    |    |    | || |  
        *   --F----F-||-F----F----F----F----F----F-||-F  
        *     |    | || |    |    |    |    |    | || |           基本单位    --F----F---
        *   --C----F-||-C----F----C----F----C----F-||-C =ckend-1  =========>   |    |
        *     |    | || |    |    |    |    |    | || |            group     --C----F---
        *   --F----F-||-F----F----F----F----F----F-||-F
        *     |    | || |    |    |    |    |    | || |            进程在细网格上的负责区域：==== 所围
        *   --C----F-||-C----F----C----F----C----F-||-C =ckbeg     细网格和粗网格周围都有一圈halo区
        *     |    | || |    |    |    |    |    | || |
        *   --F----F-||-F----F----F----F----F----F-||-F
        *     |    | || |    |    |    |    |    | || |
        *   --F----F-||-F----F----F----F----F----F-||-F
        * 
        *  group中左下角的点，在粗网格、细网格中是重合的
        当info.fine_base_idx[0] == 1时
        *   --F-||-F----F----F----F----F----F-||-F----F
        *     | || |    |    |    |    |    | || |    |  
        *   --F-||-F----F----F----F----F----F-||-F----F  
        *     | || |    |    |    |    |    | || |    |           基本单位    --F----F---
        *   --C-||-F----C----F----C----F----C-||-F----C =ckend-1  =========>   |    |
        *     | || |    |    |    |    |    | || |    |            group     --F----C---
        *   --F-||-F----F----F----F----F----F-||-F----F
        *     | || |    |    |    |    |    | || |    |            进程在细网格上的负责区域：==== 所围
        *   --C-||-F----C----F----C----F----C-||-F----C =ckbeg     细网格和粗网格周围都有一圈halo区
        *     | || |    |    |    |    |    | || |    |
        *   --F-||-F----F----F----F----F----F-||-F----F
        *     | || |    |    |    |    |    | || |    |
        *   --F-||-F----F----F----F----F----F-||-F----F
        *
        *  group中右下角的点，在粗网格、细网格中是重合的
        */
            #pragma omp parallel for collapse(2) schedule(static) 
            for (idx_t  j = cjbeg;  j < cjend; j++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                // 优化版：差别不大，单节点带宽250~270GB/s
                idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 该粗位置在细网格中的对应索引
                idx_t ck = ckbeg;
                idx_t fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;
                const  data_t * c_col_i = c_data + j * c_ki_size + ci * c_k_size + ck;
                const  data_t * c_col_di= c_col_i + di * c_k_size;// 偏移量
                       data_t * f_col_i = f_data + j * f_ki_size + fi * f_k_size + fk;
                       data_t * f_col_di= f_col_i + di * f_k_size;// 偏移量
                {// bottom 的两个F点
                    *(f_col_i  - 1)  = 0.5 *  *c_col_i;
                    *(f_col_di - 1) = 0.25* (*c_col_i + *c_col_di);
                }
                for ( ; ck < ckend - 1; ) {
                    *f_col_i        = *c_col_i;
                    *f_col_di       = 0.5 * (*c_col_i + *c_col_di);
                    *(f_col_i  + 1) = 0.5 * (*c_col_i + *(c_col_i + 1));
                    *(f_col_di + 1) = 0.25* (*c_col_i + *c_col_di + *(c_col_i + 1) + *(c_col_di + 1));

                    ck ++; c_col_i ++; c_col_di ++;
                    fk += 2; f_col_i += 2; f_col_di += 2;// assert(sz == 2)
                }// 循环结束时ck == ckend - 1
                {// top 的两个F点
                    *f_col_i        = *c_col_i;
                    *f_col_di       = 0.5 * (*c_col_i + *c_col_di);
                    *(f_col_i  + 1) = 0.5 *  *c_col_i;
                    *(f_col_di + 1) = 0.25* (*c_col_i + *c_col_di);
                }
            }
        }
        else {
            MPI_Abort(MPI_COMM_WORLD, -309);
        }
    }
    else if (type == Vtx_2d9_OpDep) {
        par_coar_vec.update_halo();
        const seq_structMatrix<idx_t, data_t, data_t> * P_local = Pmat->local_matrix;
        CHECK_LOCAL_HALO(*P_local, c_vec);
        assert(P_local->num_diag == 9);
        if (info.type == SEMI_XY) {
            //                 East
            //  ^ J(外)      6  7  8
            //  |     South  3  4  5  North    
            //  ----> I(内)  0  1  2
            //                 West 
            assert(info.stride[2] == 1 && info.fine_base_idx[2] == 0);
            const idx_t ofi = 1 - sx * info.fine_base_idx[0];
            const idx_t ofj = 1 - sy * info.fine_base_idx[1];
            assert(info.fine_base_idx[0] == 0 && info.fine_base_idx[1] == 0);// 为了确定P[...]具体取哪个数
            #pragma omp parallel for collapse(2) schedule(static) 
            for (idx_t cj = cjbeg; cj < cjend; cj++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;// 该粗位置在细网格中的对应索引
                idx_t fj = f_vec.halo_y + info.fine_base_idx[1] + (cj - cjbeg) * sy;
                const data_t* c_iZjZ = c_data + C_VECIDX(0, ci, cj);
                      data_t* f_iZjZ = f_data + F_VECIDX(0, fi, fj);
                const data_t* c_iOjZ = c_iZjZ + ofi * c_k_size, * c_iZjO = c_iZjZ + ofj * c_ki_size,
                            * c_iOjO = c_iOjZ + ofj * c_ki_size;
                      data_t* f_iOjZ = f_iZjZ + ofi * f_k_size, * f_iZjO = f_iZjZ + ofj * f_ki_size,
                            * f_iOjO = f_iOjZ + ofj * f_ki_size;
                // 注意assert了这里所有的offset i和offset j都是1
                const data_t* P_iZjZ = P_local->data + cj * P_local->slice_dki_size + ci * P_local->slice_dk_size + ckbeg * P_local->num_diag,
                            * P_iOjZ = P_iZjZ + ofi * P_local->slice_dk_size,
                            * P_iZjO = P_iZjZ + ofj * P_local->slice_dki_size,
                            * P_iOjO = P_iOjZ + ofj * P_local->slice_dki_size;
                for (idx_t k = ckbeg; k < ckend; k++) {
                    f_iZjZ[k] = P_iZjZ[4] * c_iZjZ[k];
                    f_iOjZ[k] = P_iZjZ[5] * c_iZjZ[k] + P_iOjZ[3] * c_iOjZ[k];
                    f_iZjO[k] = P_iZjZ[7] * c_iZjZ[k] + P_iZjO[1] * c_iZjO[k];
                    f_iOjO[k] = P_iZjZ[8] * c_iZjZ[k] + P_iOjZ[6] * c_iOjZ[k] + P_iZjO[2] * c_iZjO[k] + P_iOjO[0] * c_iOjO[k];
                    P_iZjZ += 9;
                    P_iOjZ += 9;
                    P_iZjO += 9;
                    P_iOjO += 9;
                }
            }
        }
        else if (info.type == DECP_XZ) {
            assert(info.fine_base_idx[2] >= 1);// 非周期方向z不能挨在边上
            assert(info.fine_base_idx[0] == 0);
            assert(info.stride[1] == 1);
            #pragma omp parallel for collapse(2) schedule(static) 
            for (idx_t  j = cjbeg;  j < cjend; j++)
            for (idx_t ci = cibeg; ci < ciend; ci++) {
                const idx_t fi = f_vec.halo_x + info.fine_base_idx[0] + (ci - cibeg) * sx;
                const data_t* P_IZKZ = P_local->data + j * P_local->slice_dki_size + ci * P_local->slice_dk_size + ckbeg * P_local->num_diag,
                            * P_IPKZ = P_IZKZ                                      +      P_local->slice_dk_size,
                            * P_IZKP = P_IZKZ                                                                    +         P_local->num_diag,
                            * P_IPKP = P_IPKZ                                                                    +         P_local->num_diag;
                const data_t* c_col_iZ = c_data + j * c_ki_size + ci * c_k_size,
                            * c_col_iP = c_col_iZ +   c_k_size;// 偏移量
                data_t  * f_col_iZ = f_data + j * f_ki_size + fi * f_k_size ,
                        * f_col_iP = f_col_iZ +   f_k_size;// 偏移量
                idx_t ck = ckbeg, fk = f_vec.halo_z + info.fine_base_idx[2] + (ck - ckbeg) * sz;
                {// bottom 两个
                    f_col_iZ[fk-1] = P_IZKZ[3] * c_col_iZ[ck];
                    f_col_iP[fk-1] = P_IZKZ[6] * c_col_iZ[ck]
                                    +P_IPKZ[0] * c_col_iP[ck];
                }
                for ( ; ck < ckend - 1; ck ++, fk += 2) {
                    f_col_iZ[fk  ] = P_IZKZ[4] * c_col_iZ[ck];// v(i,k) 与自己重合点
                    f_col_iZ[fk+1] = P_IZKZ[5] * c_col_iZ[ck] + P_IZKP[3] * c_col_iZ[ck+1];
                    f_col_iP[fk  ] = P_IZKZ[7] * c_col_iZ[ck] + P_IPKZ[1] * c_col_iP[ck  ];
                    f_col_iP[fk+1] = P_IZKZ[8] * c_col_iZ[ck] + P_IZKP[6] * c_col_iZ[ck+1]
                                    +P_IPKZ[2] * c_col_iP[ck] + P_IPKP[0] * c_col_iP[ck+1];
                    P_IZKZ = P_IZKP; P_IZKP += 9;
                    P_IPKZ = P_IPKP; P_IPKP += 9;
                }
                {// top 两个：没有ck+1处的粗点了
                    f_col_iZ[fk  ] = P_IZKZ[4] * c_col_iZ[ck];// v(i,k) 与自己重合点
                    f_col_iZ[fk+1] = P_IZKZ[5] * c_col_iZ[ck];
                    f_col_iP[fk  ] = P_IZKZ[7] * c_col_iZ[ck] + P_IPKZ[1] * c_col_iP[ck  ];
                    f_col_iP[fk+1] = P_IZKZ[8] * c_col_iZ[ck]
                                    +P_IPKZ[2] * c_col_iP[ck];
                }
            }
        }
        else {
            MPI_Abort(MPI_COMM_WORLD, -319);
        }
    }
    else {
        assert(false);
    }
}

#undef C_VECIDX
#undef F_VECIDX

#endif