#ifndef SMG_SEQ_STRUCT_MV_HPP
#define SMG_SEQ_STRUCT_MV_HPP

#include "common.hpp"

// 向量只需要一种精度
template<typename idx_t, typename data_t>
class seq_structVector {
public:
    const idx_t local_x;// lon方向的格点数(仅含计算区域)
    const idx_t local_y;// lat方向的格点数(仅含计算区域)
    const idx_t local_z;// 垂直方向的格点数(仅含计算区域)
    const idx_t halo_x;// lon方向的halo区宽度
    const idx_t halo_y;// lat方向的halo区宽度
    const idx_t halo_z;// 垂直方向的halo区宽度
    data_t * data;

    // 数据存储顺序从内到外为(k, j, i)
    idx_t slice_k_size;
    idx_t slice_ki_size;

    seq_structVector(idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz);
    // 拷贝构造函数，开辟同样规格的data
    seq_structVector(const seq_structVector & model);
    ~seq_structVector();

    void init_debug(idx_t off_x, idx_t off_y, idx_t off_z);
    void print_level(idx_t ilev);

    void operator=(data_t val);
    void set_halo(data_t val);
};

// 矩阵需要两种精度：数据的存储精度data_t，和计算时的精度calc_t
template<typename idx_t, typename data_t, typename calc_t>
class seq_structMatrix {
public:
    idx_t num_diag;// 矩阵对角线数（19）
    idx_t local_x;// lon方向的格点数(仅含计算区域)
    idx_t local_y;// lat方向的格点数(仅含计算区域)
    idx_t local_z;// 垂直方向的格点数(仅含计算区域)
    idx_t halo_x;// lon方向的halo区宽度
    idx_t halo_y;// lat方向的halo区宽度
    idx_t halo_z;// 垂直方向的halo区宽度
    data_t * data;

    // 数据存储顺序从内到外为(diag, k, j, i)
    idx_t slice_dk_size;
    idx_t slice_dki_size;

    seq_structMatrix(idx_t num_d, idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz);
    // 拷贝构造函数，开辟同样规格的data
    seq_structMatrix(const seq_structMatrix & model);
    ~seq_structMatrix();

    void init_debug(idx_t off_x, idx_t off_y, idx_t off_z);
    void print_level_diag(idx_t ilev, idx_t idiag);
    void operator=(data_t val);
    void set_diag_val(idx_t d, data_t val);

    void extract_diag(idx_t idx_diag) const;
    void truncate() {
#ifdef __aarch64__
        idx_t tot_len = (2 * halo_y + local_y) * (2 * halo_x + local_x) * (2 * halo_z + local_z) * num_diag;
        for (idx_t i = 0; i < tot_len; i++) {
            __fp16 tmp = (__fp16) data[i];
            // if (i == local_x * local_y * local_z * 4) printf("seqMat truncate %.20e to", data[i]);
            data[i] = (data_t) tmp;
            // if (i == local_x * local_y * local_z * 4) printf("%.20e\n", data[i]);
        }
#else
        printf("architecture not support truncated to fp16\n");
#endif
    }

    void mul(const data_t factor) {
        idx_t tot_len = (local_x + halo_x * 2) * (local_y + halo_y * 2) * (local_z + halo_z * 2) * num_diag;
        for (idx_t i = 0; i < tot_len; i++)
            data[i] *= factor;
    }

    // 矩阵接受的SpMV运算是以操作精度的
    void Mult(const seq_structVector<idx_t, calc_t> & x, seq_structVector<idx_t, calc_t> & y,
            const seq_structVector<idx_t, calc_t> * sqrtD_ptr = nullptr) const;
    void (*spmv)(const idx_t num, const idx_t vec_k_size, const idx_t vec_ki_size,
        const data_t * A_jik, const data_t * x_jik, data_t * y_jik, const data_t * dummy) = nullptr;
    void (*spmv_scaled)(const idx_t num, const idx_t vec_k_size, const idx_t vec_ki_size,
        const data_t * A_jik, const data_t * x_jik, data_t * y_jik, const data_t * sqD_jik) = nullptr;
};

/*
 * * * * * seq_structVetor * * * * * 
 */

template<typename idx_t, typename data_t>
seq_structVector<idx_t, data_t>::seq_structVector(idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz)
    : local_x(lx), local_y(ly), local_z(lz), halo_x(hx), halo_y(hy), halo_z(hz)
{
    idx_t   tot_x = local_x + 2 * halo_x,
            tot_y = local_y + 2 * halo_y,
            tot_z = local_z + 2 * halo_z;
    data = new data_t[tot_x * tot_y * tot_z];
#ifdef DEBUG
    for (idx_t i = 0; i < tot_x * tot_y * tot_z; i++) data[i] = -99;
#endif
    slice_k_size  = local_z + 2 * halo_z;
    slice_ki_size = slice_k_size * (local_x + 2 * halo_x);
}

template<typename idx_t, typename data_t>
seq_structVector<idx_t, data_t>::seq_structVector(const seq_structVector & model)
    : local_x(model.local_x), local_y(model.local_y), local_z(model.local_z),
      halo_x (model.halo_x) , halo_y (model.halo_y ), halo_z (model.halo_z) , 
      slice_k_size(model.slice_k_size), slice_ki_size(model.slice_ki_size)
{
    idx_t   tot_x = local_x + 2 * halo_x,
            tot_y = local_y + 2 * halo_y,
            tot_z = local_z + 2 * halo_z;
    data = new data_t[tot_x * tot_y * tot_z];
}

template<typename idx_t, typename data_t>
seq_structVector<idx_t, data_t>::~seq_structVector() {
    delete data;
    data = nullptr;
}

template<typename idx_t, typename data_t>
void seq_structVector<idx_t, data_t>::init_debug(idx_t off_x, idx_t off_y, idx_t off_z) 
{
    idx_t tot = slice_ki_size * (local_y + 2 * halo_y);
    for (idx_t i = 0; i < tot; i++)
        data[i] = 0.0;

    idx_t xbeg = halo_x, xend = xbeg + local_x,
            ybeg = halo_y, yend = ybeg + local_y,
            zbeg = halo_z, zend = zbeg + local_z;
    for (idx_t j = ybeg; j < yend; j++) {
        for (idx_t i = xbeg; i < xend; i++) {
            for (idx_t k = zbeg; k < zend; k++) {
                data[k + i * slice_k_size + j * slice_ki_size] 
                    = 100.0 * (off_x + i - xbeg) + off_y + j - ybeg + 1e-2 * (off_z + k - zbeg);
            }
        }
    }
}

template<typename idx_t, typename data_t>
void seq_structVector<idx_t, data_t>::print_level(idx_t ilev) 
{
    assert(ilev >= 0 && ilev < local_z);
    idx_t xbeg = 0, xend = xbeg + local_x + 2 * halo_x,
            ybeg = 0, yend = ybeg + local_y + 2 * halo_y;
    printf("lev %d: \n", ilev);
    for (idx_t j = ybeg; j < yend; j++) {
        for (idx_t i = xbeg; i < xend; i++) {
            printf("%12.7e ", data[ilev + i * slice_k_size + j * slice_ki_size]);
        }
        printf("\n");
    }
}

template<typename idx_t, typename data_t, typename res_t>
res_t seq_vec_dot(const seq_structVector<idx_t, data_t> & x, const seq_structVector<idx_t, data_t> & y) {
    CHECK_LOCAL_HALO(x, y);

    const idx_t xbeg = x.halo_x, xend = xbeg + x.local_x,
                ybeg = x.halo_y, yend = ybeg + x.local_y,
                zbeg = x.halo_z, zend = zbeg + x.local_z;
    const idx_t slice_k_size = x.slice_k_size, slice_ki_size = x.slice_ki_size;
    res_t dot = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:dot) schedule(static)
    for (idx_t j = ybeg; j < yend; j++)
    for (idx_t i = xbeg; i < xend; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        const data_t * x_data = x.data + ji_loc, * y_data = y.data + ji_loc;
        for (idx_t k = zbeg; k < zend; k++)
            dot += (res_t) x_data[k] * (res_t) y_data[k];
    }
    return dot;
}

template<typename idx_t, typename data_t, typename scalar_t>
void seq_vec_add(const seq_structVector<idx_t, data_t> & v1, scalar_t alpha, 
                 const seq_structVector<idx_t, data_t> & v2, seq_structVector<idx_t, data_t> & v) 
{
    CHECK_LOCAL_HALO(v1, v2);
    CHECK_LOCAL_HALO(v1, v );
    
    const data_t *  v1_data = v1.data;
    const data_t *  v2_data = v2.data;
          data_t * res_data = v.data;
    
    const idx_t ibeg = v1.halo_x, iend = ibeg + v1.local_x,
                jbeg = v1.halo_y, jend = jbeg + v1.local_y,
                kbeg = v1.halo_z, kend = kbeg + v1.local_z;
    const idx_t vec_k_size = v1.slice_k_size, vec_ki_size = v1.slice_ki_size;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_ki_size + i * vec_k_size;
        for (idx_t k = kbeg; k < kend; k++)
            res_data[ji_loc + k] = v1_data[ji_loc + k] + alpha * v2_data[ji_loc + k];
    }
}

template<typename idx_t, typename data_t>
void seq_vec_copy(const seq_structVector<idx_t, data_t> & src, seq_structVector<idx_t, data_t> & dst)
{
    CHECK_LOCAL_HALO(src, dst);
    
    const data_t * src_data = src.data;
          data_t * dst_data = dst.data;
    const idx_t ibeg = src.halo_x, iend = ibeg + src.local_x,
                jbeg = src.halo_y, jend = jbeg + src.local_y,
                kbeg = src.halo_z, kend = kbeg + src.local_z;
    const idx_t vec_k_size = src.slice_k_size, vec_ki_size = src.slice_ki_size;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_ki_size + i * vec_k_size;
        for (idx_t k = kbeg; k < kend; k++)
            dst_data[ji_loc + k] = src_data[ji_loc + k];
    }
}

template<typename idx_t, typename data_t>
void seq_structVector<idx_t, data_t>::operator=(data_t val) {
    idx_t   xbeg = halo_x, xend = xbeg + local_x,
            ybeg = halo_y, yend = ybeg + local_y,
            zbeg = halo_z, zend = zbeg + local_z;
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = ybeg; j < yend; j++)
    for (idx_t i = xbeg; i < xend; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        for (idx_t k = zbeg; k < zend; k++)
                data[ji_loc + k] = val;
    }
}

template<typename idx_t, typename data_t>
void seq_structVector<idx_t, data_t>::set_halo(data_t val) {
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = 0; j < halo_y; j++)
    for (idx_t i = 0; i < halo_x * 2 + local_x; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            data[ji_loc + k] = val;
    }

    for (idx_t j = halo_y; j < halo_y + local_y; j++)
    for (idx_t i = 0; i < halo_x; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            data[ji_loc + k] = val;
    }
        
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = halo_y; j < halo_y + local_y; j++) 
    for (idx_t i = halo_x; i < halo_x + local_x; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        for (idx_t k = 0; k < halo_z; k++)
            data[ji_loc + k] = val;
        for (idx_t k = halo_z + local_z; k < halo_z * 2 + local_z; k++)
            data[ji_loc + k] = val;
    }

    for (idx_t j = halo_y; j < halo_y + local_y; j++)
    for (idx_t i = halo_x + local_x; i < halo_x * 2 + local_x; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            data[ji_loc + k] = val;
    }

    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = halo_y + local_y; j < halo_y * 2 + local_y; j++)
    for (idx_t i = 0; i < halo_x * 2 + local_x; i++) {
        idx_t ji_loc = j * slice_ki_size + i * slice_k_size;
        for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
            data[ji_loc + k] = val;
    }

#ifdef DEBUG
    for (idx_t j = 0; j < halo_y * 2 + local_y; j++)
        for (idx_t i = 0; i < halo_x * 2 + local_x; i++)
            for (idx_t k = 0; k < halo_z * 2 + local_z; k++)
                if (data[k + i * slice_k_size + j * slice_ki_size] != 0.0) {
                    printf("%d %d %d %.5e\n", j, i, k, data[k + i * slice_k_size + j * slice_ki_size]);
                }
#endif
}

template<typename idx_t, typename data_t, typename scalar_t>
void seq_vec_mul_by_scalar(const scalar_t coeff, const seq_structVector<idx_t, data_t> & src, seq_structVector<idx_t, data_t> & dst) 
{
    CHECK_LOCAL_HALO(src, dst);
    
    const data_t * src_data = src.data;
          data_t * dst_data = dst.data;
    const idx_t ibeg = src.halo_x, iend = ibeg + src.local_x,
                jbeg = src.halo_y, jend = jbeg + src.local_y,
                kbeg = src.halo_z, kend = kbeg + src.local_z;
    const idx_t vec_k_size = src.slice_k_size, vec_ki_size = src.slice_ki_size;

    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_ki_size + i * vec_k_size;
        for (idx_t k = kbeg; k < kend; k++)
            dst_data[ji_loc + k] = coeff * src_data[ji_loc + k];
    }
}

template<typename idx_t, typename data_t, typename scalar_t>
void seq_vec_scale(const scalar_t coeff, seq_structVector<idx_t, data_t> & vec) {
    
    data_t * data = vec.data;
    const idx_t ibeg = vec.halo_x, iend = ibeg + vec.local_x,
                jbeg = vec.halo_y, jend = jbeg + vec.local_y,
                kbeg = vec.halo_z, kend = kbeg + vec.local_z;
    const idx_t vec_k_size = vec.slice_k_size, vec_ki_size = vec.slice_ki_size;

    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t ji_loc = j * vec_ki_size + i * vec_k_size;
        for (idx_t k = kbeg; k < kend; k++)
            data[ji_loc + k] *= coeff;
    }
}

template<typename idx_t, typename data_t1, typename data_t2>
void seq_vec_elemwise_mul(seq_structVector<idx_t, data_t1> & inout_vec, const seq_structVector<idx_t, data_t2> & scaleplate)
{
    CHECK_LOCAL_HALO(inout_vec, scaleplate);
    const idx_t jbeg = inout_vec.halo_y, jend = jbeg + inout_vec.local_y,
                ibeg = inout_vec.halo_x, iend = ibeg + inout_vec.local_x,
                kbeg = inout_vec.halo_z, kend = kbeg + inout_vec.local_z;
    const idx_t vec_ki_size = inout_vec.slice_ki_size, vec_k_size = inout_vec.slice_k_size;
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t offset = j * vec_ki_size + i * vec_k_size;
        data_t1 * dst_ptr = inout_vec.data + offset;
        const data_t2 * src_ptr = scaleplate.data + offset;
        for (idx_t k = kbeg; k < kend; k++) 
            dst_ptr[k] *= src_ptr[k];
    }
}

template<typename idx_t, typename data_t1, typename data_t2>
void seq_vec_elemwise_div(seq_structVector<idx_t, data_t1> & inout_vec, const seq_structVector<idx_t, data_t2> & scaleplate)
{
    CHECK_LOCAL_HALO(inout_vec, scaleplate);
    const idx_t jbeg = inout_vec.halo_y, jend = jbeg + inout_vec.local_y,
                ibeg = inout_vec.halo_x, iend = ibeg + inout_vec.local_x,
                kbeg = inout_vec.halo_z, kend = kbeg + inout_vec.local_z;
    const idx_t vec_ki_size = inout_vec.slice_ki_size, vec_k_size = inout_vec.slice_k_size;
    #pragma omp parallel for collapse(2) schedule(static) 
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        idx_t offset = j * vec_ki_size + i * vec_k_size;
        data_t1 * dst_ptr = inout_vec.data + offset;
        const data_t2 * src_ptr = scaleplate.data + offset;
        for (idx_t k = kbeg; k < kend; k++) 
            dst_ptr[k] /= src_ptr[k];
    }
}

/*
 * * * * * seq_structMatrix * * * * * 
 */
#include "kernels_3d7.hpp"
#include "kernels_3d19.hpp"
#include "kernels_3d27.hpp"
template<typename idx_t, typename data_t, typename calc_t>
seq_structMatrix<idx_t, data_t, calc_t>::seq_structMatrix(idx_t num_d, idx_t lx, idx_t ly, idx_t lz, idx_t hx, idx_t hy, idx_t hz)
    : num_diag(num_d), local_x(lx), local_y(ly), local_z(lz), halo_x(hx), halo_y(hy), halo_z(hz)
{
    idx_t tot = num_diag * (local_x + 2 * halo_x) * (local_y + 2 * halo_y) * (local_z + 2 * halo_z);
    data = new data_t[tot];
#ifdef DEBUG
    for (idx_t i = 0; i < tot; i++) data[i] = -9999.9;
#endif
    slice_dk_size  = num_diag * (local_z + 2 * halo_z);
    slice_dki_size = slice_dk_size * (local_x + 2 * halo_x);
    switch (num_diag)
    {
    case  7: spmv = AOS_spmv_3d7 <idx_t, data_t>; spmv_scaled = AOS_spmv_3d7_scaled<idx_t, data_t>; break;
    case 19: spmv = AOS_spmv_3d19<idx_t, data_t>; break;
    case 27: spmv = AOS_spmv_3d27<idx_t, data_t>; break;
    default: break;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
seq_structMatrix<idx_t, data_t, calc_t>::seq_structMatrix(const seq_structMatrix & model)
    : num_diag(model.num_diag),
      local_x(model.local_x), local_y(model.local_y), local_z(model.local_z),
      halo_x (model.halo_x) , halo_y (model.halo_y ), halo_z (model.halo_z) , 
      slice_dk_size(model.slice_dk_size), slice_dki_size(model.slice_dki_size)
{
    idx_t tot = num_diag * (local_x + 2 * halo_x) * (local_y + 2 * halo_y) * (local_z + 2 * halo_z);
    data = new data_t[tot];
}

template<typename idx_t, typename data_t, typename calc_t>
seq_structMatrix<idx_t, data_t, calc_t>::~seq_structMatrix() {
    delete data;
    data = nullptr;
}

template<typename idx_t, typename data_t, typename calc_t>
void seq_structMatrix<idx_t, data_t, calc_t>::print_level_diag(idx_t ilev, idx_t idiag) 
{
    assert(ilev >= 0 && ilev < local_z && idiag >= 0 && idiag < num_diag);
    idx_t   xbeg = 0, xend = xbeg + local_x + 2 * halo_x,
            ybeg = 0, yend = ybeg + local_y + 2 * halo_y;
    printf("lev %d with %d-th diag: \n", ilev, idiag);
    for (idx_t j = ybeg; j < yend; j++) {
        for (idx_t i = xbeg; i < xend; i++) {
            printf("%12.6f ", data[idiag + ilev * num_diag + i * slice_dk_size + j * slice_dki_size]);
        }
        printf("\n");
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void seq_structMatrix<idx_t, data_t, calc_t>::Mult(
    const seq_structVector<idx_t, calc_t> & x, seq_structVector<idx_t, calc_t> & y,
    const seq_structVector<idx_t, calc_t> * sqrtD_ptr) const
{
    CHECK_LOCAL_HALO(*this, x);
    CHECK_LOCAL_HALO(x , y);
    const data_t* mat_data = data, 
                * aux_data = (sqrtD_ptr) ? sqrtD_ptr->data : nullptr;
    void (*kernel)(const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, data_t*, const data_t*)
        = (sqrtD_ptr) ? spmv_scaled : spmv;

    const calc_t * x_data = x.data;
    calc_t * y_data = y.data;

    idx_t   ibeg = halo_x, iend = ibeg + local_x,
            jbeg = halo_y, jend = jbeg + local_y,
            kbeg = halo_z, kend = kbeg + local_z;
    idx_t vec_k_size = x.slice_k_size, vec_ki_size = x.slice_ki_size;
    const idx_t col_height = kend - kbeg;

    #pragma omp parallel for collapse(2) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++) {
        const data_t * A_jik = mat_data + j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
        const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
        const data_t * aux_jik = (sqrtD_ptr) ? (aux_data + vec_off) : nullptr;
        const calc_t * x_jik = x_data + vec_off;
        calc_t * y_jik = y_data + vec_off;
        kernel(col_height, vec_k_size, vec_ki_size, A_jik, x_jik, y_jik, aux_jik);
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void seq_structMatrix<idx_t, data_t, calc_t>::operator=(data_t val) 
{
    for (idx_t j = 0; j < halo_y * 2 + local_y; j++) 
    for (idx_t i = 0; i < halo_x * 2 + local_x; i++)
    for (idx_t k = 0; k < halo_z * 2 + local_z; k++) 
    for (idx_t d = 0; d < num_diag; d++) {
        idx_t loc = d + k * num_diag + i * slice_dk_size + j * slice_dki_size;
        data[loc] = val;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void seq_structMatrix<idx_t, data_t, calc_t>::set_diag_val(idx_t d, data_t val) 
{
    const idx_t jbeg = halo_y, jend = jbeg + local_y,
                ibeg = halo_x, iend = ibeg + local_x,
                kbeg = halo_z, kend = kbeg + local_z;
    for (idx_t j = jbeg; j < jend; j++) 
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        idx_t loc = d + k * num_diag + i * slice_dk_size + j * slice_dki_size;
        data[loc] = val;
    }
}

#endif