#ifndef SMG_GMG_COARSEN_HPP
#define SMG_GMG_COARSEN_HPP

#include "GMG_types.hpp"
#include <string.h>
#include "restrict.hpp"
#include "prolong.hpp"
#include "RAP_3d27.hpp"
#include "RAP_3d7.hpp"

template<typename idx_t, typename data_t>
void XYZ_standard_coarsen(const par_structVector<idx_t, data_t> & fine_vec, const bool periodic[3], 
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_y = 0, idx_t base_z = 0)
{
    assert(stride == 2);
    // 要求进程内数据大小可以被步长整除
    assert(fine_vec.local_vector->local_x % stride == 0);
    coar_to_fine.fine_base_idx[0] = base_x;
    coar_to_fine.stride[0] = stride;

    assert(fine_vec.local_vector->local_y % stride == 0);
    coar_to_fine.fine_base_idx[1] = base_y;
    coar_to_fine.stride[1] = stride;

    assert(fine_vec.local_vector->local_z % stride == 0);
    coar_to_fine.fine_base_idx[2] = base_z;
    coar_to_fine.stride[2] = stride;

    int my_pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0)
        printf("coarsen base idx: x %d y %d z %d\n", 
            coar_to_fine.fine_base_idx[0], coar_to_fine.fine_base_idx[1], coar_to_fine.fine_base_idx[2]);
}

template<typename idx_t, typename data_t>
void XY_semi_coarsen(const par_structVector<idx_t, data_t> & fine_vec, const bool periodic[3], 
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_y = 0)
{
    assert(stride == 2);
    // 要求进程内数据大小可以被步长整除
    assert(fine_vec.local_vector->local_x % stride == 0);
    coar_to_fine.fine_base_idx[0] = base_x;
    coar_to_fine.stride[0] = stride;

    assert(fine_vec.local_vector->local_y % stride == 0);
    coar_to_fine.fine_base_idx[1] = base_y;
    coar_to_fine.stride[1] = stride;

    coar_to_fine.fine_base_idx[2] = 0;// Z方向不做粗化
    coar_to_fine.stride[2] = 1;

    int my_pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0)
        printf("coarsen base idx: x %d y %d z %d\n", 
            coar_to_fine.fine_base_idx[0], coar_to_fine.fine_base_idx[1], coar_to_fine.fine_base_idx[2]);
}

template<typename idx_t, typename data_t, typename calc_t>
par_structMatrix<idx_t, data_t, data_t> * Galerkin_RAP_3d(const Restrictor<idx_t, calc_t> & rstr, 
    const par_structMatrix<idx_t, data_t, data_t> & fine_mat, const Interpolator<idx_t, calc_t> & prlg, const COAR_TO_FINE_INFO<idx_t> & info)
{
    assert( info.stride[0] == info.stride[1] && info.stride[0] == 2);
    assert( info.fine_base_idx[0] == info.fine_base_idx[1] && 
            info.fine_base_idx[1] == info.fine_base_idx[2] && info.fine_base_idx[0] == 0);

    int my_pid; MPI_Comm_rank(fine_mat.comm_pkg->cart_comm, &my_pid);
    if (my_pid == 0) printf("using struct Galerkin(RAP)\n");
    int procs_dims = sizeof(fine_mat.comm_pkg->cart_ids) / sizeof(int); assert(procs_dims == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(fine_mat.comm_pkg->cart_comm, procs_dims, num_procs, periods, coords);

    assert(fine_mat.input_dim[0] == fine_mat.output_dim[0] && fine_mat.input_dim[1] == fine_mat.output_dim[1]
        && fine_mat.input_dim[2] == fine_mat.output_dim[2]);
    const idx_t coar_gx = fine_mat.input_dim[0] / info.stride[0],
                coar_gy = fine_mat.input_dim[1] / info.stride[1],
                coar_gz = fine_mat.input_dim[2] / info.stride[2];
    // printf("coar gx %d gy %d gz %d\n", coar_gx, coar_gy, coar_gz);
    assert(coar_gx % num_procs[1] == 0 && coar_gy % num_procs[0] == 0 && coar_gz % num_procs[2] == 0);

    par_structMatrix<idx_t, data_t, data_t> * coar_mat = nullptr;
    par_structMatrix<idx_t, data_t, data_t> * padding_mat = nullptr;
    seq_structMatrix<idx_t, data_t, data_t> * fine_mat_local = nullptr;
    seq_structMatrix<idx_t, data_t, data_t> * coar_mat_local = nullptr;
    
    if (fine_mat.num_diag == 7 && ((prlg.type == Plg_linear_8cell && rstr.type == Rst_8cell)
                                || (prlg.type == Plg_linear_4cell && rstr.type == Rst_4cell) )) {// 下一层粗网格也是7对角
        coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 7,
            coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
        fine_mat_local = fine_mat.local_matrix;
        coar_mat_local = coar_mat->local_matrix;
    }
    else {
        if (fine_mat.num_diag != 27) {// 需要补全
            padding_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 27,
                fine_mat.input_dim[0], fine_mat.input_dim[1], fine_mat.input_dim[2],
                num_procs[1], num_procs[0], num_procs[2]);
            const seq_structMatrix<idx_t, data_t, data_t> & A_part = *(fine_mat.local_matrix);
                  seq_structMatrix<idx_t, data_t, data_t> & A_full = *(padding_mat->local_matrix);
            CHECK_LOCAL_HALO(A_part, A_full);        
            assert(A_full.num_diag == 27);
            const idx_t jbeg = 0, jend = A_full.halo_y * 2 + A_full.local_y;
            const idx_t ibeg = 0, iend = A_full.halo_x * 2 + A_full.local_x;
            const idx_t kbeg = 0, kend = A_full.halo_z * 2 + A_full.local_z;

#define MATIDX(mat, k, i, j)  (k) * (mat).num_diag + (i) * (mat).slice_dk_size + (j) * (mat).slice_dki_size
            if (A_part.num_diag == 7) {
                #pragma omp parallel for collapse(3) schedule(static)
                for (idx_t j = jbeg; j < jend; j++)
                for (idx_t i = ibeg; i < iend; i++)
                for (idx_t k = kbeg; k < kend; k++) {
                    const data_t * src = A_part.data + MATIDX(A_part, k, i, j);
                    data_t       * dst = A_full.data + MATIDX(A_full, k, i, j);

                    dst[ 0] = dst[ 1] = dst[ 2] = dst[ 3] = 0.0;
                    dst[ 4] = src[0];
                    dst[ 5] = dst[ 6] = dst[ 7] = dst[ 8] = dst[ 9] = 0.0;
                    dst[10] = src[1];
                    dst[11] = 0.0;
                    dst[12] = src[2];
                    dst[13] = src[3];
                    dst[14] = src[4];
                    dst[15] = 0.0;
                    dst[16] = src[5];
                    dst[17] = dst[18] = dst[19] = dst[20] = dst[21] = 0.0;
                    dst[22] = src[6];
                    dst[23] = dst[24] = dst[25] = dst[26] = 0.0;
                }
            } else if (A_part.num_diag == 19) {
                #pragma omp parallel for collapse(3) schedule(static)
                for (idx_t j = jbeg; j < jend; j++)
                for (idx_t i = ibeg; i < iend; i++)
                for (idx_t k = kbeg; k < kend; k++) {
                    const data_t * src = A_part.data + MATIDX(A_part, k, i, j);
                    data_t       * dst = A_full.data + MATIDX(A_full, k, i, j);
                    dst[ 0] = 0.0    ; dst[ 1] = src[ 0]; dst[ 2] = 0.0    ;
                    dst[ 3] = src[ 1]; dst[ 4] = src[ 2]; dst[ 5] = src[ 3];
                    dst[ 6] = 0.0    ; dst[ 7] = src[ 4]; dst[ 8] = 0.0    ;

                    dst[ 9] = src[ 5]; dst[10] = src[ 6]; dst[11] = src[ 7];
                    dst[12] = src[ 8]; dst[13] = src[ 9]; dst[14] = src[10];
                    dst[15] = src[11]; dst[16] = src[12]; dst[17] = src[13];
                    
                    dst[18] = 0.0    ; dst[19] = src[14]; dst[20] = 0.0    ;
                    dst[21] = src[15]; dst[22] = src[16]; dst[23] = src[17];
                    dst[24] = 0.0    ; dst[25] = src[18]; dst[26] = 0.0    ;
                }
            }
#undef MATIDX
            assert(padding_mat->check_Dirichlet());
            fine_mat_local = padding_mat->local_matrix;
        } else
            fine_mat_local = fine_mat.local_matrix;
        
        coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 27,
                coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
        coar_mat_local = coar_mat->local_matrix;
    }

    // if (rstr_type != Rst_8cell) {
    //     if (my_pid == 0) printf("  using R8 to build Galerkin ...\n");
    //     rstr_type = Rst_8cell;
    // }
    // // assert(prlg_type == Plg_linear_64cell);
    // if (prlg_type != Plg_linear_64cell) {
    //     if (my_pid == 0) printf("  using P64 to build Galerkin ...\n");
    //     prlg_type = Plg_linear_64cell;
    // }
    // assert(fine_mat.num_diag == 27 && coar_mat.num_diag == 27);

    CHECK_HALO(*fine_mat_local, *coar_mat_local);
    const idx_t hx = fine_mat_local->halo_x, hy = fine_mat_local->halo_y, hz = fine_mat_local->halo_z;
    const idx_t f_lx = fine_mat_local->local_x, f_ly = fine_mat_local->local_y, f_lz = fine_mat_local->local_z,
                c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
    
    // 确定各维上是否是边界
    const bool x_lbdr = fine_mat.comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = fine_mat.comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = fine_mat.comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = fine_mat.comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = fine_mat.comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = fine_mat.comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    if (fine_mat.num_diag == 7 && ((prlg.type == Plg_linear_8cell && rstr.type == Rst_8cell)
                                || (prlg.type == Plg_linear_4cell && rstr.type == Rst_4cell) )) {
    // if (false) {
        if (rstr.type == Rst_4cell) {// 半粗化
            if (my_pid == 0) printf("  using \033[1;35m3d7-Galerkin semiXY\033[0m...\n");
            RAP_3d7_semiXY(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                hx, hy, hz
            );
        }
        else {
            if (my_pid == 0) printf("  using \033[1;35m3d7-Galerkin full\033[0m...\n");
            RAP_3d7(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                hx, hy, hz
            );
        }
       
    } else {
        if (rstr.type == Rst_4cell) {// 半粗化
            assert(prlg.type == Plg_linear_4cell);
            if (my_pid == 0) printf("  using \033[1;35m3d27-Galerkin semiXY\033[0m...\n");
            RAP_3d27_semiXY(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                hx, hy, hz
            );
        }
        else {
            assert(prlg.type == Plg_linear_8cell);
            if (my_pid == 0) printf("  using \033[1;35m3d27-Galerkin full\033[0m...\n");
            RAP_3d27(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                hx, hy, hz
            );
        }
    }

    coar_mat->update_halo();
    // Check if Dirichlet boundary condition met
    assert(coar_mat->check_Dirichlet());

    // {// 打印出来看看
    //     int num_proc; MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    //     if (my_pid == 0 && num_proc == 1) {
    //         FILE * fp = fopen("coar_mat.txt", "w+");
    //         const idx_t CX = c_lx + hx * 2,
    //                     CZ = c_lz + hz * 2;
    //         for (idx_t J = hy; J < hy + c_ly; J++)
    //         for (idx_t I = hx; I < hx + c_lx; I++)
    //         for (idx_t K = hz; K < hz + c_lz; K++) {
    //             const data_t * ptr = coar_mat_local.data + coar_mat.num_diag * (K + CZ * (I + CX * J));
    //             for (idx_t d = 0; d < coar_mat.num_diag; d++) 
    //                 fprintf(fp, " %16.10e", ptr[d]);
    //             fprintf(fp, "\n");
    //         }
    //         fclose(fp);
    //     }
    // }

    // 原来的细网格有多少个非规则点，粗网格就直接选它们都作为粗点
    idx_t num_irrgPts = fine_mat.num_irrgPts;
    coar_mat->num_irrgPts = num_irrgPts;
    coar_mat->irrgPts = new irrgPts_mat<idx_t> [num_irrgPts];
    // 确定gid
    assert(fine_mat.input_dim[0] == fine_mat.output_dim[0] && coar_mat->input_dim[0] == coar_mat->output_dim[0]
        && fine_mat.input_dim[1] == fine_mat.output_dim[1] && coar_mat->input_dim[1] == coar_mat->output_dim[1]
        && fine_mat.input_dim[2] == fine_mat.output_dim[2] && coar_mat->input_dim[2] == coar_mat->output_dim[2] );
    idx_t fine_num_struct = fine_mat.input_dim[0] * fine_mat.input_dim[1] * fine_mat.input_dim[2];
    idx_t coar_num_struct = coar_mat->input_dim[0] * coar_mat->input_dim[1] * coar_mat->input_dim[2];
    for (idx_t ir = 0; ir < num_irrgPts; ir++) {
        coar_mat->irrgPts[ir].gid = coar_num_struct + fine_mat.irrgPts[ir].gid - fine_num_struct;
    }
    // 确定nnz以及每个非零元是和哪个粗邻居相对应
    // std::vector<std::tuple<idx_t, std::tuple<idx_t,idx_t,idx_t>, data_t> > my_row;// 自己该非规则点行的非零元（别的结构点对自己的影响）
    // std::vector<std::tuple<idx_t, std::tuple<idx_t,idx_t,idx_t>, data_t> > other_row;// 别的行的非零元（自己对别的结构点的影响）
    // 由于每个非规则点既是细点，同时又是粗点，所以经自己细点的R和P一定都是1.0，
    // 所以只需要考虑自己细点的细邻居会产生什么样的R和P

    const idx_t num_max = 8;
    const idx_t off_max[num_max * 3] = {
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1
    };
    const data_t R_vals[num_max] = {rstr.a0, rstr.a1, rstr.a1, rstr.a1, rstr.a2, rstr.a2, rstr.a2, rstr.a3};
    const data_t P_vals[num_max] = {prlg.a0, prlg.a1, prlg.a1, prlg.a1, prlg.a2, prlg.a2, prlg.a2, prlg.a3};
    idx_t R_num = -1;// 一个细点经限制而贡献到的粗点数
    if      (rstr.type == Rst_8cell ) R_num = 1;
    else if (rstr.type == Rst_64cell) R_num = 8;
    idx_t P_num = -1;
    if      (prlg.type == Plg_linear_8cell ) P_num = 1;
    else if (prlg.type == Plg_linear_64cell) P_num = 8;
    assert(R_num != -1 && P_num != -1);
    // 用全局的一维索引记录
    std::unordered_map<idx_t, data_t> my_row;
    std::unordered_map<idx_t, data_t> my_col;
    std::vector<std::tuple<idx_t, idx_t, data_t, data_t> > collect;
    for (idx_t ir = 0; ir < num_irrgPts; ir++) {
        my_row.clear();
        my_col.clear();

        idx_t my_coar_gid = coar_mat->irrgPts[ir].gid;
        idx_t pbeg = fine_mat.irrgPts[ir].beg, pend = pbeg + fine_mat.irrgPts[ir].nnz;
        for (idx_t p = pbeg; p < pend; p++) {// 遍历自己细点的每个细邻居
            const idx_t fi_s_ngb = fine_mat.irrgPts_ngb_ijk[p*3  ],
                        fj_s_ngb = fine_mat.irrgPts_ngb_ijk[p*3+1],
                        fk_s_ngb = fine_mat.irrgPts_ngb_ijk[p*3+2];// 该细邻居的全局三维坐标
            if (fi_s_ngb == -1) {// 细邻居是自己
                assert(fj_s_ngb == -1 && fk_s_ngb == -1);
                data_t val =  fine_mat.irrgPts_A_vals[p*2  ];
                assert(val == fine_mat.irrgPts_A_vals[p*2+1]);
                my_row[my_coar_gid] += val;
                my_col[my_coar_gid] += val;
                continue;
            }
            assert(fine_mat.offset_x <= fi_s_ngb && fi_s_ngb < fine_mat.offset_x + fine_mat.local_matrix->local_x
                && fine_mat.offset_y <= fj_s_ngb && fj_s_ngb < fine_mat.offset_y + fine_mat.local_matrix->local_y
                && fine_mat.offset_z <= fk_s_ngb && fk_s_ngb < fine_mat.offset_z + fine_mat.local_matrix->local_z);
            const idx_t ci_s_ngb = fi_s_ngb >> 1,
                        cj_s_ngb = fj_s_ngb >> 1,
                        ck_s_ngb = fk_s_ngb >> 1;// 该细邻居对应的某个最邻近的粗点的全局三维坐标
            const idx_t sign_off_ci = fi_s_ngb == (ci_s_ngb << 1) ? -1 : 1,
                        sign_off_cj = fj_s_ngb == (cj_s_ngb << 1) ? -1 : 1,
                        sign_off_ck = fk_s_ngb == (ck_s_ngb << 1) ? -1 : 1;
            assert(coar_mat->offset_x <= ci_s_ngb && ci_s_ngb < coar_mat->offset_x + coar_mat->local_matrix->local_x
                && coar_mat->offset_y <= cj_s_ngb && cj_s_ngb < coar_mat->offset_y + coar_mat->local_matrix->local_y
                && coar_mat->offset_z <= ck_s_ngb && ck_s_ngb < coar_mat->offset_z + coar_mat->local_matrix->local_z);
#ifdef DEBUG
            printf("ir %d f_s_ngb (%d,%d,%d) c_s_ngb (%d,%d,%d) sign (%d,%d,%d)\n", ir,
                fi_s_ngb, fj_s_ngb, fk_s_ngb, ci_s_ngb, cj_s_ngb, ck_s_ngb, 
                sign_off_ci, sign_off_cj, sign_off_ck);
#endif
            // 找可能从这个细邻居R的粗点
            for (idx_t j = 0; j < R_num; j++) {
                idx_t ciR = ci_s_ngb + sign_off_ci * off_max[j*3  ];
                idx_t cjR = cj_s_ngb + sign_off_cj * off_max[j*3+1];
                idx_t ckR = ck_s_ngb + sign_off_ck * off_max[j*3+2];
                // if (ciR < 0 || ciR >= coar_mat->input_dim[0] || cjR < 0 || cjR >= coar_mat->input_dim[1] || 
                //     ckR < 0 || ckR >= coar_mat->input_dim[2] ) continue;// 边界
                if (coar_mat->offset_x > ciR || ciR >= coar_mat->offset_x + coar_mat->local_matrix->local_x ||
                    coar_mat->offset_y > cjR || cjR >= coar_mat->offset_y + coar_mat->local_matrix->local_y ||
                    coar_mat->offset_z > ckR || ckR >= coar_mat->offset_z + coar_mat->local_matrix->local_z)
                    continue;// 出了本进程边界，就不管了
                // 注意j在最外维 (j*NX + i)*NZ + k
                idx_t idxR = (cjR * coar_mat->input_dim[0] + ciR) * coar_mat->input_dim[2] + ckR;
#ifdef DEBUG
                printf(" ciR %d cjR %d ckR %d idxR %d\n", ciR, cjR, ckR, idxR);
#endif
                my_col[idxR] += fine_mat.irrgPts_A_vals[p*2+1] * R_vals[j];
            }           
            // 找可能P到这个细邻居的粗点
            for (idx_t j = 0; j < P_num; j++) {
                idx_t ciP = ci_s_ngb + sign_off_ci * off_max[j*3  ];
                idx_t cjP = cj_s_ngb + sign_off_cj * off_max[j*3+1];
                idx_t ckP = ck_s_ngb + sign_off_ck * off_max[j*3+2];
                // if (ciP < 0 || ciP >= coar_mat->input_dim[0] || cjP < 0 || cjP >= coar_mat->input_dim[1] || 
                //     ckP < 0 || ckP >= coar_mat->input_dim[2] ) continue;// 边界
                if (coar_mat->offset_x > ciP || ciP >= coar_mat->offset_x + coar_mat->local_matrix->local_x ||
                    coar_mat->offset_y > cjP || cjP >= coar_mat->offset_y + coar_mat->local_matrix->local_y ||
                    coar_mat->offset_z > ckP || ckP >= coar_mat->offset_z + coar_mat->local_matrix->local_z)
                    continue;// 出了本进程边界，就不管了
                // 注意j在最外维
                idx_t idxP = (cjP * coar_mat->input_dim[0] + ciP) * coar_mat->input_dim[2] + ckP;
#ifdef DEBUG
                printf(" ciP %d cjP %d ckP %d idxP %d\n", ciP, cjP, ckP, idxP);
#endif
                my_row[idxP] += fine_mat.irrgPts_A_vals[p*2  ] * P_vals[j];
            }
        }

        idx_t my_nnz = 0;
        for (typename std::unordered_map<idx_t,data_t>::iterator it = my_row.begin(); it != my_row.end(); it++) {
            idx_t target = it->first;
            data_t from = it->second;
            data_t to = 0.0;
            if (my_col.find(target) != my_col.end()) {
                to = my_col[target];
                my_col.erase(target);
            }
            std::tuple<idx_t, idx_t, data_t, data_t> obj(my_coar_gid, target, from, to);
            collect.push_back(obj);
            my_nnz ++;
        }
        // 如果my_col仍有剩余
        for (typename std::unordered_map<idx_t,data_t>::iterator it = my_col.begin(); it != my_col.end(); it++) {
            idx_t target = it->first;
            data_t to = it->second;
            std::tuple<idx_t, idx_t, data_t, data_t> obj(my_coar_gid, target, 0.0, to);
            collect.push_back(obj);
            my_nnz ++;
        }

        coar_mat->irrgPts[ir].beg = (ir == 0) ? 0 : coar_mat->irrgPts[ir-1].beg + coar_mat->irrgPts[ir-1].nnz;
        coar_mat->irrgPts[ir].nnz = my_nnz;
        // printf(" Done ir %d with %d nnz\n", ir, coar_mat->irrgPts[ir].nnz);
    }
    // 分段做插入排序
    for (idx_t ir = 0; ir < num_irrgPts; ir++) {
        idx_t pbeg = coar_mat->irrgPts[ir].beg, pend = pbeg + coar_mat->irrgPts[ir].nnz;
        for (idx_t p = pbeg + 1; p < pend; p++) {
            if (std::get<1>(collect[p]) < std::get<1>(collect[p-1])) {// 存在逆序
                std::tuple<idx_t, idx_t, data_t, data_t> obj = collect[p];
                idx_t loc;
                for (loc = p - 1; loc >= pbeg && std::get<1>(obj) < std::get<1>(collect[loc]); loc--)
                    collect[loc + 1] = collect[loc];
                collect[loc + 1] = obj;
            }
        }
    }
    // 转移到矩阵数据结构中
    idx_t tot_nnz = collect.size();
    coar_mat->irrgPts_ngb_ijk = new idx_t [tot_nnz * 3];
    coar_mat->irrgPts_A_vals = new data_t [tot_nnz * 2];
    for (idx_t p = 0, ir = 0; p < tot_nnz; p++) {
        idx_t my_coar_gid = std::get<0>(collect[p]);
        assert(my_coar_gid == coar_mat->irrgPts[ir].gid);
        idx_t ngb_coar_gid = std::get<1>(collect[p]);// 邻居结构点的全局索引
        idx_t cj, ci, ck;
        if (ngb_coar_gid == my_coar_gid) {
            cj = ci = ck = -1;
        } else {
            // 注意j在最外维，i在中间维 (j*NX + i)*NZ + k
            cj =  ngb_coar_gid / (coar_mat->input_dim[0] * coar_mat->input_dim[2]);
            ci = (ngb_coar_gid -  cj * coar_mat->input_dim[0] * coar_mat->input_dim[2]) / coar_mat->input_dim[2];
            ck =  ngb_coar_gid - (cj * coar_mat->input_dim[0] + ci) * coar_mat->input_dim[2];
        }
        coar_mat->irrgPts_ngb_ijk[p*3  ] = ci;
        coar_mat->irrgPts_ngb_ijk[p*3+1] = cj;
        coar_mat->irrgPts_ngb_ijk[p*3+2] = ck;
        coar_mat->irrgPts_A_vals[p*2  ] = std::get<2>(collect[p]);
        coar_mat->irrgPts_A_vals[p*2+1] = std::get<3>(collect[p]);
        if (p == coar_mat->irrgPts[ir].beg + coar_mat->irrgPts[ir].nnz - 1) ir++;
    }
#ifdef DEBUG
    // just print
    for (idx_t ir = 0; ir < num_irrgPts; ir++) {
        idx_t pbeg = coar_mat->irrgPts[ir].beg, pend = pbeg + coar_mat->irrgPts[ir].nnz;
        printf(" proc %d got %d: \n", my_pid, coar_mat->irrgPts[ir].gid);
        for (idx_t p = pbeg; p < pend; p++) {
            printf("    (%d,%d,%d) %.8e %.8e\n",
                coar_mat->irrgPts_ngb_ijk[p*3  ], coar_mat->irrgPts_ngb_ijk[p*3+1], coar_mat->irrgPts_ngb_ijk[p*3+2], 
                coar_mat->irrgPts_A_vals [p*2  ], coar_mat->irrgPts_A_vals [p*2+1]);
        }
    }
#endif

    if (padding_mat != nullptr) delete padding_mat;
    if (fine_mat.DiagGroups_separated) coar_mat->separate_Diags();
    return coar_mat;
}

#endif