#ifndef SMG_GMG_COARSEN_HPP
#define SMG_GMG_COARSEN_HPP

#include "GMG_types.hpp"
#include <string.h>
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

template<typename idx_t, typename data_t>
par_structMatrix<idx_t, data_t, data_t> * Galerkin_RAP_3d(RESTRICT_TYPE rstr_type, const par_structMatrix<idx_t, data_t, data_t> & fine_mat,
    PROLONG_TYPE prlg_type, const COAR_TO_FINE_INFO<idx_t> & info)
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
    
    if (fine_mat.num_diag == 7 && ((prlg_type == Plg_linear_8cell && rstr_type == Rst_8cell)
                                || (prlg_type == Plg_linear_4cell && rstr_type == Rst_4cell) )) {// 下一层粗网格也是7对角
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

    if (fine_mat.num_diag == 7 && ((prlg_type == Plg_linear_8cell && rstr_type == Rst_8cell)
                                || (prlg_type == Plg_linear_4cell && rstr_type == Rst_4cell) )) {
    // if (false) {
        if (rstr_type == Rst_4cell) {// 半粗化
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
        if (rstr_type == Rst_4cell) {// 半粗化
            assert(prlg_type == Plg_linear_4cell);
            if (my_pid == 0) printf("  using \033[1;35m3d27-Galerkin semiXY\033[0m...\n");
            RAP_3d27_semiXY(fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                hx, hy, hz
            );
        }
        else {
            assert(prlg_type == Plg_linear_8cell);
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

    if (padding_mat != nullptr) delete padding_mat;
    if (fine_mat.DiagGroups_separated) coar_mat->separate_Diags();
    return coar_mat;
}

#endif