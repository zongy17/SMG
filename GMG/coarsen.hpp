#ifndef SMG_GMG_COARSEN_HPP
#define SMG_GMG_COARSEN_HPP

#include "restrict.hpp"
#include "prolong.hpp"
#include <string.h>
#include "RAP_3d19.hpp"
#include "RAP_3d27.hpp"
#include "RAP_3d7.hpp"
#include "RAP_2d9.hpp"

template<typename idx_t, typename data_t>
void XYZ_standard_coarsen(const par_structVector<idx_t, data_t> & fine_vec, const bool periodic[3], 
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_y = 0, idx_t base_z = 0)
{
    coar_to_fine.type = FULL_XYZ;
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
}

template<typename idx_t, typename data_t>
void XY_semi_coarsen(const par_structVector<idx_t, data_t> & fine_vec, const bool periodic[3], 
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_y = 0)
{
    coar_to_fine.type = SEMI_XY;
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
}

template<typename idx_t, typename data_t>
void XZ_decp_coarsen(const par_structVector<idx_t, data_t> & fine_vec, const bool periodic[2],
    COAR_TO_FINE_INFO<idx_t> & coar_to_fine, idx_t stride = 2, idx_t base_x = 0, idx_t base_z = 0)
{
    coar_to_fine.type = DECP_XZ;
    assert(stride == 2);

    assert(fine_vec.local_vector->local_x % stride == 0);
    coar_to_fine.fine_base_idx[0] = base_x;
    coar_to_fine.stride[0] = stride;

    // y方向（lat）不动
    coar_to_fine.fine_base_idx[1] = 0;
    coar_to_fine.stride[1] = 1;

    // z方向，一般不满足细网格点数目的要求，则不能为周期性
    assert(periodic[2] == false);
    coar_to_fine.fine_base_idx[2] = base_z;// 非周期性的粗点从边界的下一个点开始取
    coar_to_fine.stride[2] = stride;
}


template<typename idx_t, typename data_t>
par_structMatrix<idx_t, data_t, data_t> * Galerkin_RAP(const Restrictor<idx_t, data_t> & rstr, const par_structMatrix<idx_t, data_t, data_t> & fine_mat,
    const Interpolator<idx_t, data_t> & prlg, const COAR_TO_FINE_INFO<idx_t> & info)
{
    par_structMatrix<idx_t, data_t, data_t> * coar_mat = nullptr;
    seq_structMatrix<idx_t, data_t, data_t> * fine_mat_local = fine_mat.local_matrix;
    seq_structMatrix<idx_t, data_t, data_t> * coar_mat_local = nullptr;

    int my_pid; MPI_Comm_rank(fine_mat.comm_pkg->cart_comm, &my_pid);
    int procs_dims = sizeof(fine_mat.comm_pkg->cart_ids) / sizeof(int); assert(procs_dims == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(fine_mat.comm_pkg->cart_comm, procs_dims, num_procs, periods, coords);

    assert(fine_mat.input_dim[0] == fine_mat.output_dim[0] && fine_mat.input_dim[1] == fine_mat.output_dim[1]
        && fine_mat.input_dim[2] == fine_mat.output_dim[2]);
    const idx_t hx = fine_mat_local->halo_x, hy = fine_mat_local->halo_y, hz = fine_mat_local->halo_z;
    const idx_t f_lx = fine_mat_local->local_x, f_ly = fine_mat_local->local_y, f_lz = fine_mat_local->local_z;
    // 确定各维上是否是边界
    const bool x_lbdr = fine_mat.comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = fine_mat.comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = fine_mat.comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = fine_mat.comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = fine_mat.comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = fine_mat.comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    const idx_t coar_gx = fine_mat.input_dim[0] / info.stride[0],
                coar_gy = fine_mat.input_dim[1] / info.stride[1],
                coar_gz = fine_mat.input_dim[2] / info.stride[2];
    // printf("coar gx %d gy %d gz %d\n", coar_gx, coar_gy, coar_gz);
    assert(coar_gx % num_procs[1] == 0 && coar_gy % num_procs[0] == 0 && coar_gz % num_procs[2] == 0);

    if (info.type == FULL_XYZ) {// 全三维粗化
        assert(rstr.type == Cell_3d8 || rstr.type == Cell_3d64);
        assert(prlg.type == Cell_3d8 || prlg.type == Cell_3d64);
        if (fine_mat.num_diag == 7 && (prlg.type == Cell_3d8 && rstr.type == Cell_3d8)) {
            // 下一层粗网格仍然为3d7的形式
            coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm,  7,
                coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
            coar_mat_local = coar_mat->local_matrix;
            CHECK_HALO(*fine_mat_local, *coar_mat_local);
            const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
            if (my_pid == 0) printf("  using \033[1;35mRCell3d8_A3d7_PCell3d8\033[0m...\n");
            RCell3d8_A3d7_PCell3d8(
                fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                hx, hy, hz
            );
        }
        else {// 下一层会膨胀至3d27
            coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 27,
                coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
            coar_mat_local = coar_mat->local_matrix;
            CHECK_HALO(*fine_mat_local, *coar_mat_local);
            // const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
            // 需要为每一种特定类型3dx的A矩阵生成一个kernel：RCell3d8_A3dx_PCell3d64
            MPI_Abort(MPI_COMM_WORLD, -411);
        }
    }
    else if (info.type == SEMI_XY) {
        if (rstr.type == Cell_2d4 || rstr.type == Cell_2d16) {
            assert(prlg.type == Cell_2d4 || prlg.type == Cell_2d16);
            if (fine_mat.num_diag == 7 && (prlg.type == Cell_2d4 && rstr.type == Cell_2d4)) {
                // 下一层粗网格仍然为3d7的形式
                coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm,  7,
                    coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
                coar_mat_local = coar_mat->local_matrix;
                CHECK_HALO(*fine_mat_local, *coar_mat_local);
                const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
                if (my_pid == 0) printf("  using \033[1;35mRCell2d4_A3d7_PCell2d4\033[0m...\n");
                RCell2d4_A3d7_PCell2d4(
                    fine_mat_local->data, f_lx + hx * 2, f_ly + hy * 2, f_lz + hz * 2,
                    coar_mat_local->data, c_lx + hx * 2, c_ly + hy * 2, c_lz + hz * 2,
                    hx, hx + c_lx,    hy, hy + c_ly,    hz, hz + c_lz,
                    x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                    hx, hy, hz
                );
            }
            else {// 下一层会膨胀至3d27
                coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 27,
                    coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
                coar_mat_local = coar_mat->local_matrix;
                CHECK_HALO(*fine_mat_local, *coar_mat_local);
                const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
                // 需要为每一种特定类型3dx的A矩阵生成一个kernel：RCell2d4_A3dx_PCell2d16
                if (fine_mat_local->num_diag == 19) {
                    if (my_pid == 0) printf("  using \033[1;35mRCell2d4_A3d19_PCell2d16\033[0m...\n");
                    RCell2d4_A3d19_PCell2d16(
                        fine_mat_local->data, f_lx + hx*2, f_ly + hy*2, f_lz + hz*2,
                        coar_mat_local->data, c_lx + hx*2, c_ly + hy*2, c_lz + hz*2,
                        hx, hx + c_lx,   hy, hy + c_ly,   hz, hz + c_lz,
                        x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                        hx, hy, hz
                    );
                }
                else if (fine_mat_local->num_diag == 27) {
                    if (my_pid == 0) printf("  using \033[1;35mRCell2d4_A3d27_PCell2d16\033[0m...\n");
                    RCell2d4_A3d27_PCell2d16(
                        fine_mat_local->data, f_lx + hx*2, f_ly + hy*2, f_lz + hz*2,
                        coar_mat_local->data, c_lx + hx*2, c_ly + hy*2, c_lz + hz*2,
                        hx, hx + c_lx,   hy, hy + c_ly,   hz, hz + c_lz,
                        x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                        hx, hy, hz
                    );
                }
                else MPI_Abort(MPI_COMM_WORLD, -412);
            }
        }
        else if (rstr.type == Vtx_2d9 || rstr.type == Vtx_2d9_OpDep) {
            assert(prlg.type == rstr.type);
            coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 27,
                    coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
            coar_mat_local = coar_mat->local_matrix;
            CHECK_HALO(*fine_mat_local, *coar_mat_local);
            const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
            if (rstr.type == Vtx_2d9) {
                if (fine_mat_local->num_diag == 19) {
                    if (my_pid == 0) printf("  using \033[1;35mRVtx2d9_A3d19_PVtx2d9\033[0m...\n");
                    RVtx2d9_A3d19_PVtx2d9(
                        fine_mat_local->data, f_lx+2*hx, f_ly+2*hy, f_lz+2*hz,
                        coar_mat_local->data, c_lx+2*hx, c_ly+2*hy, c_lz+2*hz,
                        hx, hx + c_lx,   hy, hy + c_ly,   hz, hz + c_lz,
                        x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                        hx, hy, hz, info.fine_base_idx[0], info.fine_base_idx[1], info.fine_base_idx[2]
                    );
                }
                else if (fine_mat_local->num_diag == 27) {
                    if (my_pid == 0) printf("  using \033[1;35mRVtx2d9_A3d27_PVtx2d9\033[0m...\n");
                    RVtx2d9_A3d27_PVtx2d9(
                        fine_mat_local->data, f_lx+2*hx, f_ly+2*hy, f_lz+2*hz,
                        coar_mat_local->data, c_lx+2*hx, c_ly+2*hy, c_lz+2*hz,
                        hx, hx + c_lx,   hy, hy + c_ly,   hz, hz + c_lz,
                        x_lbdr, x_ubdr, y_lbdr, y_ubdr, z_lbdr, z_ubdr,
                        hx, hy, hz, info.fine_base_idx[0], info.fine_base_idx[1], info.fine_base_idx[2]
                    );
                }
                else MPI_Abort(MPI_COMM_WORLD, -808);
            }
            else {// Vtx_2d9_OpDep
                CHECK_LOCAL_HALO(*(rstr.Rmat->local_matrix), *coar_mat_local);
                CHECK_LOCAL_HALO(*(prlg.Pmat->local_matrix), *coar_mat_local);
                MPI_Abort(MPI_COMM_WORLD, -808);
            }
        }
        else {
            MPI_Abort(MPI_COMM_WORLD, -405);
        }
    }
    else if (info.type == DECP_XZ) {
        assert(coar_gy == fine_mat.input_dim[1]);
        seq_structMatrix<idx_t, data_t, data_t> * extmat_2d9 = nullptr;
        assert(fine_mat.num_diag >= 9);
        if (fine_mat.num_diag > 9) {// 需要extract（同时解耦）出XZ面上的矩阵系数
            extmat_2d9 = new seq_structMatrix<idx_t, data_t, data_t>(9, f_lx, f_ly, f_lz, hx, hy, hz);
            const idx_t tot_elems = (f_lx + hx*2)* (f_ly + hy*2) * (f_lz + hz*2);
            std::vector<idx_t> extract_ids;
            if      (fine_mat_local->num_diag == 19) extract_ids = {5, 6, 7, 8, 9, 10, 11, 12, 13};
            else if (fine_mat_local->num_diag == 27) extract_ids = {9, 10, 11, 12, 13, 14, 15, 16, 17};
            else assert(false);
            
            #pragma omp parallel for schedule(static)
            for (idx_t ie = 0; ie < tot_elems; ie++) {
                const data_t * src_ptr = fine_mat_local->data + ie * fine_mat_local->num_diag;
                data_t * dst_ptr = extmat_2d9->data + ie * 9;
                for (idx_t d = 0; d < 9; d++)
                    dst_ptr[d] = src_ptr[extract_ids[d]];
            }
            // extract之后将指针修改位置
            fine_mat_local = extmat_2d9;
        }// 否则 == 9则不需要

        if (rstr.type == Cell_2d4 || rstr.type == Cell_2d16) {
            assert(prlg.type == Cell_2d4 || prlg.type == Cell_2d16);
            if (fine_mat.num_diag == 5 && (prlg.type == Cell_2d4 && rstr.type == Cell_2d4)) {
                MPI_Abort(MPI_COMM_WORLD, -408);
            }
            else {// 下一层会膨胀至2d9
                coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 9,
                    coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
                coar_mat_local = coar_mat->local_matrix;
                CHECK_HALO(*fine_mat_local, *coar_mat_local);
                const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
                assert(rstr.type == Cell_2d4);
                for (idx_t j = hy; j < hy + c_ly; j++) {
                    const data_t * AF = fine_mat_local->data + j * fine_mat_local->slice_dki_size;
                          data_t * AC = coar_mat_local->data + j * coar_mat_local->slice_dki_size;
                    if (prlg.type == Cell_2d4) {
                        RCell2d4_A2d9_PCell2d4(
                            AF, f_lx+2*hx, f_lz+2*hz,    AC, c_lx+2*hx, c_lz+2*hz,
                            hx, hx+c_lx, hz, hz+c_lz, x_lbdr, x_ubdr, z_lbdr, z_ubdr,
                            hx, hy
                        );
                    } else {// Cell_2d16
                        RCell2d4_A2d9_PCell2d16(
                            AF, f_lx+2*hx, f_lz+2*hz,    AC, c_lx+2*hx, c_lz+2*hz,
                            hx, hx+c_lx, hz, hz+c_lz, x_lbdr, x_ubdr, z_lbdr, z_ubdr,
                            hx, hy
                        );
                    }
                }
            }
        }
        else if (rstr.type == Vtx_2d9 || rstr.type == Vtx_2d9_OpDep) {
            assert(prlg.type == rstr.type);// vertex类型应该对称
            coar_mat = new par_structMatrix<idx_t, data_t, data_t>(fine_mat.comm_pkg->cart_comm, 9,
                coar_gx, coar_gy, coar_gz, num_procs[1], num_procs[0], num_procs[2]);
            coar_mat_local = coar_mat->local_matrix;
            CHECK_HALO(*fine_mat_local, *coar_mat_local);
            const idx_t c_lx = coar_mat_local->local_x, c_ly = coar_mat_local->local_y, c_lz = coar_mat_local->local_z;
            // Call kernel
            if (rstr.type == Vtx_2d9) {
                for (idx_t j = hy; j < hy + c_ly; j++) {
                    const data_t * AF = fine_mat_local->data + j * fine_mat_local->slice_dki_size;
                          data_t * AC = coar_mat_local->data + j * coar_mat_local->slice_dki_size;
                    RVtx2d9_A2d9_PVtx2d9(
                        AF, f_lx+2*hx, f_lz+2*hz,    AC, c_lx+2*hx, c_lz+2*hz,
                        hx, hx+c_lx, hz, hz+c_lz,    x_lbdr, x_ubdr, z_lbdr, z_ubdr,
                        hx, hz, info.fine_base_idx[0], info.fine_base_idx[2]
                    );
                }
            }
            else {// 
                CHECK_LOCAL_HALO(*(rstr.Rmat->local_matrix), *coar_mat_local);
                CHECK_LOCAL_HALO(*(prlg.Pmat->local_matrix), *coar_mat_local);
                for (idx_t j = hy; j < hy + c_ly; j++) {
                    const data_t * AF = fine_mat_local->data + j * fine_mat_local->slice_dki_size;
                    const idx_t coar_offset = j * coar_mat_local->slice_dki_size;
                          data_t* AC = coar_mat_local->data          + coar_offset;
                    const data_t* RC = rstr.Rmat->local_matrix->data + coar_offset,
                                * PC = prlg.Pmat->local_matrix->data + coar_offset;
                    RVtxOD2d9_A2d9_PVtxOD2d9(
                        AF, f_lx+2*hx, f_lz+2*hz,    AC, c_lx+2*hx, c_lz+2*hz, RC, PC,
                        hx, hx+c_lx, hz, hz+c_lz,    x_lbdr, x_ubdr, z_lbdr, z_ubdr,
                        hx, hz, info.fine_base_idx[0], info.fine_base_idx[2]
                    );
                }
            }
        }
        else {// 未知分支
            MPI_Abort(MPI_COMM_WORLD, -405);
        }
        if (extmat_2d9 != nullptr) { delete extmat_2d9; extmat_2d9 = nullptr;}// 释放临时内存
        fine_mat_local = fine_mat.local_matrix;// 修改恢复指针
    }
    else {
        MPI_Abort(MPI_COMM_WORLD, -404);
    }

    coar_mat->update_halo();
#ifndef CROSS_POLAR
    // Check if Dirichlet boundary condition met
    assert(coar_mat->check_Dirichlet());
#endif

    if (fine_mat.DiagGroups_separated) coar_mat->separate_Diags();
    return coar_mat;
}

#endif