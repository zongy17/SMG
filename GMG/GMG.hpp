#ifndef SMG_GMG_HPP
#define SMG_GMG_HPP

#include "GMG_types.hpp"
#include "coarsen.hpp"
#include "restrict.hpp"
#include "prolong.hpp"

#include "../include/json.hpp"
#include <fstream>
using json = nlohmann::json;

#include <vector>
#include <string>
// #include "../copy_with_trunc.hpp"

template<typename idx_t, typename data_t, typename calc_t>
class GeometricMultiGrid : public Solver<idx_t, data_t, calc_t> {
public:
    idx_t num_levs;
    std::vector<COARSEN_TYPE> coarsen_types;
    std::vector<COARSE_OP_TYPE> coarse_op_types;
    std::vector<RELAX_TYPE> relax_types;
    std::vector<RSTR_PRLG_TYPE> restrict_types;
    std::vector<RSTR_PRLG_TYPE> prolong_types;
    std::string mat_repo;
    
    idx_t num_grid_sweeps[2] = {1, 1};

    const Operator<idx_t, calc_t, calc_t> * oper = nullptr;// operator (often as matrix-A of the problem)

    MPI_Comm comm;
    bool own_A0;
    // 各层网格上的方程：Au=f
    bool scale_before_setup_smoothers = false;
    par_structMatrix<idx_t, data_t, calc_t>** A_array_low  = nullptr;// 各层网格的A矩阵
    par_structMatrix<idx_t, calc_t, calc_t>** A_array_high = nullptr;
    par_structVector<idx_t, calc_t>** U_array = nullptr;// 各层网格的解向量
    par_structVector<idx_t, calc_t>** F_array = nullptr;// 各层网格的右端向量
    par_structVector<idx_t, calc_t>** aux_arr = nullptr;
    // 各层上的平滑子，可以选用不同的
    Solver<idx_t, data_t, calc_t> ** smoother = nullptr;
    // 层间转移的算子，可以不同层间选用不同的
    Restrictor<idx_t, calc_t> ** R_ops = nullptr;
    Interpolator<idx_t, calc_t> ** P_ops = nullptr;

    // 需要记录细、粗网格点映射关系的数据
    // 映射两端的各是一个三维向量par_structVector<idx_t, oper_t>
    COAR_TO_FINE_INFO<idx_t> * coar_to_fine_maps = nullptr;

    GeometricMultiGrid(std::string config_file, std::string matrices_repo);

    ~GeometricMultiGrid();

    void SetOperator(const Operator<idx_t, calc_t, calc_t> & op) { 
        this->oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        Setup(*((par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper)));
    }
    void SetRelaxWeights(const data_t * wts, idx_t num);

    void truncate() {
        int my_pid; MPI_Comm_rank(comm, &my_pid);
        if (my_pid == 0) printf("Warning: GMG truncated\n");
        // 逐层光滑子进行截断，以及截断掉A_array[...]
        for (idx_t i = 0; i < num_levs; i++) {
            // if (relax_types[i] != GaussElim) {// 最粗层的直接LU求解不能截断！！！
                A_array_high[i]->truncate();
                smoother[i]->truncate();
            // }
        }
        // 限制和插值算子不用截断，因为本来多重网格中的向量都是正常精度
    }

    // 外部接口，决定了是否要用0初值优化
    void Mult(const par_structVector<idx_t, calc_t> & input, 
                    par_structVector<idx_t, calc_t> & output, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(input, output);
        this->zero_guess = false;// reset for safety concern
    }

protected:
    void Setup(const par_structMatrix<idx_t, calc_t, calc_t> & A_problem);
    void V_Cycle(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const ;
    void Mult(const par_structVector<idx_t, calc_t> & input, par_structVector<idx_t, calc_t> & output) const {
        V_Cycle(input, output);
    }
};

template<typename idx_t, typename data_t, typename calc_t>
GeometricMultiGrid<idx_t, data_t, calc_t>::GeometricMultiGrid(std::string config_file, std::string matrices_repo)
{
    mat_repo = matrices_repo;
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (my_pid == 0) {// 0号进程负责读取配置文件
        std::ifstream f(config_file);
        json data = json::parse(f);
        num_levs = data["num_levs"];
        printf("num_levs %d\n", num_levs);
        // 粗化类型：三维，二维，或者 解耦的二维
        for (idx_t i = 0; i < num_levs - 1; i++) {
            if      (data["Coarsen"][i] == "FULL_XYZ") coarsen_types.push_back(FULL_XYZ);
            else if (data["Coarsen"][i] == "SEMI_XY" ) coarsen_types.push_back(SEMI_XY);
            else if (data["Coarsen"][i] == "DECP_XZ" ) coarsen_types.push_back(DECP_XZ);
            else MPI_Abort(MPI_COMM_WORLD, -707);
        }
        // 限制算子
        for (idx_t i = 0; i < num_levs - 1; i++) {
            if      (data["restrict"][i] == "Vtx_2d9"      ) restrict_types.push_back(Vtx_2d9);
            else if (data["restrict"][i] == "Vtx_2d9_OpDep") restrict_types.push_back(Vtx_2d9_OpDep);
            else if (data["restrict"][i] == "Vtx_2d5"      ) restrict_types.push_back(Vtx_2d5);
            else if (data["restrict"][i] == "Vtx_inject"   ) restrict_types.push_back(Vtx_inject);
            else if (data["restrict"][i] == "Cell_2d4"     ) restrict_types.push_back(Cell_2d4);
            else if (data["restrict"][i] == "Cell_2d16"    ) restrict_types.push_back(Cell_2d16);
            else if (data["restrict"][i] == "Cell_3d8"     ) restrict_types.push_back(Cell_3d8);
            else if (data["restrict"][i] == "Cell_3d64"    ) restrict_types.push_back(Cell_3d64);
            else MPI_Abort(MPI_COMM_WORLD, -709);
        }
        // 插值算子
        for (idx_t i = 0; i < num_levs - 1; i++) {
            if      (data["interp"][i] == "Vtx_2d9"      ) prolong_types.push_back(Vtx_2d9);
            else if (data["interp"][i] == "Vtx_2d9_OpDep") prolong_types.push_back(Vtx_2d9_OpDep);
            else if (data["interp"][i] == "Vtx_2d5"      ) prolong_types.push_back(Vtx_2d5);
            else if (data["interp"][i] == "Vtx_inject"   ) prolong_types.push_back(Vtx_inject);
            else if (data["interp"][i] == "Cell_2d4"     ) prolong_types.push_back(Cell_2d4);
            else if (data["interp"][i] == "Cell_2d16"    ) prolong_types.push_back(Cell_2d16);
            else if (data["interp"][i] == "Cell_3d8"     ) prolong_types.push_back(Cell_3d8);
            else if (data["interp"][i] == "Cell_3d64"    ) prolong_types.push_back(Cell_3d64);
            else MPI_Abort(MPI_COMM_WORLD, -710);
        }
        // 粗网格构造方式：重离散，或者 Galerkin
        for (idx_t i = 0; i < num_levs - 1; i++) {
            if      (data["CoarOper"][i] == "rediscrete") coarse_op_types.push_back(DISCRETIZED);
            else if (data["CoarOper"][i] == "Galerkin"  ) coarse_op_types.push_back(GALERKIN);
            else MPI_Abort(MPI_COMM_WORLD, -708);
        }
        // 光滑子
        for (idx_t i = 0; i < num_levs; i++) {
            if      (data["smoother"][i] == "PGS"     ) relax_types.push_back(PGS);
            else if (data["smoother"][i] == "LGS"     ) relax_types.push_back(LGS);
            else if (data["smoother"][i] == "PILU"    ) relax_types.push_back(PILU);
            else if (data["smoother"][i] == "BILU3d7" ) relax_types.push_back(BILU3d7);
            else if (data["smoother"][i] == "BILU3d15") relax_types.push_back(BILU3d15);
            else if (data["smoother"][i] == "BILU3d19") relax_types.push_back(BILU3d19);
            else if (data["smoother"][i] == "BILU3d27") relax_types.push_back(BILU3d27);
            else if (data["smoother"][i] == "LU"      ) relax_types.push_back(GaussElim);
            else MPI_Abort(MPI_COMM_WORLD, -706);
        }
    }

    // 广播获取配置参数
    MPI_Bcast(&num_levs, 1, (sizeof(idx_t) == 4)? MPI_INT : MPI_LONG_INT, 0, MPI_COMM_WORLD);
    if (my_pid != 0) {
        relax_types.reserve(num_levs);
        coarsen_types.reserve(num_levs - 1);
        restrict_types.reserve(num_levs - 1);
        prolong_types.reserve(num_levs - 1);
        coarse_op_types.reserve(num_levs - 1);
    }
    MPI_Bcast(coarsen_types.data(), num_levs - 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    if (my_pid != 0) {
        printf("proc %d: ", my_pid);
        for (idx_t i = 0; i < num_levs - 1; i++) printf("%d ", relax_types[i]);
        printf("\n");
    }
#endif
    MPI_Bcast(restrict_types.data(), num_levs - 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    if (my_pid != 0) {
        printf("proc %d: ", my_pid);
        for (idx_t i = 0; i < num_levs - 1; i++) printf("%d ", restrict_types[i]);
        printf("\n");
    }
#endif
    MPI_Bcast(prolong_types.data(), num_levs - 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    if (my_pid != 0) {
        printf("proc %d: ", my_pid);
        for (idx_t i = 0; i < num_levs - 1; i++) printf("%d ", prolong_types[i]);
        printf("\n");
    }
#endif
    MPI_Bcast(coarse_op_types.data(), num_levs - 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    if (my_pid != 0) {
        printf("proc %d: ", my_pid);
        for (idx_t i = 0; i < num_levs - 1; i++) printf("%d ", coarse_op_types[i]);
        printf("\n");
    }
#endif
    MPI_Bcast(relax_types.data(), num_levs, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    if (my_pid != 0) {
        printf("proc %d: ", my_pid);
        for (idx_t i = 0; i < num_levs; i++) printf("%d ", relax_types[i]);
        printf("\n");
    }
#endif
    // 分配指针空间
    A_array_low = new par_structMatrix<idx_t, data_t, calc_t>* [num_levs];
    A_array_high= new par_structMatrix<idx_t, calc_t, calc_t>* [num_levs];
    U_array = new par_structVector<idx_t, calc_t>* [num_levs];
    F_array = new par_structVector<idx_t, calc_t>* [num_levs];
    aux_arr = new par_structVector<idx_t, calc_t>* [num_levs];
    smoother = new Solver<idx_t, data_t, calc_t>* [num_levs];
    R_ops   = new Restrictor<idx_t, calc_t>      * [num_levs - 1];
    P_ops   = new Interpolator<idx_t, calc_t>    * [num_levs - 1];
    coar_to_fine_maps = new COAR_TO_FINE_INFO<idx_t> [num_levs - 1];
    for (idx_t i = 0; i < num_levs; i++) {
        A_array_high[i] = nullptr;
        A_array_low[i] = nullptr;
        smoother[i]= nullptr;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
GeometricMultiGrid<idx_t, data_t, calc_t>::~GeometricMultiGrid() {
    for (idx_t i = 0; i < num_levs; i++) {
        if (i == 0 && own_A0) {
            delete A_array_high[i]; A_array_high[i] = nullptr;
            delete A_array_low [i]; A_array_low [i] = nullptr;
        }
        delete U_array[i]; U_array[i] = nullptr;
        delete F_array[i]; F_array[i] = nullptr;
        delete aux_arr[i]; aux_arr[i] = nullptr;
        delete smoother[i]; smoother[i] = nullptr;
    }
    delete[] A_array_high; A_array_high = nullptr;
    delete[] A_array_low ; A_array_low  = nullptr;
    delete[] U_array; U_array = nullptr;
    delete[] F_array; F_array = nullptr;
    delete[] aux_arr; aux_arr = nullptr;
    delete[] coar_to_fine_maps; coar_to_fine_maps = nullptr;
    for (idx_t i = 0; i < num_levs - 1; i++) {
        delete R_ops[i]; R_ops[i] = nullptr; 
        delete P_ops[i]; P_ops[i] = nullptr;
    }
    delete[] R_ops; R_ops = nullptr;
    delete[] P_ops; P_ops = nullptr;
}

template<typename idx_t, typename data_t, typename calc_t>
void GeometricMultiGrid<idx_t, data_t, calc_t>::Setup(const par_structMatrix<idx_t, calc_t, calc_t> & A_problem)
{
    // 根据外层问题的进程划分，来确定预条件多重网格的进程划分
    comm = A_problem.comm_pkg->cart_comm;
    MPI_Barrier(comm);
    double t = wall_time();

    int my_pid;
    MPI_Comm_rank(comm, &my_pid);
    int procs_dims = sizeof(A_problem.comm_pkg->cart_ids) / sizeof(int); assert(procs_dims == 3);
    int num_procs[3], periods[3], coords[3];
    MPI_Cart_get(comm, procs_dims, num_procs, periods, coords);

    if (my_pid == 0) printf("GMG\n");
    if (num_levs == 1) if (my_pid == 0) printf("  Warning: only 1 layer exists.\n");

    {// 先从外迭代的矩阵拷贝最密层的数据A_array[0]
        idx_t gx, gy, gz;
        gx = A_problem.local_matrix->local_x * num_procs[1];
        gy = A_problem.local_matrix->local_y * num_procs[0];
        gz = A_problem.local_matrix->local_z * num_procs[2];

        A_array_high[0] = new par_structMatrix<idx_t, calc_t, calc_t>(comm, A_problem.num_diag, gx, gy, gz, num_procs[1], num_procs[0], num_procs[2]);
        own_A0 = true;
        {// copy data
            const   seq_structMatrix<idx_t, calc_t, calc_t> & out_mat = *(A_problem.local_matrix);
                    seq_structMatrix<idx_t, calc_t, calc_t> & gmg_mat = *(A_array_high[0]->local_matrix);
            CHECK_LOCAL_HALO(out_mat, gmg_mat);
            idx_t tot_len = (out_mat.halo_y * 2 + out_mat.local_y)
                        *   (out_mat.halo_x * 2 + out_mat.local_x)
                        *   (out_mat.halo_z * 2 + out_mat.local_z) *  out_mat.num_diag;
            #pragma omp parallel for schedule(static)
            for (idx_t i = 0; i < tot_len; i++)
                gmg_mat.data[i] = out_mat.data[i];
        }
        U_array[0] = new par_structVector<idx_t, calc_t>(comm, gx, gy, gz, num_procs[1], num_procs[0], num_procs[2], A_problem.num_diag != 7);
        F_array[0] = new par_structVector<idx_t, calc_t>(*U_array[0]);
        aux_arr[0] = new par_structVector<idx_t, calc_t>(*F_array[0]);

        if (my_pid == 0) {
            printf("  lev #%d : global %4d x %4d x %4d local %3d x %3d x %3d\n", 0, 
                U_array[0]->global_size_x        , U_array[0]->global_size_y        , U_array[0]->global_size_z ,
                U_array[0]->local_vector->local_x, U_array[0]->local_vector->local_y, U_array[0]->local_vector->local_z);
        }
    }
    // 再构建各层粗网格
    for (idx_t ilev = 1; ilev < num_levs; ilev++) {
        // printf("proc %d constructing %d-th lev\n", my_pid, ilev);
        idx_t gx, gy, gz;
        gx = A_array_high[ilev - 1]->local_matrix->local_x * num_procs[1]; // global_size_x
        gy = A_array_high[ilev - 1]->local_matrix->local_y * num_procs[0]; // global_size_y
        gz = A_array_high[ilev - 1]->local_matrix->local_z * num_procs[2]; // global_size_z

#ifdef CROSS_POLAR
        const bool periodic[3] = {true , true , false};// 虽然y方向不是严格的周期，但为计算粗化后的格点数，当成周期的
#else
        const bool periodic[3] = {false, false, false};// {x, y, z}三个方向的周期性
#endif

        const COARSEN_TYPE & shrk_type = coarsen_types[ilev - 1];// grid-shrink type 网格缩小的类型，三维或二维
        if (shrk_type == FULL_XYZ) {// 三维全粗化
            XYZ_standard_coarsen(*U_array[ilev - 1], periodic, coar_to_fine_maps[ilev - 1]);
        }
        else if (shrk_type == SEMI_XY) {// 二维半粗化，只粗化XY面
            XY_semi_coarsen     (*U_array[ilev - 1], periodic, coar_to_fine_maps[ilev - 1]);
        }
        else if (shrk_type == DECP_XZ) {// 解耦后做二维粗化，只粗化XZ面
            XZ_decp_coarsen     (*U_array[ilev - 1], periodic, coar_to_fine_maps[ilev - 1], 2, 0, gz%2);
        }
        else {
            if (my_pid == 0) printf("Invalid coarsen types of %d\n", shrk_type);
            MPI_Abort(comm, -1000);
        }
        if (my_pid == 0) {
            printf("coarsen base idx: x %d y %d z %d\n", coar_to_fine_maps[ilev - 1].fine_base_idx[0],
                coar_to_fine_maps[ilev - 1].fine_base_idx[1], coar_to_fine_maps[ilev - 1].fine_base_idx[2]);
        }

        // 做一些相容性检查
        idx_t new_gx, new_gy, new_gz;
        const RSTR_PRLG_TYPE& rstr_type = restrict_types[ilev - 1], & prlg_type = prolong_types [ilev - 1];
        const idx_t* strides = coar_to_fine_maps[ilev - 1].stride;
        if (rstr_type == Vtx_2d9_OpDep || rstr_type == Vtx_2d9 || rstr_type == Vtx_2d5 || rstr_type == Vtx_inject) {
            assert(shrk_type != FULL_XYZ);
            // vertex-center类型的粗化应该避开边界，细层某维若为偶数，则必须要为周期性
            if (shrk_type == SEMI_XY) {
                assert((gx % strides[0] == 1) || periodic[0]); assert((gy % strides[1] == 1) || periodic[1]); assert(strides[2] == 1);
            }
            else if (shrk_type == DECP_XZ) {
                assert((gx % strides[0] == 1) || periodic[0]); assert(strides[1] == 1); assert((gz % strides[2] == 1) || periodic[2]); 
            }
            assert(prlg_type == rstr_type);// vertex类型的限制和插值应该是对称的
        }
        else if (rstr_type == Cell_2d4 || rstr_type == Cell_2d16) {
            assert(prlg_type == Cell_2d4 || prlg_type == Cell_2d16);// cell类型的插值和限制可不对称，但应该同为cell类型
            assert(shrk_type != FULL_XYZ);
            // cell-center类型的粗化要求细网格各维均为偶数
            if (shrk_type == SEMI_XY) {
                assert(gx % strides[0] == 0); assert(gy % strides[1] == 0); assert(strides[2] == 1);
            }
            else if (shrk_type == DECP_XZ) {
                assert(gx % strides[0] == 0); assert(strides[1] == 1); assert(gz % strides[2] == 0);
            }
        }
        else if (rstr_type == Cell_3d8 || rstr_type == Cell_3d64) {
            assert(prlg_type == Cell_3d8 || prlg_type == Cell_3d64);
            assert(shrk_type == FULL_XYZ);
            assert(gx % strides[0] == 0); assert(gy % strides[1] == 0); assert(gz % strides[2] == 0);
        }

        // 根据粗化步长和偏移计算出下一（粗）层的大小
        new_gx = gx / strides[0];
        new_gy = gy / strides[1];
        new_gz = gz / strides[2];
        U_array     [ilev] = new par_structVector<idx_t,         calc_t>(comm,     new_gx, new_gy, new_gz,
            num_procs[1], num_procs[0], num_procs[2], true);

        // 建立限制和插值算子
        R_ops[ilev - 1] = new Restrictor<idx_t, calc_t>(rstr_type, coarse_op_types[ilev - 1] == DISCRETIZED);// 注意Galerkin方法所用的限制算子
        switch (rstr_type)
        {
        case Vtx_inject : if (my_pid == 0) printf("  using Vtx_inject restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Vtx_2d5    : if (my_pid == 0) printf("  using Vtx_2d5 restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Vtx_2d9    : if (my_pid == 0) printf("  using Vtx_2d9 restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Vtx_2d9_OpDep: if (my_pid == 0) printf("  using Vtx_2d9_OpDep restriction of %d-th to %d-th lev\n", ilev-1, ilev);
            // 需要传递细网格A矩阵来计算限制矩阵的值
            R_ops[ilev - 1]->setup_operator(*A_array_high[ilev - 1], *U_array[ilev-1], *U_array[ilev], coar_to_fine_maps[ilev-1]);
            break;
        case Cell_2d4   : if (my_pid == 0) printf("  using Cell_2d4   restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Cell_2d16  : if (my_pid == 0) printf("  using Cell_2d16  restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Cell_3d8   : if (my_pid == 0) printf("  using Cell_3d8   restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Cell_3d64  : if (my_pid == 0) printf("  using Cell_3d64  restriction of %d-th to %d-th lev\n", ilev-1, ilev); break;
        default: if (my_pid == 0) printf("Error while setting restrictor: INVALID restrict type of %d\n", rstr_type); MPI_Abort(MPI_COMM_WORLD, -201);
        }
        
        P_ops[ilev - 1] = new Interpolator<idx_t, calc_t>(prlg_type);
        switch (prlg_type)
        {
        case Vtx_inject : if (my_pid == 0) printf("  using Vtx_inject interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Vtx_2d5    : if (my_pid == 0) printf("  using Vtx_2d5 interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Vtx_2d9    : if (my_pid == 0) printf("  using Vtx_2d9 interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Vtx_2d9_OpDep: if (my_pid == 0) printf("  using Vtx_2d9_OpDep interpolation of %d-th to %d-th lev\n", ilev-1, ilev);
            assert(rstr_type == Vtx_2d9_OpDep);
            P_ops[ilev - 1]->setup_operator(*(R_ops[ilev - 1]->Rmat));
            break;
        case Cell_2d4   : if (my_pid == 0) printf("  using Cell_2d4  interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Cell_2d16  : if (my_pid == 0) printf("  using Cell_2d16 interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Cell_3d8   : if (my_pid == 0) printf("  using Cell_3d8  interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        case Cell_3d64  : if (my_pid == 0) printf("  using Cell_3d64 interpolation of %d-th to %d-th lev\n", ilev-1, ilev); break;
        default: if (my_pid == 0) printf("Error while setting interpolator: INVALID prolong type of %d\n", prlg_type); MPI_Abort(MPI_COMM_WORLD, -202);
        }

        // 构造粗网格矩阵
        if (coarse_op_types[ilev - 1] == DISCRETIZED) {
            assert(shrk_type == SEMI_XY);
            A_array_high[ilev] = new par_structMatrix<idx_t, calc_t, calc_t>(comm, A_problem.num_diag, 
                new_gx, new_gy, new_gz, num_procs[1], num_procs[0], num_procs[2]);
            A_array_high[ilev]->read_data(mat_repo + "/" + std::to_string(new_gx) + "x" + std::to_string(new_gy) + "x" + std::to_string(new_gz));
        }
        else if (coarse_op_types[ilev - 1] == GALERKIN) {
            A_array_high[ilev] = Galerkin_RAP(*R_ops[ilev - 1], *A_array_high[ilev - 1], *P_ops[ilev - 1], 
                coar_to_fine_maps[ilev - 1]);
            assert(A_array_high[ilev]->input_dim[0] == new_gx
                && A_array_high[ilev]->input_dim[1] == new_gy && A_array_high[ilev]->input_dim[2] == new_gz );
        }
        
#ifdef DEBUG // 将粗层网格打印出来观察一下
        A_array_high[ilev]->write_data(".", std::to_string(ilev));
#endif

        F_array[ilev] = new par_structVector<idx_t, calc_t>(*U_array[ilev]);
        aux_arr[ilev] = new par_structVector<idx_t, calc_t>(*U_array[ilev]);

        if (my_pid == 0) {
            printf("  lev #%d : global %4d x %4d x %4d local %3d x %3d x %3d\n", ilev, 
                U_array[ilev]->global_size_x        , U_array[ilev]->global_size_y        , U_array[ilev]->global_size_z ,
                U_array[ilev]->local_vector->local_x, U_array[ilev]->local_vector->local_y, U_array[ilev]->local_vector->local_z);
        }
    }// ilev = [1, num_levs)

    if (scale_before_setup_smoothers) {
        if (my_pid == 0) printf("\033[1;35mScale after RAP before setup smoothers\033[0m\n");
        // 各层矩阵做scaling
        for (idx_t i = 0; i < num_levs; i++) {
            if (relax_types[i] != GaussElim)
                A_array_high[i]->scale(10.0);
            else
                // A_array_high[i]->scale(0.1);
                A_array_high[i]->scale(1.0);
        }
    }

#ifdef COMPRESS
    if (my_pid == 0) printf("Warning: GMG compress data of A along x\n");
    for (idx_t ilev = 0; ilev < num_levs; ilev++) {
        A_array_high[ilev]->compress();// try to average along x-direction
    }
#endif
    // 如有必要，截断A矩阵
    if constexpr (sizeof(data_t) != sizeof(calc_t)) {
        assert(sizeof(data_t) < sizeof(calc_t));
        if (my_pid == 0) printf("Warning: GMG data_t %ld bytes, calc_t %ld bytes\n", sizeof(data_t), sizeof(calc_t));
        for (idx_t i = 0; i < num_levs; i++) {
            // if (relax_types[i] != PILU) {// what if BILU????
                idx_t gx, gy, gz;
                gx = A_array_high[i]->local_matrix->local_x * num_procs[1]; // global_size_x
                gy = A_array_high[i]->local_matrix->local_y * num_procs[0]; // global_size_y
                gz = A_array_high[i]->local_matrix->local_z * num_procs[2];
                A_array_low[i] = new par_structMatrix<idx_t, data_t, calc_t>(comm, A_array_high[i]->num_diag, gx, gy, gz, num_procs[1], num_procs[0], num_procs[2]);
                const   seq_structMatrix<idx_t, calc_t, calc_t> & src_h = *(A_array_high[i]->local_matrix);
                // 当SpMV需要转换精度时，换成SOA来
                A_array_low[i]->separate_truncate_Diags(src_h);
                if (A_array_high[i]->scaled) {// 拷贝度量矩阵
                    A_array_low[i]->scaled = true;
                    A_array_low[i]->sqrt_D = new seq_structVector<idx_t, calc_t>(*(A_array_high[i]->sqrt_D));
                    const idx_t tot_len = (A_array_high[i]->sqrt_D->local_x + A_array_high[i]->sqrt_D->halo_x * 2)
                                        * (A_array_high[i]->sqrt_D->local_y + A_array_high[i]->sqrt_D->halo_y * 2)
                                        * (A_array_high[i]->sqrt_D->local_z + A_array_high[i]->sqrt_D->halo_z * 2);
                    const calc_t * src_data = A_array_high[i]->sqrt_D->data;
                    calc_t * dst_data = A_array_low[i]->sqrt_D->data;
                    #pragma omp parallel for schedule(static)
                    for (idx_t p = 0; p < tot_len; p++)
                        dst_data[p] = src_data[p];
                }
            // }
        }
    }
    
    // 建立各层的平滑子
        // 设置完平滑子之后就可以释放不需要的A了
        for (idx_t i = 0; i < num_levs; i++) {
            if (relax_types[i] == PGS) {
                if (my_pid == 0) printf("  using \033[1;35mpointwise-GS\033[0m as smoother of %d-th lev\n", i);
                smoother[i] = new PointGS    <idx_t, data_t, calc_t>(SYMMETRIC);
                smoother[i]->SetOperator(*A_array_high[i]);
                // delete A_array_high[i]; A_array_high[i] = nullptr;
            }
            else if (relax_types[i] == LGS) {
                if (my_pid == 0) printf("  using \033[1;35mlinewise-GS\033[0m as smoother of %d-th lev\n", i);
                smoother[i] = new LineGS     <idx_t, data_t, calc_t>(SYMMETRIC, VERT, U_array[i]->comm_pkg);
                smoother[i]->SetOperator(*A_array_high[i]);// 先走一遍流程，将生成的东西的空间分配出来
                // delete A_array_high[i]; A_array_high[i] = nullptr;
            }
            else if (relax_types[i] == PILU) {
                if (my_pid == 0) printf("  using \033[1;35mplanewise-ILU\033[0m as smoother of %d-th lev\n", i);
                smoother[i] = new PlaneILU   <idx_t, data_t, calc_t>;
                smoother[i]->SetOperator(*A_array_high[i]);// 先走一遍流程，将生成的东西的空间分配出来
                // if (sizeof(data_t) != sizeof(oper_t)) {
                //     PlaneILU<idx_t, oper_t, calc_t> prec_high;
                //     prec_high.SetOperator(*A_array_high[i]);
                //     copy_w_trunc_PILU(prec_high, *((PlaneILU<idx_t, data_t, oper_t, res_t>*)(smoother[i])));
                // }
                // delete A_array_low[i]; A_array_low[i] = nullptr;
            }
            else if (relax_types[i] == BILU3d7 || relax_types[i] == BILU3d15 ||
                    relax_types[i] == BILU3d19 || relax_types[i] == BILU3d27) {
                BlockILU_type type_3d = ILU_3D27;
                if      (relax_types[i] == BILU3d7 ) type_3d = ILU_3D7;
                else if (relax_types[i] == BILU3d15) type_3d = ILU_3D15;
                else if (relax_types[i] == BILU3d19) type_3d = ILU_3D19;
                else if (relax_types[i] == BILU3d27) type_3d = ILU_3D27;
                if (my_pid == 0) printf("  using \033[1;35mblockwise-ILU type %d\033[0m as smoother of %d-th lev\n", type_3d, i);
                smoother[i] = new BlockILU<idx_t, data_t, calc_t>(type_3d);
                smoother[i]->SetOperator(*A_array_high[i]);
                // if (sizeof(data_t) != sizeof(oper_t)) {
                //     BlockILU<idx_t, oper_t, oper_t, res_t> prec_high(type_3d);
                //     prec_high.SetOperator(*A_array_high[i]);
                //     copy_w_trunc_BILU(prec_high, *((BlockILU<idx_t, data_t, oper_t, res_t>*)(smoother[i])));
                // }
            }
            else if (relax_types[i] == GaussElim) {
                DenseLU_type type;
                if      (A_array_high[i]->num_diag ==  7) type = DenseLU_3D7;
                else if (A_array_high[i]->num_diag == 27) type = DenseLU_3D7;
                else assert(false);
                if (my_pid == 0) printf("  using \033[1;35mdense-LU type %d\033[0m as smoother of %d-th lev\n", type, i);
                smoother[i] = new DenseLU<idx_t, data_t, calc_t>(type);
                t += wall_time();// LU分解的时间单独算
                smoother[i]->SetOperator(*A_array_high[i]);
                t += ((DenseLU<idx_t, data_t, calc_t>*)smoother[i])->setup_time;
                t -= wall_time();
            }
            else {
                if (my_pid == 0) printf("Error while setting smoother: INVALID relax type of %d\n", relax_types[i]);
                MPI_Abort(MPI_COMM_WORLD, -200);
            }
        }
    
    // 稳妥起见，初始化，避免之后出nan，同时也是为了Dirichlet边界
    for (idx_t ilev = 0; ilev < num_levs; ilev++) {
        U_array[ilev]->set_val(0.0, true);
        F_array[ilev]->set_val(0.0, true);
        aux_arr[ilev]->set_val(0.0, true);
    }
    
    MPI_Barrier(comm);
    t = wall_time() - t;
    if (my_pid == 0) printf("Setup costs %.6f s\n", t);

#ifndef NDEBUG // 检验限制和插值正确性
    // 检验限制算子的正确性
    U_array[0]->set_val(1.5);
    double res = vec_dot<idx_t, calc_t, double>(*(U_array[0]), *(U_array[0]));
    if (my_pid == 0) printf("(U0, U0): %.12e\n", res);

    for (idx_t i = 0; i < num_levs - 1; i++) {
        R_ops[i]->apply(*U_array[i], *U_array[i+1], coar_to_fine_maps[i]);
        res = vec_dot<idx_t, calc_t, double>(*U_array[i+1], *U_array[i+1]);
        if (my_pid == 0) printf("(U%d, U%d): %.12e\n", i+1, i+1, res);
    }
    // 检验插值算子的正确性
    // U_array[1]->set_val(1.25);

    for (idx_t i = num_levs - 1; i >= 1; i--) {
        U_array[i-1]->set_val(0.0);// 细网格层向量先清空
        P_ops[i-1]->apply(*U_array[i], *U_array[i-1], coar_to_fine_maps[i-1]);
        res = vec_dot<idx_t, calc_t, double>(*U_array[i-1], *U_array[i-1]);
        if (my_pid == 0) printf("(U%d, U%d): %.12e\n", i-1, i-1, res);
    }
#endif
}

template<typename idx_t, typename data_t, typename calc_t>
void GeometricMultiGrid<idx_t, data_t, calc_t>::V_Cycle(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    assert(this->oper != nullptr);
    CHECK_OUTPUT_DIM(*this, b);
    CHECK_INPUT_DIM(*this, x);

    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 最细层的数据准备：残差即为右端向量b
    vec_copy(b, *F_array[0]);
    if (!this->zero_guess) {
        vec_copy(x, *U_array[0]);
    }// 否则，由第0层的光滑子负责将第0层初始解设为0
    // 但注意如果没有将初始解设为0，同时在光滑子里面又走了zero_guess==false的分支，则会误用数组里面的“伪值”计算而致错误

    idx_t i = 0;// ilev

    // 除了最粗层，都能从细往粗走
    for ( ; i < num_levs - 1; i++) {
        // double ck_dot = vec_dot<idx_t, calc_t, double>(*F_array[i], *F_array[i]);
        // if (my_pid == 0) printf("before smth%d (f,f) = %.10e\n", i, ck_dot);

        for (idx_t j = 0; j < num_grid_sweeps[0]; j++)
            // 当前层对残差做平滑（对应前平滑），平滑后结果在U_array中：相当于在当前层解一次Mu=f
            smoother[i]->Mult(*F_array[i], *U_array[i], this->zero_guess && j == 0);
        
        // ck_dot = vec_dot<idx_t, calc_t, double>(*U_array[i], *U_array[i]);
        // if (my_pid == 0) printf("after  smth%d (u,u) = %.10e\n", i, ck_dot);
        
        // 计算在当前层平滑后的残差
        if constexpr (sizeof(data_t) == sizeof(calc_t))
            A_array_high[i]->Mult(*U_array[i], *aux_arr[i], false);
        else
            A_array_low [i]->Mult(*U_array[i], *aux_arr[i], false);

        vec_add(*F_array[i], -1.0, *aux_arr[i], *aux_arr[i]);// 此时残差存放在aux_arr中

        // 将当前层残差限制到下一层去
        R_ops[i]->apply(*aux_arr[i], *F_array[i+1], coar_to_fine_maps[i]);

        // 由下一层的光滑子负责将下一层初始解设为0
    }

    // double ck_dot = vec_dot<idx_t, calc_t, double>(*F_array[i], *F_array[i]);
    // if (my_pid == 0) printf("before smth%d (f,f) = %.10e\n", i, ck_dot);

    assert(i == num_levs - 1);// 最粗层做前平滑
    if (relax_types[i] == GaussElim) {// 直接法只做一次
            if (A_array_high[i]->scaled) {
                seq_vec_elemwise_div(*(F_array[i]->local_vector), *(A_array_high[i]->sqrt_D));// 计算Fbar = D^{-1/2}*F
                smoother[i]->Mult(*F_array[i], *U_array[i], this->zero_guess);// 计算 Ubar = Abar^{-1}*Fbar
                seq_vec_elemwise_div(*(U_array[i]->local_vector), *(A_array_high[i]->sqrt_D));// 计算U = D^{1/2}*Ubar
            } else {
                smoother[i]->Mult(*F_array[i], *U_array[i], this->zero_guess);
            }
    } else {
        for (idx_t j = 0; j < num_grid_sweeps[0]; j++)
            smoother[i]->Mult(*F_array[i], *U_array[i], this->zero_guess && j == 0);
    }

    // ck_dot = vec_dot<idx_t, calc_t, double>(*U_array[i], *U_array[i]);
    // if (my_pid == 0) printf("after  smth%d (u,u) = %.10e\n", i, ck_dot);

    // 除了最细层，都能从粗往细走
    for ( ; i >= 1; i--) {
        if (relax_types[i] != GaussElim) {// 如果是直接法，不必再解一次了
            // 当前层对残差做平滑（对应后平滑），平滑后结果在U_array中：相当于在当前层解一次Au=f
            for (idx_t j = 0; j < num_grid_sweeps[1]; j++)
                smoother[i]->Mult(*F_array[i], *U_array[i], false);
        } else {
            assert(i == num_levs - 1);// 一般直接法放在最粗层
        }

        // 不用计算当前层平滑后的残差，直接插值回到更细一层，并更新细层的解
        P_ops[i-1]->apply(*U_array[i], *aux_arr[i-1], coar_to_fine_maps[i-1]);

        vec_add(*U_array[i-1], 1.0, *aux_arr[i-1], *U_array[i-1]);
    }
    
    assert(i == 0);
    // 最细层做后平滑
    for (idx_t j = 0; j < num_grid_sweeps[1]; j++)
        smoother[i]->Mult(*F_array[i], *U_array[i], false);
    // 最后将结果拷出来
    vec_copy(*U_array[0], x);
}

template<typename idx_t, typename data_t, typename calc_t>
void GeometricMultiGrid<idx_t, data_t, calc_t>::SetRelaxWeights(const data_t * wts, idx_t num)
{
    assert(this->oper != nullptr);
    assert(num <= num_levs);
    int my_pid;
    MPI_Comm_rank(comm, &my_pid);

    if (my_pid == 0) printf("Set weights for smoothers as: ");
    for (idx_t i = 0; i < num; i++) {
        smoother[i]->SetRelaxWeight(wts[i]);
        if (my_pid == 0) printf("%.3f  ", wts[i]);
    }
    if (my_pid == 0) printf("\n");
}

#endif