#ifndef SMG_POINT_GS_HPP
#define SMG_POINT_GS_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"

template<typename idx_t, typename data_t>
struct IrrgPts_Effect
{
    idx_t loc_id;// local idx to access irrgPts_vec array
    idx_t i, j, k;// 该非规则点所影响的结构点的三维全局坐标
    data_t val;// 影响的值
    IrrgPts_Effect() {}
    IrrgPts_Effect(idx_t ir, idx_t i, idx_t j, idx_t k, data_t v): loc_id(ir), i(i), j(j), k(k), val(v) {}
    bool operator < (const IrrgPts_Effect & b) const {
        if (j < b.j) return true;
        else if (j > b.j) return false;
        else {// j == b.j
            if (i < b.i) return true;
            else if (i > b.i) return false;
            else {// i == b.i
                assert(k != b.k);
                return k < b.k;
            }
        }
    }
};

// data_t是数据存储的精度，calc_t是算子作用时的精度（对应向量的精度）
template<typename idx_t, typename data_t, typename calc_t>
class PointGS : public Solver<idx_t, data_t, calc_t> {
public:
    // 对称GS：0 for sym, 1 for forward, -1 backward
    SCAN_TYPE scan_type = SYMMETRIC;
    mutable bool last_time_forward = false;

    // operator (often as matrix-A)
    const Operator<idx_t, calc_t, calc_t> * oper = nullptr;

    // separate diagonal values if for efficiency concern is needed
    // should only be used when separation is cheap
    bool LU_separated = false;
    seq_structMatrix<idx_t, data_t, calc_t> * L = nullptr;
    seq_structMatrix<idx_t, data_t, calc_t> * U = nullptr;
    void separate_LU();

    bool DiagGroups_separated = false;
    idx_t DiagGroups_cnt = 0;
    seq_structMatrix<idx_t, data_t, calc_t> ** DiagGroups = nullptr;
    // AOS => SOA
    void separate_Diags();

    // 将A矩阵内的非规则点对结构点的影响提取出来，并按照自然序排序，以便在做结构点
    void prepare_irrgPts();
    idx_t num_irrgPts_effect = 0;
    IrrgPts_Effect<idx_t, data_t> * irrg_to_Struct = nullptr;// 局部非规则点的一维序号，对应结构点的三维坐标（已带偏移），本结构点受非结构点的影响

    void (*AOS_forward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_forward_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_backward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_backward_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*AOS_forward_zero_irr) (const idx_t, const idx_t, const idx_t, const data_t,
        const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*, const idx_t /* which k */, const calc_t /* contrib to this k */) = nullptr;
    void (*AOS_forward_ALL_irr) (const idx_t, const idx_t, const idx_t, const data_t,
        const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*, const idx_t /* which k */, const calc_t /* contrib to this k */) = nullptr;
    void (*AOS_backward_zero_irr) (const idx_t, const idx_t, const idx_t, const data_t,
        const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*, const idx_t /* which k */, const calc_t /* contrib to this k */) = nullptr;
    void (*AOS_backward_ALL_irr) (const idx_t, const idx_t, const idx_t, const data_t,
        const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*, const idx_t /* which k */, const calc_t /* contrib to this k */) = nullptr;

    void (*SOA_forward_zero)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*SOA_forward_ALL)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*SOA_backward_zero)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*SOA_backward_ALL)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*SOA_forward_zero_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*, const calc_t*) = nullptr;
    void (*SOA_forward_ALL_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*, const calc_t*) = nullptr;
    void (*SOA_backward_zero_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*, const calc_t*) = nullptr;
    void (*SOA_backward_ALL_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*, const calc_t*) = nullptr;

    PointGS() : Solver<idx_t, data_t, calc_t>() {  }
    PointGS(SCAN_TYPE type) : Solver<idx_t, data_t, calc_t>(), scan_type(type) {
        if (type == FORW_BACK)      last_time_forward = false;
        else if (type == BACK_FORW) last_time_forward = true;
    }

    virtual void SetOperator(const Operator<idx_t, calc_t, calc_t> & op) {
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        const idx_t num_diag = ((const par_structMatrix<idx_t, calc_t, calc_t>&)op).num_diag;
        if constexpr (sizeof(calc_t) != sizeof(data_t)) {
#ifdef __aarch64__
            separate_Diags();
            static_assert(sizeof(data_t) < sizeof(calc_t));
            if constexpr (sizeof(data_t) == 2 && sizeof(calc_t) == 4) {// 单-半精度混合计算
                switch (num_diag)
                {
                case 7:
                    SOA_forward_zero = SOA_point_forward_zero_3d7_Cal32Stg16;
                    SOA_forward_ALL  = SOA_point_forward_ALL_3d7_Cal32Stg16;
                    SOA_backward_zero= nullptr;// 没必要实现，实际不会用到
                    SOA_backward_ALL = SOA_point_backward_ALL_3d7_Cal32Stg16;
                    SOA_forward_zero_irr = SOA_point_forward_zero_3d7_Cal32Stg16_irr;
                    SOA_forward_ALL_irr  = SOA_point_forward_ALL_3d7_Cal32Stg16_irr;
                    SOA_backward_zero_irr= nullptr;
                    SOA_backward_ALL_irr = SOA_point_backward_ALL_3d7_Cal32Stg16_irr;
                    break;
                default:
                    MPI_Abort(MPI_COMM_WORLD, -10202);
                }
            }
            else {// 双-半精度混合
                assert(false);
            }
#else
            assert(false);
#endif
        }
        else {// 纯单一精度或者双-单混合
            separate_LU();
            switch(num_diag)
            {
            case 7:
                AOS_forward_zero        = AOS_point_forward_zero_3d7<idx_t, data_t, calc_t>;
                AOS_forward_ALL         = AOS_point_forward_ALL_3d7<idx_t, data_t, calc_t>;
                AOS_backward_zero       = AOS_point_backward_zero_3d7<idx_t, data_t, calc_t>;
                AOS_backward_ALL        = AOS_point_backward_ALL_3d7<idx_t, data_t, calc_t>;
                AOS_forward_zero_irr    = AOS_point_forward_zero_3d7_irr<idx_t, data_t, calc_t>;
                AOS_forward_ALL_irr     = AOS_point_forward_ALL_3d7_irr<idx_t, data_t, calc_t>;
                AOS_backward_zero_irr   = AOS_point_backward_zero_3d7_irr<idx_t, data_t, calc_t>;
                AOS_backward_ALL_irr    = AOS_point_backward_ALL_3d7_irr<idx_t, data_t, calc_t>;
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10200);
            }
        }
        prepare_irrgPts();
    }

    virtual void SetScanType(SCAN_TYPE type) {scan_type = type;}

    void truncate() {
        int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: PGS truncated\n");
        if (LU_separated) {
            assert(L != nullptr && U != nullptr);// 在截断之前务必已经做完分解了
	    	L->truncate();
		    U->truncate();
        }
        if (DiagGroups_separated) {
            printf("Warning: NOT trunc Diags\n");
        }
    }

protected:
    // 近似求解一个（残差）方程，以b为右端向量，返回x为近似解
    void Mult(const par_structVector<idx_t, calc_t> & b, 
                    par_structVector<idx_t, calc_t> & x) const;
public:
    void Mult(const par_structVector<idx_t, calc_t> & b,
                    par_structVector<idx_t, calc_t> & x, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(b, x);
        this->zero_guess = false;// reset for safety concern
    }

    void AOS_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;
    void AOS_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;

    void SOA_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;
    void SOA_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;

    virtual ~PointGS();
};

template<typename idx_t, typename data_t, typename calc_t>
PointGS<idx_t, data_t, calc_t>::~PointGS() {
    if (LU_separated) {
        if (L != nullptr) {delete L; L = nullptr;}
        if (U != nullptr) {delete U; U = nullptr;}
    }
    if (DiagGroups_separated) {
        for (idx_t id = 0; id < DiagGroups_cnt; id++) {
            delete DiagGroups[id];
            DiagGroups[id] = nullptr;
        }
        delete [] DiagGroups; DiagGroups = nullptr;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::prepare_irrgPts()
{
    const par_structMatrix<idx_t, calc_t, calc_t> & par_A = *((par_structMatrix<idx_t, calc_t, calc_t> *)(this->oper));
    if (num_irrgPts_effect != 0)
        delete irrg_to_Struct;

    std::vector<IrrgPts_Effect<idx_t,data_t> > container;
    for (idx_t ir = 0; ir < par_A.num_irrgPts; ir++) {
        idx_t pbeg = par_A.irrgPts[ir].beg, pend = pbeg + par_A.irrgPts[ir].nnz;
        for (idx_t p = pbeg; p < pend; p++) {
            if (par_A.irrgPts_ngb_ijk[p*3] != -1) {
                idx_t loc_i = par_A.irrgPts_ngb_ijk[p*3  ] - par_A.offset_x + par_A.local_matrix->halo_x;
                idx_t loc_j = par_A.irrgPts_ngb_ijk[p*3+1] - par_A.offset_y + par_A.local_matrix->halo_y;
                idx_t loc_k = par_A.irrgPts_ngb_ijk[p*3+2] - par_A.offset_z + par_A.local_matrix->halo_z;
                data_t val = par_A.irrgPts_A_vals[p*2+1];
                IrrgPts_Effect<idx_t, data_t> obj(ir, loc_i, loc_j, loc_k, val);
                container.push_back(obj);
            }
        }
    }
    std::sort(container.begin(), container.end());
    // check sorted
    num_irrgPts_effect = container.size();
    for (idx_t i = 0; i < num_irrgPts_effect - 1; i++) {
        idx_t curr_i = container[i].i;
        idx_t curr_j = container[i].j;
        idx_t curr_k = container[i].k;
        idx_t next_i = container[i+1].i;
        idx_t next_j = container[i+1].j;
        idx_t next_k = container[i+1].k;
#ifdef DISABLE_OMP
        assert(curr_j < next_j || (curr_j==next_j && curr_i < next_i) || (curr_j==next_j && curr_i==next_i && curr_k < next_k));
#else
        // 取巧的写法：因为特别地，在做level-based的并行sptrsv时，
        // 每个level内都只有一个点会被非规则点影响，且这些点的(x,z)坐标都相同
        assert(curr_i == next_i && curr_k == next_k);
        assert(curr_j < next_j);
#endif
    }
    // Copy
    irrg_to_Struct = new IrrgPts_Effect<idx_t,data_t> [num_irrgPts_effect];
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    for (idx_t i = 0; i < num_irrgPts_effect; i++) {
        irrg_to_Struct[i] = container[i];
#ifdef DEBUG
        printf(" proc %d locally %d => (%d,%d,%d) of %lf\n", 
            my_pid, container[i].loc_id, container[i].i, container[i].j, container[i].k, container[i].val);
#endif
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::Mult(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const {
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);

#ifdef PROFILE
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double t = 0.0, bytes, mint, maxt;
    int warm_cnt = 0, test_cnt = 1;
#endif

        switch (scan_type)
        {
        // 零初值优化，注意在对称GS中的后扫和单纯后扫的零初值优化是不一样的，小心！！！
        // 需要注意这个x传进来时可能halo区并不是0，如果不设置0并通信更新halo区，会导致使用错误的值，从而收敛性变慢
        // 还需要注意，这个x传进来时可能数据区不是0，如果不采用0初值的优化代码，而直接采用naive版本但是又没有数据区清零，则会用到旧值，从而收敛性变慢
        case SYMMETRIC:
            if (this->zero_guess) 
                x.set_val(0.0, true);
            else
                x.update_halo();
#ifdef PROFILE
            for (int te = 0; te < warm_cnt; te++) {
                if constexpr (sizeof(data_t) == sizeof(calc_t))
                    AOS_ForwardPass(b, x);
                else
                    SOA_ForwardPass(b, x);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
            for (int te = 0; te < test_cnt; te++) {
#endif
            if constexpr (sizeof(data_t) == sizeof(calc_t))
                AOS_ForwardPass(b, x);
            else
                SOA_ForwardPass(b, x);
#ifdef PROFILE
            }
            t = wall_time() - t; t /= test_cnt;
            MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, b.comm_pkg->cart_comm);
            MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, b.comm_pkg->cart_comm);
            if (my_pid == 0) {
                const idx_t op_nd = ((const par_structMatrix<idx_t, calc_t, calc_t>*)oper)->num_diag;
                int num_diag = this->zero_guess ? (op_nd / 2 + 1) : op_nd;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * 2;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("PGS-F data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), num_diag, bytes, mint, maxt, bytes/maxt, bytes/mint);
            }
#endif
            // 是否要注释掉这行决定后扫零初值的，取决于迭代次数是否会被显著影响
            // 一般在前扫和后扫中间加一次通信，有利于减少迭代数，次数和访存量的减少需要权衡
            this->zero_guess = false;
            x.update_halo();// 通信完之后halo区是非零的，用普通版本的后扫
#ifdef PROFILE
            for (int te = 0; te < warm_cnt; te++) {
                if constexpr (sizeof(data_t) == sizeof(calc_t))    
                    AOS_BackwardPass(b, x);
                else
                    SOA_BackwardPass(b, x);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
            for (int te = 0; te < test_cnt; te++) {
#endif
            if constexpr (sizeof(data_t) == sizeof(calc_t))    
                AOS_BackwardPass(b, x);
            else
                SOA_BackwardPass(b, x);
#ifdef PROFILE
            }
            t = wall_time() - t; t /= test_cnt;
            MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, b.comm_pkg->cart_comm);
            MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, b.comm_pkg->cart_comm);
            if (my_pid == 0) {
                const idx_t op_nd = ((const par_structMatrix<idx_t, calc_t, calc_t>*)oper)->num_diag;
                int num_diag = this->zero_guess ? (op_nd / 2 + 1) : op_nd;
                bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
                      * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * 2;// 向量的数据量
                bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
                bytes *= num_procs;
                bytes /= (1024 * 1024 * 1024);// GB
                printf("PGS-B data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                     sizeof(data_t), sizeof(calc_t), num_diag, bytes, mint, maxt, bytes/maxt, bytes/mint);
            }
#endif
            break;
        case FORWARD:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            if constexpr (sizeof(data_t) == sizeof(calc_t))
                AOS_ForwardPass(b, x);
            else
                SOA_ForwardPass(b, x);
            break;
        case BACKWARD:
            if (this->zero_guess)
                x.set_val(0.0, true);
            else
                x.update_halo();
            if constexpr (sizeof(data_t) == sizeof(calc_t))    
                AOS_BackwardPass(b, x);
            else
                SOA_BackwardPass(b, x);
            break;
        default:// do nothing, just act as an identity operator
            vec_copy(b, x);
            break;
        }
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::AOS_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    assert(LU_separated);

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;

    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const data_t * L_data = L->data, * U_data = U->data;
    const idx_t slice_dki_size = L->slice_dki_size, slice_dk_size = L->slice_dk_size, num_diag = L->num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t, 
                    const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*kernel_irr) (const idx_t, const idx_t, const idx_t, const data_t,
                    const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*,
                    const idx_t /* which k */, const calc_t /* contrib to this k */) = nullptr;
    const bool & scaled = par_A->scaled;
    const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    if (this->zero_guess) {
        kernel = scaled ? nullptr : AOS_forward_zero;
        kernel_irr = scaled ? nullptr : AOS_forward_zero_irr;
    } else {
        kernel = scaled ? nullptr : AOS_forward_ALL;
        kernel_irr = scaled ? nullptr : AOS_forward_ALL_irr;
    }
    assert(kernel);
    assert(kernel_irr);

    // 前扫先处理非规则点
    for (idx_t ir = 0; ir < par_A->num_irrgPts; ir++) {
        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid && x.irrgPts[ir].gid == b.irrgPts[ir].gid);
        const idx_t pbeg = par_A->irrgPts[ir].beg, pend = pbeg + par_A->irrgPts[ir].nnz;
        calc_t tmp = 0.0;
        assert(par_A->irrgPts_ngb_ijk[(pend-1)*3] == -1);
        calc_t diag_val = par_A->irrgPts_A_vals[(pend-1)<<1];
        #pragma omp parallel for schedule(static) reduction(+:tmp)
        for (idx_t p = pbeg; p < pend - 1; p++) {// 跳过了对角元
            const idx_t ngb_i = par_A->irrgPts_ngb_ijk[p*3  ],
                        ngb_j = par_A->irrgPts_ngb_ijk[p*3+1],
                        ngb_k = par_A->irrgPts_ngb_ijk[p*3+2];// global coord
            const idx_t i = ibeg + ngb_i - par_A->offset_x,
                        j = jbeg + ngb_j - par_A->offset_y,
                        k = kbeg + ngb_k - par_A->offset_z;
            tmp += par_A->irrgPts_A_vals[p<<1] * x_data[k + i * vec_k_size + j * vec_ki_size];
        }
        tmp = b.irrgPts[ir].val - tmp;
        x.irrgPts[ir].val *= (1.0 - weight);
        x.irrgPts[ir].val += weight * tmp / diag_val;
    }

    // 再处理结构点：边遍历三维向量边检查是否碰到非规则的邻居
    idx_t ptr = 0, irr_ngb_i = -1, irr_ngb_j = -1, irr_ngb_k = -1;
    bool need_to_check = ptr < num_irrgPts_effect;
    // printf(" proc %d got %d need_to %d\n", my_pid, num_irrgPts_effect, need_to_check);
    if (need_to_check) {
        irr_ngb_i = irrg_to_Struct[ptr].i;  irr_ngb_j = irrg_to_Struct[ptr].j;  irr_ngb_k = irrg_to_Struct[ptr].k;
    }

    if (num_threads > 1) {// level-based的多线程并行
        // level是等值线 slope * j + i = Const, 对于3d7和3d15 斜率为1, 对于3d19和3d27 斜率为2
        const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[0] = dim_1 - 1;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j + 1] = -1;// 初始化为-1
        const idx_t wait_offi = slope - 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = 0; lid < nlevs; lid++) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_end - 1; it >= t_beg; it--) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
                    idx_t i_to_wait = (i == iend - 1) ? i_lev : (i_lev + wait_offi);
                    const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
                    const idx_t vec_off = j * vec_ki_size    + i * vec_k_size    + kbeg;
                    const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off;
                    const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
                    calc_t * x_jik = x_data + vec_off;
                    const calc_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }
                    
                    if (task_check) {
                        idx_t ir = irrg_to_Struct[ptr].loc_id;
                        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                        const calc_t contrib = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;// 非规则点对该柱中某个点的影响值

                        kernel_irr(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - kbeg, contrib);

                        need_to_check = (++ptr) < num_irrgPts_effect;
                        if (need_to_check) {
                            assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                            assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                            irr_ngb_j = irrg_to_Struct[ptr].j;
                        }
                    } else {
                        kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik);
                    }

                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev+1] = i_lev;
                }
                #pragma omp barrier // sync for ptr,need_to_check,irr_ngb_j when each lev done
            }
        }
    }
    else {
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
            const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
            const idx_t vec_off = j * vec_ki_size    + i * vec_k_size    + kbeg;
            const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off;
            const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            if (task_check) {
                idx_t ir = irrg_to_Struct[ptr].loc_id;
                assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                const calc_t contrib = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;// 非规则点对该柱中某个点的影响值

                kernel_irr(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - kbeg, contrib);

                need_to_check = (++ptr) < num_irrgPts_effect;
                if (need_to_check) {
                    assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                    assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                    irr_ngb_j = irrg_to_Struct[ptr].j;
                }
            } else {
                kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik);
            }
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::AOS_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t> *)(this->oper);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    assert(LU_separated);

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;
    
    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const data_t * L_data = L->data, * U_data = U->data;
    const idx_t slice_dki_size = L->slice_dki_size, slice_dk_size = L->slice_dk_size, num_diag = L->num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t, 
                    const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*kernel_irr) (const idx_t, const idx_t, const idx_t, const data_t,
                    const data_t*, const data_t*, const calc_t*, calc_t*, const calc_t*,
                    const idx_t /* which k */, const calc_t /* contrib to this k */) = nullptr;
    const bool & scaled = par_A->scaled;
    const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    if (this->zero_guess) {
        kernel = scaled ? nullptr : AOS_backward_zero;
        kernel_irr = scaled ? nullptr : AOS_backward_zero_irr;
    } else {
        kernel = scaled ? nullptr : AOS_backward_ALL;
        kernel_irr = scaled ? nullptr : AOS_backward_ALL_irr;
    }
    assert(kernel);
    assert(kernel_irr);

    // 后扫先处理结构点
    idx_t ptr = num_irrgPts_effect - 1, irr_ngb_i = -1, irr_ngb_j = -1, irr_ngb_k = -1;
    bool need_to_check = ptr >= 0;
    if (need_to_check) {
        irr_ngb_i = irrg_to_Struct[ptr].i;  irr_ngb_j = irrg_to_Struct[ptr].j;  irr_ngb_k = irrg_to_Struct[ptr].k;
    }

    if (num_threads > 1) {// level-based的多线程并行
        const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j] = dim_1;// 初始化
        const idx_t wait_offi = - (slope - 1);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = nlevs - 1; lid >= 0; lid--) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_beg; it < t_end; it++) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
                    idx_t i_to_wait = (i == ibeg) ? i_lev : (i_lev + wait_offi);
                    const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + (kend - 1) * num_diag;
                    const idx_t vec_off = j * vec_ki_size + i * vec_k_size + (kend - 1);
                    const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off;
                    const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
                    calc_t * x_jik = x_data + vec_off;
                    const calc_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
                    // 中间的不需等待
                    if (task_check) {
                        idx_t ir = irrg_to_Struct[ptr].loc_id;
                        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                        const calc_t contrib = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;// 非规则点对该柱中某个点的影响值

                        kernel_irr(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - (kend - 1), contrib);

                        need_to_check = (--ptr) >= 0;
                        if (need_to_check) {
                            assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                            assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                            irr_ngb_j = irrg_to_Struct[ptr].j;
                        }
                    } else {
                        kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik);
                    }
                    
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev] = i_lev;
                }
                #pragma omp barrier // sync for ptr,need_to_check,irr_ngb_j when each lev done
            }
        }
    }
    else {
        for (idx_t j = jend - 1; j >= jbeg; j--)
        for (idx_t i = iend - 1; i >= ibeg; i--) {
            bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
            const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + (kend - 1) * num_diag;
            const idx_t vec_off = j * vec_ki_size + i * vec_k_size + (kend - 1);
            const data_t * L_jik = L_data + mat_off, * U_jik = U_data + mat_off;
            const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            if (task_check) {
                idx_t ir = irrg_to_Struct[ptr].loc_id;
                assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                const calc_t contrib = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;// 非规则点对该柱中某个点的影响值

                kernel_irr(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik, irr_ngb_k - (kend - 1), contrib);

                need_to_check = (--ptr) >= 0;
                if (need_to_check) {
                    assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                    assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                    irr_ngb_j = irrg_to_Struct[ptr].j;
                }
            } else {
                kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, sqD_jik);
            }
        }
    }

    // 再处理非规则点
    assert(par_A->num_irrgPts == x.num_irrgPts && x.num_irrgPts == b.num_irrgPts);
    for (idx_t ir = par_A->num_irrgPts - 1; ir >= 0; ir--) {
        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid && x.irrgPts[ir].gid == b.irrgPts[ir].gid);
        const idx_t pbeg = par_A->irrgPts[ir].beg, pend = pbeg + par_A->irrgPts[ir].nnz;
        calc_t tmp = 0.0;
        assert(par_A->irrgPts_ngb_ijk[(pend-1)*3] == -1);
        calc_t diag_val = par_A->irrgPts_A_vals[(pend-1)<<1];
        #pragma omp parallel for schedule(static) reduction(+:tmp)
        for (idx_t p = pbeg; p < pend - 1; p++) {// 跳过了对角元
            const idx_t ngb_i = par_A->irrgPts_ngb_ijk[p*3  ],
                        ngb_j = par_A->irrgPts_ngb_ijk[p*3+1],
                        ngb_k = par_A->irrgPts_ngb_ijk[p*3+2];// global coord
            const idx_t i = ibeg + ngb_i - par_A->offset_x,
                        j = jbeg + ngb_j - par_A->offset_y,
                        k = kbeg + ngb_k - par_A->offset_z;
            tmp += par_A->irrgPts_A_vals[p<<1] * x_data[k + i * vec_k_size + j * vec_ki_size];
        }
        tmp = b.irrgPts[ir].val - tmp;
        x.irrgPts[ir].val *= (1.0 - weight);
        x.irrgPts[ir].val += weight * tmp / diag_val;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::separate_LU() {
    assert(this->oper != nullptr);
    assert(!LU_separated);
    assert(sizeof(data_t) == sizeof(calc_t));
    // 提取矩阵对角元到向量，提取L和U到另一个矩阵
    assert(this->oper->input_dim[0] == this->oper->output_dim[0] &&
           this->oper->input_dim[1] == this->oper->output_dim[1] &&
           this->oper->input_dim[2] == this->oper->output_dim[2] );

    const seq_structMatrix<idx_t, calc_t, calc_t> & mat = *(((par_structMatrix<idx_t, calc_t, calc_t> *) oper)->local_matrix);
    const idx_t diag_block_width = 1;
    assert((mat.num_diag - diag_block_width) % 2 ==0);

    L = new seq_structMatrix<idx_t, data_t, calc_t>( (mat.num_diag + diag_block_width) / 2, // 包含对角线
                                            mat.local_x, mat.local_y, mat.local_z, mat.halo_x, mat.halo_y, mat.halo_z);
    U = new seq_structMatrix<idx_t, data_t, calc_t>(*L);

    const idx_t jbeg = 0, jend = mat.local_y + 2 * mat.halo_y,
                ibeg = 0, iend = mat.local_x + 2 * mat.halo_x,
                kbeg = 0, kend = mat.local_z + 2 * mat.halo_z;

    if (mat.num_diag == 7) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
        /*  
                  /--------------/                    
                 / |    4       / |                   
                /--|-----------/  |
               |   |           |  |
               |   |------6----|--|
   z   y       | 1 |    3      | 5|
   ^  ^        |---|-- 0 ------|  |
   | /         |   |           |  |
   |/          |   /-----------|--/ 
   O-------> x | /      2      | / 
               |/--------------|/
                            
                    D : 3
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,
                    U : 4 ,  5,  6
                */
                L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
                L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];
                L->data[L_jik_loc + 3] = mat.data[A_jik_loc + 3];// 最后一个位置放对角元

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 4];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 5];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc + 6];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc + 3];// 最后一个位置放对角元

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
    }
    else if (mat.num_diag == 19) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
                /*
                    D : 9
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,  3,  4,  5,  6,  7,  8
                    U : 10, 11, 12, 13, 14, 15, 16, 17, 18
                */
                L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
                L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];
                L->data[L_jik_loc + 3] = mat.data[A_jik_loc + 3];
                L->data[L_jik_loc + 4] = mat.data[A_jik_loc + 4];
                L->data[L_jik_loc + 5] = mat.data[A_jik_loc + 5];
                L->data[L_jik_loc + 6] = mat.data[A_jik_loc + 6];
                L->data[L_jik_loc + 7] = mat.data[A_jik_loc + 7];
                L->data[L_jik_loc + 8] = mat.data[A_jik_loc + 8];
                L->data[L_jik_loc + 9] = mat.data[A_jik_loc + 9];// 最后一个位置放对角元

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 10];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 11];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc + 12];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc + 13];
                U->data[U_jik_loc + 4] = mat.data[A_jik_loc + 14];
                U->data[U_jik_loc + 5] = mat.data[A_jik_loc + 15];
                U->data[U_jik_loc + 6] = mat.data[A_jik_loc + 16];
                U->data[U_jik_loc + 7] = mat.data[A_jik_loc + 17];
                U->data[U_jik_loc + 8] = mat.data[A_jik_loc + 18];
                U->data[U_jik_loc + 9] = mat.data[A_jik_loc + 9];// 最后一个位置放对角元

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
    }
    else if (mat.num_diag == 27) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
                /*
                    D : 13
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
                    U : 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
                */
                L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
                L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];
                L->data[L_jik_loc + 3] = mat.data[A_jik_loc + 3];
                L->data[L_jik_loc + 4] = mat.data[A_jik_loc + 4];
                L->data[L_jik_loc + 5] = mat.data[A_jik_loc + 5];
                L->data[L_jik_loc + 6] = mat.data[A_jik_loc + 6];
                L->data[L_jik_loc + 7] = mat.data[A_jik_loc + 7];
                L->data[L_jik_loc + 8] = mat.data[A_jik_loc + 8];
                L->data[L_jik_loc + 9] = mat.data[A_jik_loc + 9];
                L->data[L_jik_loc +10] = mat.data[A_jik_loc +10];
                L->data[L_jik_loc +11] = mat.data[A_jik_loc +11];
                L->data[L_jik_loc +12] = mat.data[A_jik_loc +12];
                L->data[L_jik_loc +13] = mat.data[A_jik_loc +13];

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc +14];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc +15];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc +16];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc +17];
                U->data[U_jik_loc + 4] = mat.data[A_jik_loc +18];
                U->data[U_jik_loc + 5] = mat.data[A_jik_loc +19];
                U->data[U_jik_loc + 6] = mat.data[A_jik_loc +20];
                U->data[U_jik_loc + 7] = mat.data[A_jik_loc +21];
                U->data[U_jik_loc + 8] = mat.data[A_jik_loc +22];
                U->data[U_jik_loc + 9] = mat.data[A_jik_loc +23];
                U->data[U_jik_loc +10] = mat.data[A_jik_loc +24];
                U->data[U_jik_loc +11] = mat.data[A_jik_loc +25];
                U->data[U_jik_loc +12] = mat.data[A_jik_loc +26];
                U->data[U_jik_loc +13] = mat.data[A_jik_loc +13];

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
    }
    else {
        printf("PointGS::separate_LDU: num_diag of %d not yet supported\n", mat.num_diag);
        MPI_Abort(MPI_COMM_WORLD, -4000);
    }
    LU_separated = true;
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::separate_Diags() {
    assert(this->oper != nullptr);
    assert(!DiagGroups_separated);
    assert(sizeof(data_t) < sizeof(calc_t));

    const par_structMatrix<idx_t, calc_t, calc_t> & par_A = *((par_structMatrix<idx_t, calc_t, calc_t> *)oper);
    const seq_structMatrix<idx_t, calc_t, calc_t> & seq_A = *(par_A.local_matrix);

    int my_pid; MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
    if (my_pid == 0) {
        printf("Warning: PGS::separate_diags() truncate calc_t of %ld to data_t of %ld bytes\n",
            sizeof(calc_t), sizeof(data_t));
    }

    switch (seq_A.num_diag)
    {
    case  7: DiagGroups_cnt = 2; break;// (0,1,2,3) (3,4,5,6)
    case 19: DiagGroups_cnt = 5; break;// (0,1,2,3) (4,5,6,7) (8,9,10) (11,12,13,14) (15,16,17,18)
    case 27: DiagGroups_cnt = 7; break;// (0,1,2,3) (4,5,6,7) (8,9,10,11) (12,13,14) (15,16,17,18) (19,20,21,22) (23,24,25,26)
    default: MPI_Abort(par_A.comm_pkg->cart_comm, -70891);
    }
    
    DiagGroups = new seq_structMatrix<idx_t, data_t, calc_t>* [DiagGroups_cnt];
    const idx_t hx = seq_A.halo_x,  hy = seq_A.halo_y,  hz = seq_A.halo_z,
                lx = seq_A.local_x, ly = seq_A.local_y, lz = seq_A.local_z;
    const idx_t tot_elems = (lx + hx*2) * (ly + hy*2) * (lz + hz*2);
    if (seq_A.num_diag == 7) {
        DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        #pragma omp parallel for schedule(static)
        for (idx_t e = 0; e < tot_elems; e++) {
            const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
            data_t * L_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * U_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
            L_ptr[0] = aos_ptrs[0]; L_ptr[1] = aos_ptrs[1]; L_ptr[2] = aos_ptrs[2]; L_ptr[3] = aos_ptrs[3];
            U_ptr[0] = aos_ptrs[3]; U_ptr[1] = aos_ptrs[4]; U_ptr[2] = aos_ptrs[5]; U_ptr[3] = aos_ptrs[6];
        }
    }
    else if (seq_A.num_diag == 19) {
        DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[2] = new seq_structMatrix<idx_t, data_t, calc_t>(3, lx, ly, lz, hx, hy, hz);
        DiagGroups[3] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[4] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        #pragma omp parallel for schedule(static)
        for (idx_t e = 0; e < tot_elems; e++) {
            const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
            data_t * G0_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * G1_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
            data_t * G2_ptr = DiagGroups[2]->data + e * DiagGroups[2]->num_diag;
            data_t * G3_ptr = DiagGroups[3]->data + e * DiagGroups[3]->num_diag;
            data_t * G4_ptr = DiagGroups[4]->data + e * DiagGroups[4]->num_diag;
            G0_ptr[0] = aos_ptrs[0]; G0_ptr[1] = aos_ptrs[1]; G0_ptr[2] = aos_ptrs[2]; G0_ptr[3] = aos_ptrs[3];
            G1_ptr[0] = aos_ptrs[4]; G1_ptr[1] = aos_ptrs[5]; G1_ptr[2] = aos_ptrs[6]; G1_ptr[3] = aos_ptrs[7];
            G2_ptr[0] = aos_ptrs[8]; G2_ptr[1] = aos_ptrs[9]; G2_ptr[2] = aos_ptrs[10];
            G3_ptr[0] = aos_ptrs[11]; G3_ptr[1] = aos_ptrs[12]; G3_ptr[2] = aos_ptrs[13]; G3_ptr[3] = aos_ptrs[14];
            G4_ptr[0] = aos_ptrs[15]; G4_ptr[1] = aos_ptrs[16]; G4_ptr[2] = aos_ptrs[17]; G4_ptr[3] = aos_ptrs[18];
        }
    } else if (seq_A.num_diag == 27) {
        DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[2] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[3] = new seq_structMatrix<idx_t, data_t, calc_t>(3, lx, ly, lz, hx, hy, hz);
        DiagGroups[4] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[5] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[6] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        #pragma omp parallel for schedule(static)
        for (idx_t e = 0; e < tot_elems; e++) {
            const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
            data_t * G0_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * G1_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
            data_t * G2_ptr = DiagGroups[2]->data + e * DiagGroups[2]->num_diag;
            data_t * G3_ptr = DiagGroups[3]->data + e * DiagGroups[3]->num_diag;
            data_t * G4_ptr = DiagGroups[4]->data + e * DiagGroups[4]->num_diag;
            data_t * G5_ptr = DiagGroups[5]->data + e * DiagGroups[5]->num_diag;
            data_t * G6_ptr = DiagGroups[6]->data + e * DiagGroups[6]->num_diag;
            G0_ptr[0] = aos_ptrs[0]; G0_ptr[1] = aos_ptrs[1]; G0_ptr[2] = aos_ptrs[2]; G0_ptr[3] = aos_ptrs[3];
            G1_ptr[0] = aos_ptrs[4]; G1_ptr[1] = aos_ptrs[5]; G1_ptr[2] = aos_ptrs[6]; G1_ptr[3] = aos_ptrs[7];
            G2_ptr[0] = aos_ptrs[8]; G2_ptr[1] = aos_ptrs[9]; G2_ptr[2] = aos_ptrs[10];G2_ptr[3] = aos_ptrs[11];
            G3_ptr[0] = aos_ptrs[12]; G3_ptr[1] = aos_ptrs[13]; G3_ptr[2] = aos_ptrs[14];
            G4_ptr[0] = aos_ptrs[15]; G4_ptr[1] = aos_ptrs[16]; G4_ptr[2] = aos_ptrs[17]; G4_ptr[3] = aos_ptrs[18];
            G5_ptr[0] = aos_ptrs[19]; G5_ptr[1] = aos_ptrs[20]; G5_ptr[2] = aos_ptrs[21]; G5_ptr[3] = aos_ptrs[22];
            G6_ptr[0] = aos_ptrs[23]; G6_ptr[1] = aos_ptrs[24]; G6_ptr[2] = aos_ptrs[25]; G6_ptr[3] = aos_ptrs[26];
        }
    }
    DiagGroups_separated = true;
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::SOA_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    assert(DiagGroups_separated);
    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;

    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const bool & scaled = par_A->scaled;

    void (*kernel)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*kernel_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*, const calc_t*) = nullptr;
    idx_t num_arrs;
    if (this->zero_guess) {
        kernel = scaled ? nullptr : SOA_forward_zero;
        kernel_irr = scaled ? nullptr : SOA_forward_zero_irr;
        num_arrs = (DiagGroups_cnt >> 1) + (DiagGroups_cnt & 0x1);// 要跟不同par_A.num_diag的情形对得上
    } else {
        kernel = scaled ? nullptr : SOA_forward_ALL;
        kernel_irr = scaled ? nullptr : SOA_forward_ALL_irr;
        num_arrs = DiagGroups_cnt;
    }
    const idx_t beg_arrId = 0;// 前扫时总是从第0个开始
    const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    assert(kernel);
    assert(kernel_irr);

    // 前扫先处理非规则点
    for (idx_t ir = 0; ir < par_A->num_irrgPts; ir++) {
        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid && x.irrgPts[ir].gid == b.irrgPts[ir].gid);
        const idx_t pbeg = par_A->irrgPts[ir].beg, pend = pbeg + par_A->irrgPts[ir].nnz;
        calc_t tmp = 0.0;
        assert(par_A->irrgPts_ngb_ijk[(pend-1)*3] == -1);
        calc_t diag_val = par_A->irrgPts_A_vals[(pend-1)<<1];
        #pragma omp parallel for schedule(static) reduction(+:tmp)
        for (idx_t p = pbeg; p < pend - 1; p++) {// 跳过了对角元
            const idx_t ngb_i = par_A->irrgPts_ngb_ijk[p*3  ],
                        ngb_j = par_A->irrgPts_ngb_ijk[p*3+1],
                        ngb_k = par_A->irrgPts_ngb_ijk[p*3+2];// global coord
            const idx_t i = ibeg + ngb_i - par_A->offset_x,
                        j = jbeg + ngb_j - par_A->offset_y,
                        k = kbeg + ngb_k - par_A->offset_z;
            tmp += par_A->irrgPts_A_vals[p<<1] * x_data[k + i * vec_k_size + j * vec_ki_size];
        }
        tmp = b.irrgPts[ir].val - tmp;
        x.irrgPts[ir].val *= (1.0 - weight);
        x.irrgPts[ir].val += weight * tmp / diag_val;
    }

    // 再处理结构点：边遍历三维向量边检查是否碰到非规则的邻居
    idx_t ptr = 0, irr_ngb_i = -1, irr_ngb_j = -1, irr_ngb_k = -1;
    bool need_to_check = ptr < num_irrgPts_effect;
    // printf(" proc %d got %d need_to %d\n", my_pid, num_irrgPts_effect, need_to_check);
    if (need_to_check) {
        irr_ngb_i = irrg_to_Struct[ptr].i;  irr_ngb_j = irrg_to_Struct[ptr].j;  irr_ngb_k = irrg_to_Struct[ptr].k;
        // printf(" proc %d before : %d %d %d\n", my_pid, irr_ngb_i, irr_ngb_j, irr_ngb_k);
    }

    if (num_threads > 1) {
        const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[0] = dim_1 - 1;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j + 1] = -1;// 初始化为-1
        const idx_t wait_offi = slope - 1;
        #pragma omp parallel
        {
            const data_t * A_jik[num_arrs];
            calc_t irg_buf[col_height];
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = 0; lid < nlevs; lid++) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_end - 1; it >= t_beg; it--) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
                    idx_t i_to_wait = (i == iend - 1) ? i_lev : (i_lev + wait_offi);
                    const idx_t vec_off = j * vec_ki_size    + i * vec_k_size    + kbeg;
                    for (idx_t id = 0; id < num_arrs; id++) {
                        idx_t gid = beg_arrId + id;
                        A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + i * DiagGroups[gid]->slice_dk_size + kbeg * DiagGroups[gid]->num_diag;
                    }
                    const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
                    calc_t * x_jik = x_data + vec_off;
                    const calc_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }
                    
                    // 中间的不需等待
                    if (task_check) {
                        for (idx_t k = 0; k < col_height; k++)
                            irg_buf[k] = 0.0;

                        idx_t ir = irrg_to_Struct[ptr].loc_id; 
                        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                        irg_buf[irr_ngb_k - kbeg] = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;

                        kernel_irr(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik, irg_buf);

                        need_to_check = (++ptr) < num_irrgPts_effect;
                        if (need_to_check) {
                            assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                            assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                            irr_ngb_j = irrg_to_Struct[ptr].j;
                        }
                    } else {
                        kernel(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik);
                    }
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev+1] = i_lev;
                }
                #pragma omp barrier
            }
        }
    }
    else {
        const data_t * A_jik[num_arrs];
        calc_t irg_buf[col_height];
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
            const idx_t vec_off = j * vec_ki_size    + i * vec_k_size    + kbeg;
            for (idx_t id = 0; id < num_arrs; id++) {
                idx_t gid = beg_arrId + id;
                A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                    + i * DiagGroups[gid]->slice_dk_size + kbeg * DiagGroups[gid]->num_diag;
            }
            const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            if (task_check) {
                for (idx_t k = 0; k < col_height; k++)
                    irg_buf[k] = 0.0;

                idx_t ir = irrg_to_Struct[ptr].loc_id; 
                assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                irg_buf[irr_ngb_k - kbeg] = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;

                kernel_irr(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik, irg_buf);

                need_to_check = (++ptr) < num_irrgPts_effect;
                if (need_to_check) {
                    assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                    assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                    irr_ngb_j = irrg_to_Struct[ptr].j;
                }
            } else {
                kernel(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik);
            }
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void PointGS<idx_t, data_t, calc_t>::SOA_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
    const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t> *)(this->oper);
    CHECK_LOCAL_HALO(x_vec, b_vec);
    assert(DiagGroups_separated);

    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;
    
    const calc_t weight = this->weight;
    const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const bool & scaled = par_A->scaled;

    void (*kernel)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*) = nullptr;
    void (*kernel_irr)
        (const idx_t, const idx_t, const idx_t, const calc_t, const data_t**, const calc_t*, calc_t*, const calc_t*, const calc_t*) = nullptr;
    idx_t num_arrs, beg_arrId;
    if (this->zero_guess) {
        kernel = scaled ? nullptr : SOA_backward_zero;
        kernel_irr = scaled ? nullptr : SOA_backward_zero_irr;
        num_arrs = (DiagGroups_cnt >> 1) + (DiagGroups_cnt & 0x1);// 要跟不同par_A.num_diag的情形对得上
        beg_arrId = DiagGroups_cnt >> 1;
    } else {
        kernel = scaled ? nullptr : SOA_backward_ALL;
        kernel_irr = scaled ? nullptr : SOA_backward_ALL_irr;
        num_arrs = DiagGroups_cnt;
        beg_arrId= 0;
    }
    const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    assert(kernel);
    assert(kernel_irr);

    // 后扫先处理结构点
    idx_t ptr = num_irrgPts_effect - 1, irr_ngb_i = -1, irr_ngb_j = -1, irr_ngb_k = -1;
    bool need_to_check = ptr >= 0;
    if (need_to_check) {
        irr_ngb_i = irrg_to_Struct[ptr].i;  irr_ngb_j = irrg_to_Struct[ptr].j;  irr_ngb_k = irrg_to_Struct[ptr].k;
    }

    if (num_threads > 1) {
        const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j] = dim_1;// 初始化
        const idx_t wait_offi = - (slope - 1);
        #pragma omp parallel
        {
            const data_t * A_jik[num_arrs];
            calc_t irg_buf[col_height];// 非规则点对一整柱的贡献
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // 各自开始计算
            idx_t nlevs = dim_1 + (dim_0 - 1) * slope;
            for (idx_t lid = nlevs - 1; lid >= 0; lid--) {
                // 每层的起始点位于左上角
                idx_t jstart_lev = MIN(lid / slope, dim_0 - 1);
                idx_t istart_lev = lid - slope * jstart_lev;
                idx_t ntasks = MIN(jstart_lev + 1, ((dim_1-1) - istart_lev) / slope + 1);
                // 确定自己分到的task范围
                idx_t my_cnt = ntasks / nt;
                idx_t t_beg = tid * my_cnt;
                idx_t remain = ntasks - my_cnt * nt;
                t_beg += MIN(remain, tid);
                if (tid < remain) my_cnt ++;
                idx_t t_end = t_beg + my_cnt;

                for (idx_t it = t_beg; it < t_end; it++) {
                    idx_t j_lev = jstart_lev - it;
					idx_t i_lev = istart_lev + it * slope;
                    idx_t j = jbeg + j_lev, i = ibeg + i_lev;// 用于数组选址计算的下标
                    bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
                    idx_t i_to_wait = (i == ibeg) ? i_lev : (i_lev + wait_offi);
                    const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kend;
                    for (idx_t id = 0; id < num_arrs; id++) {
                        idx_t gid = beg_arrId + id;
                        A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + i * DiagGroups[gid]->slice_dk_size + kend * DiagGroups[gid]->num_diag;
                    }
                    const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
                    calc_t * x_jik = x_data + vec_off;
                    const calc_t * b_jik = b_data + vec_off;
                    // 线程边界处等待
                    if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
                    if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
                    // 中间的不需等待
                    
                    if (task_check) {
                        for (idx_t k = 0; k < col_height; k++)// 预先准备好非规则点的贡献（一柱数组）
                            irg_buf[k] = 0.0;
                        
                        idx_t ir = irrg_to_Struct[ptr].loc_id;
                        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                        irg_buf[irr_ngb_k - kbeg] = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;
                        
                        kernel_irr(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik, irg_buf);

                        need_to_check = (--ptr) >= 0;
                        if (need_to_check) {
                            assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                            assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                            irr_ngb_j = irrg_to_Struct[ptr].j;
                        }
                    } else {
                        kernel(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik);
                    }
                    if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
                    else flag[j_lev] = i_lev;
                }
                #pragma omp barrier
            }
        }
    }
    else {
        const data_t * A_jik[num_arrs];
        calc_t irg_buf[col_height];// 非规则点对一整柱的贡献
        for (idx_t j = jend - 1; j >= jbeg; j--)
        for (idx_t i = iend - 1; i >= ibeg; i--) {
            bool task_check = need_to_check && j == irr_ngb_j && i == irr_ngb_i;
            const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kend;// 注意这里偏移是kend
            for (idx_t id = 0; id < num_arrs; id++) {
                idx_t gid = beg_arrId + id;
                A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                    + i * DiagGroups[gid]->slice_dk_size + kend * DiagGroups[gid]->num_diag;
            }
            const calc_t * sqD_jik = scaled ? (sqD_data + vec_off) : nullptr;
            calc_t * x_jik = x_data + vec_off;
            const calc_t * b_jik = b_data + vec_off;
            
            if (task_check) {
                for (idx_t k = 0; k < col_height; k++)// 预先准备好非规则点的贡献（一柱数组）
                    irg_buf[k] = 0.0;
                
                idx_t ir = irrg_to_Struct[ptr].loc_id;
                assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid);// 确保是同一个非规则点
                irg_buf[irr_ngb_k - kbeg] = irrg_to_Struct[ptr].val * x.irrgPts[ir].val;
                
                kernel_irr(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik, irg_buf);

                need_to_check = (--ptr) >= 0;
                if (need_to_check) {
                    assert(irr_ngb_i == irrg_to_Struct[ptr].i);
                    assert(irr_ngb_k == irrg_to_Struct[ptr].k);
                    irr_ngb_j = irrg_to_Struct[ptr].j;
                }
            } else {
                kernel(col_height, vec_k_size, vec_ki_size, weight, A_jik, b_jik, x_jik, sqD_jik);
            }
        }
    }

    // 再处理非规则点
    assert(par_A->num_irrgPts == x.num_irrgPts && x.num_irrgPts == b.num_irrgPts);
    for (idx_t ir = par_A->num_irrgPts - 1; ir >= 0; ir--) {
        assert(par_A->irrgPts[ir].gid == x.irrgPts[ir].gid && x.irrgPts[ir].gid == b.irrgPts[ir].gid);
        const idx_t pbeg = par_A->irrgPts[ir].beg, pend = pbeg + par_A->irrgPts[ir].nnz;
        calc_t tmp = 0.0;
        assert(par_A->irrgPts_ngb_ijk[(pend-1)*3] == -1);
        calc_t diag_val = par_A->irrgPts_A_vals[(pend-1)<<1];
        #pragma omp parallel for schedule(static) reduction(+:tmp)
        for (idx_t p = pbeg; p < pend - 1; p++) {// 跳过了对角元
            const idx_t ngb_i = par_A->irrgPts_ngb_ijk[p*3  ],
                        ngb_j = par_A->irrgPts_ngb_ijk[p*3+1],
                        ngb_k = par_A->irrgPts_ngb_ijk[p*3+2];// global coord
            const idx_t i = ibeg + ngb_i - par_A->offset_x,
                        j = jbeg + ngb_j - par_A->offset_y,
                        k = kbeg + ngb_k - par_A->offset_z;
            tmp += par_A->irrgPts_A_vals[p<<1] * x_data[k + i * vec_k_size + j * vec_ki_size];
        }
        tmp = b.irrgPts[ir].val - tmp;
        x.irrgPts[ir].val *= (1.0 - weight);
        x.irrgPts[ir].val += weight * tmp / diag_val;
    }
}

#endif