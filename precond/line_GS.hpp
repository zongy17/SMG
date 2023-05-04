#ifndef SMG_LINE_GS_HPP
#define SMG_LINE_GS_HPP

#include "line_Solver.hpp"

template<typename idx_t, typename data_t, typename calc_t>
class LineGS : public LineSolver<idx_t, data_t, calc_t> {
public:
    // 对称GS：0 for sym, 1 for forward, -1 backward
    const SCAN_TYPE scan_type = SYMMETRIC;

	bool LU_separated = false;
    seq_structMatrix<idx_t, data_t, calc_t> * L = nullptr;
    seq_structMatrix<idx_t, data_t, calc_t> * U = nullptr;
	void separate_LU();

	bool DiagGroups_separated = false;
    idx_t DiagGroups_cnt = 0;
    seq_structMatrix<idx_t, data_t, calc_t> ** DiagGroups = nullptr;
	// AOS => SOA
	void separate_diags();

#ifdef COMPRESS
	bool compressed = false;
#endif
	void (*AOS_forward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
    void (*AOS_backward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
    void (*AOS_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*AOS_forward_scaled_zero)
        (const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
    void (*AOS_backward_scaled_zero)
        (const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
    void (*AOS_scaled_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;

	void (*SOA_forward_zero)
		(const idx_t, const idx_t, const idx_t, const data_t**, 			  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_forward_ALL)
		(const idx_t, const idx_t, const idx_t, const data_t**, 			  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_backward_zero)
		(const idx_t, const idx_t, const idx_t, const data_t**,				  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_backward_ALL)
		(const idx_t, const idx_t, const idx_t, const data_t**, 			  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_forward_scaled_zero)
		(const idx_t, const idx_t, const idx_t, const data_t**, 			  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_forward_scaled_ALL)
		(const idx_t, const idx_t, const idx_t, const data_t**, 			  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_backward_scaled_zero)
		(const idx_t, const idx_t, const idx_t, const data_t**,				  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	void (*SOA_backward_scaled_ALL)
		(const idx_t, const idx_t, const idx_t, const data_t**, 			  const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;

    LineGS(SCAN_TYPE type, LINE_DIRECTION line_dir, const StructCommPackage * comm_pkg = nullptr) : 
		LineSolver<idx_t, data_t, calc_t>(line_dir, comm_pkg), scan_type(type) {  }
	void Mult(const par_structVector<idx_t, calc_t> & b,
                    par_structVector<idx_t, calc_t> & x) const;
    void Mult(const par_structVector<idx_t, calc_t> & b,
                    par_structVector<idx_t, calc_t> & x, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(b, x);
        this->zero_guess = false;// reset for safety concern
    }

	void truncate() {
		int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: LGS truncated\n");
		if (DiagGroups_separated) {
        	if (my_pid == 0) printf("Warning: NOT trunc Diags\n");
		}
		if (LU_separated) {
			L->truncate();
			U->truncate();
		}
    }
	void post_setup() {
#ifdef COMPRESS
		compressed = ((const par_structMatrix<idx_t, calc_t, calc_t>*)this->oper)->compressed;
#endif
		const idx_t num_diag = ((const par_structMatrix<idx_t, calc_t, calc_t>*)this->oper)->num_diag;
		if constexpr (sizeof(calc_t) != sizeof(data_t) && sizeof(data_t) == 2) {
#ifdef __aarch64__
			static_assert(sizeof(data_t) < sizeof(calc_t));
			separate_diags();
			switch(num_diag)
            {
            case 7:
                SOA_forward_zero    = SOA_line_forward_zero_3d7_Cal32Stg16;
				SOA_forward_ALL		= SOA_line_forward_ALL_3d7_Cal32Stg16;
                SOA_backward_zero   = nullptr;
                SOA_backward_ALL	= SOA_line_backward_ALL_3d7_Cal32Stg16;
                break;
			case 19:
				SOA_forward_zero	= SOA_line_forward_zero_3d19_Cal32Stg16;
				SOA_forward_ALL		= SOA_line_forward_ALL_3d19_Cal32Stg16;
				SOA_backward_zero   = nullptr;
				SOA_backward_ALL	= SOA_line_backward_ALL_3d19_Cal32Stg16;

				SOA_forward_scaled_zero	= SOA_line_forward_scaled_zero_3d19_Cal32Stg16;
				SOA_forward_scaled_ALL	= SOA_line_forward_scaled_ALL_3d19_Cal32Stg16;
				SOA_backward_scaled_zero= nullptr;
				SOA_backward_scaled_ALL	= SOA_line_backward_scaled_ALL_3d19_Cal32Stg16;
				break;
			case 27:
				SOA_forward_zero	= SOA_line_forward_zero_3d27_Cal32Stg16;
				SOA_forward_ALL		= SOA_line_forward_ALL_3d27_Cal32Stg16;
				SOA_backward_zero   = nullptr;
				SOA_backward_ALL	= SOA_line_backward_ALL_3d27_Cal32Stg16;

				SOA_forward_scaled_zero	= SOA_line_forward_scaled_zero_3d27_Cal32Stg16;
				SOA_forward_scaled_ALL	= SOA_line_forward_scaled_ALL_3d27_Cal32Stg16;
				SOA_backward_scaled_zero= nullptr;
				SOA_backward_scaled_ALL	= SOA_line_backward_scaled_ALL_3d27_Cal32Stg16;
				break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10304);
            }
#else
			assert(false);
#endif
		}
		else {// 纯单一精度
			separate_LU();
			switch(num_diag)
            {
			case 9:
				AOS_forward_zero    = AOS_line_forward_zero_2d9<idx_t, data_t, calc_t>;
                AOS_backward_zero   = AOS_line_backward_zero_2d9<idx_t, data_t, calc_t>;
                AOS_ALL        		= AOS_line_ALL_2d9<idx_t, data_t, calc_t>;
                break;
            case 7:
                AOS_forward_zero    = AOS_line_forward_zero_3d7<idx_t, data_t, calc_t>;
                AOS_backward_zero   = AOS_line_backward_zero_3d7<idx_t, data_t, calc_t>;
                AOS_ALL        		= AOS_line_ALL_3d7<idx_t, data_t, calc_t>;
                break;
			case 19:
				AOS_forward_zero	= AOS_line_forward_zero_3d19<idx_t, data_t, calc_t>;
				AOS_backward_zero   = AOS_line_backward_zero_3d19<idx_t, data_t, calc_t>;
				AOS_ALL				= AOS_line_ALL_3d19<idx_t, data_t, calc_t>;
				AOS_forward_scaled_zero	= AOS_line_forward_scaled_zero_3d19<idx_t, data_t, calc_t>;
				AOS_backward_scaled_zero= AOS_line_backward_scaled_zero_3d19<idx_t, data_t, calc_t>;
				AOS_scaled_ALL			= AOS_line_ALL_scaled_3d19<idx_t, data_t, calc_t>;
				break;
			case 27:
				AOS_forward_zero	= AOS_line_forward_zero_3d27<idx_t, data_t, calc_t>;
				AOS_backward_zero   = AOS_line_backward_zero_3d27<idx_t, data_t, calc_t>;
				AOS_ALL				= AOS_line_ALL_3d27<idx_t, data_t, calc_t>;
				AOS_forward_scaled_zero	= AOS_line_forward_scaled_zero_3d27<idx_t, data_t, calc_t>;
				AOS_backward_scaled_zero= AOS_line_backward_scaled_zero_3d27<idx_t, data_t, calc_t>;
				AOS_scaled_ALL			= AOS_line_ALL_scaled_3d27<idx_t, data_t, calc_t>;
				break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10303);
            }
		}
	}
	
    void AOS_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;
    void AOS_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;

	void SOA_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;
    void SOA_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const;
	virtual ~LineGS();
};

template<typename idx_t, typename data_t, typename calc_t>
LineGS<idx_t, data_t, calc_t>::~LineGS() {
	if (DiagGroups_separated) {
        for (idx_t id = 0; id < DiagGroups_cnt; id++) {
            delete DiagGroups[id];
            DiagGroups[id] = nullptr;
        }
        delete [] DiagGroups; DiagGroups = nullptr;
    }
	if (LU_separated) {
		delete L;
		delete U;
	}
}

template<typename idx_t, typename data_t, typename calc_t>
void LineGS<idx_t, data_t, calc_t>::Mult(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    assert(this->oper != nullptr);
    CHECK_INPUT_DIM(*this, x);
    CHECK_OUTPUT_DIM(*this, b);
    assert(b.comm_pkg->cart_comm == x.comm_pkg->cart_comm);
	assert(b.global_size_z == b.local_vector->local_z);// LGS不能做划分

#ifdef PROFILE
	int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
	int num_proc; MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	double t, maxt, mint;
	int warm_cnt = 0, test_cnt = 1;
#endif
        switch (scan_type)
        {
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
				const idx_t op_nd = ((const par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper))->num_diag;
				int num_diag = 3 + (this->zero_guess ? ((op_nd-3)/2) : (op_nd-3));
				double bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
							 * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * 2;// 向量的数据量
				bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
				bytes *= num_proc;
				bytes /= (1024 * 1024 * 1024);// GB
				printf("LGS-F data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
					sizeof(data_t), sizeof(calc_t), num_diag, bytes, mint, maxt, bytes/maxt, bytes/mint);
			}
#endif
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
			MPI_Barrier(b.comm_pkg->cart_comm);
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
				const idx_t op_nd = ((const par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper))->num_diag;
				int num_diag = 3 + (this->zero_guess ? ((op_nd-3)/2) : (op_nd-3));
				double bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
							 * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(calc_t) * 2;// 向量的数据量
				bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
				bytes *= num_proc;
				bytes /= (1024 * 1024 * 1024);// GB
				printf("LGS-B data %ld calc %ld d%d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
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
void LineGS<idx_t, data_t, calc_t>::AOS_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
	const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper);
	CHECK_LOCAL_HALO(x_vec, b_vec);
	assert(LU_separated);
	const idx_t local_x = L->local_x;// 不管有无压缩矩阵数据都能用
    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;

	const bool scaled = par_A->scaled;
	const data_t weight = this->weight;
	const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
	const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    const data_t * L_data = L->data, * U_data = (this->zero_guess) ? nullptr : U->data;
    const idx_t slice_dki_size = L->slice_dki_size, slice_dk_size = L->slice_dk_size;
	const idx_t num_diag = L->num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t,
		const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	if (this->zero_guess) {
		kernel = scaled ? AOS_forward_scaled_zero : AOS_forward_zero;
	} else {
		kernel = scaled ? AOS_scaled_ALL : AOS_ALL;
	}
	assert(kernel);

	const data_t one_minus_weight = 1.0 - weight;
	if (num_threads > 1 && par_A->stencil != stencil_offset_2d9) {// 2d9的不需要有y方向的依赖
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
			calc_t buf[(x_vec.slice_k_size + 2) * 2];
			calc_t * const sol = buf + 1, * const rhs = buf + (x_vec.slice_k_size + 2) + 1;
			calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
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
					idx_t i_to_wait = (i == iend - 1) ? i_lev : (i_lev + wait_offi);
#ifdef COMPRESS
					const idx_t mat_off = j * slice_dki_size + (compressed ? 0 : i * slice_dk_size) + kbeg * num_diag;
#else
					const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
#endif
					const data_t* L_jik = L_data + mat_off,
								* U_jik = (U_data) ? (U_data + mat_off) : nullptr;
					const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
					calc_t * x_jik = x_data + vec_off;
					const calc_t * b_jik = b_data + vec_off;
					const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf;
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
					// 线程边界处等待
					if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
					if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }

					kernel(col_height, vec_k_size, vec_ki_size, L_jik, U_jik, b_jik, x_jik, sqD_jik, rhs);
					this->tri_solver[sid]->Solve(rhs, sol);// 注意这个偏移！！kbeg == 1					
					for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];

					if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
					else flag[j_lev+1] = i_lev;
				}
			}
		}
	}
	else {
		#pragma omp parallel
		{
		calc_t buf[x_vec.slice_k_size << 1];
		calc_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
		calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
		#pragma omp for collapse(2) schedule(static)
		for (idx_t j = jbeg; j < jend; j++)
		for (idx_t i = ibeg; i < iend; i++) {
#ifdef COMPRESS
			const idx_t mat_off = j * slice_dki_size + (compressed ? 0 : i * slice_dk_size) + kbeg * num_diag;
#else
			const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
#endif
			const data_t* L_jik = L_data + mat_off,
						* U_jik = (U_data) ? (U_data + mat_off) : nullptr;
			const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
			calc_t * x_jik = x_data + vec_off;
			const calc_t * b_jik = b_data + vec_off;
			const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf;
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif

			kernel(col_height, vec_k_size, vec_ki_size, L_jik, U_jik, b_jik, x_jik, sqD_jik, rhs);
			this->tri_solver[sid]->Solve(rhs, sol);// 注意这个偏移！！kbeg == 1
			for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];
		}
		}
	}
}

template<typename idx_t, typename data_t, typename calc_t>
void LineGS<idx_t, data_t, calc_t>::AOS_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
    const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
	const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper);
	CHECK_LOCAL_HALO(x_vec, b_vec);
	assert(LU_separated);
	const idx_t local_x = U->local_x;// 不管有无压缩矩阵数据都能用
    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;
    
	const bool scaled = par_A->scaled;
	const calc_t weight = this->weight;
	const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
	const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    const data_t * L_data = (this->zero_guess) ? nullptr : L->data, * U_data = U->data;
    const idx_t slice_dki_size = U->slice_dki_size, slice_dk_size = U->slice_dk_size;
	const idx_t num_diag = U->num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t,
			const data_t*, const data_t*, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	if (this->zero_guess) {
		kernel = scaled ? AOS_backward_scaled_zero : AOS_backward_zero;
	} else {
		kernel = scaled ? AOS_scaled_ALL : AOS_ALL;
	}
	assert(kernel);
	
	const calc_t one_minus_weight = 1.0 - weight;
	if (num_threads > 1 && par_A->stencil != stencil_offset_2d9) {// 2d9的不需要有y方向的依赖
		const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j] = dim_1;// 初始化
        const idx_t wait_offi = - (slope - 1);
		// printf("slope %d wait_offi %d\n", slope, wait_offi);
		#pragma omp parallel
		{
			calc_t buf[x_vec.slice_k_size << 1];
			calc_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
			calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
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
					idx_t i_to_wait = (i == ibeg) ? i_lev : (i_lev + wait_offi);
#ifdef COMPRESS
					const idx_t mat_off = j * slice_dki_size + (compressed ? 0 : i * slice_dk_size) + kbeg * num_diag;
#else
					const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
#endif
					const data_t* U_jik = U_data + mat_off,
								* L_jik = (L_data) ? (L_data + mat_off) : nullptr;
					const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
					calc_t * x_jik = x_data + vec_off;
					const calc_t * b_jik = b_data + vec_off;
					const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf;
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
					// 线程边界处等待
					if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
					if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
					// 中间的不需等待
					kernel(col_height, vec_k_size, vec_ki_size, L_jik, U_jik, b_jik, x_jik, sqD_jik, rhs);
					this->tri_solver[sid]->Solve(rhs, sol);
					for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];
					// 写回
					if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
					else flag[j_lev] = i_lev;
				}
			}
		}
	}
	else {
		#pragma omp parallel
		{
		calc_t buf[x_vec.slice_k_size << 1];
		calc_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
		calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
		#pragma omp for collapse(2) schedule(static)
		for (idx_t j = jend - 1; j >= jbeg; j--)
		for (idx_t i = iend - 1; i >= ibeg; i--) {
#ifdef COMPRESS
					const idx_t mat_off = j * slice_dki_size + (compressed ? 0 : i * slice_dk_size) + kbeg * num_diag;
#else
					const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
#endif
			const data_t* U_jik = U_data + mat_off,
						* L_jik = (L_data) ? (L_data + mat_off) : nullptr;
			const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
			calc_t * x_jik = x_data + vec_off;
			const calc_t * b_jik = b_data + vec_off;
			const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf;
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
			kernel(col_height, vec_k_size, vec_ki_size, L_jik, U_jik, b_jik, x_jik, sqD_jik, rhs);
			this->tri_solver[sid]->Solve(rhs, sol);
			for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];
		}
		}
	}
}

template<typename idx_t, typename data_t, typename calc_t>
void LineGS<idx_t, data_t, calc_t>::separate_LU() {
	assert(this->oper != nullptr);
	assert(!LU_separated);
    // 提取矩阵对角元到向量，提取L和U到另一个矩阵
    assert(this->oper->input_dim[0] == this->oper->output_dim[0] &&
           this->oper->input_dim[1] == this->oper->output_dim[1] &&
           this->oper->input_dim[2] == this->oper->output_dim[2] );

    const seq_structMatrix<idx_t, calc_t, calc_t> & mat = *(((par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper))->local_matrix);
    // 因为本进程的上下边界需要纳入邻居的贡献，所以也需要存下来上下位置的对角线
    const idx_t diag_block_width = 3;
	assert((mat.num_diag - diag_block_width) % 2 ==0);

    L = new seq_structMatrix<idx_t, data_t, calc_t>( (mat.num_diag - diag_block_width) / 2, // 不包含对角线所在的一柱
                                            mat.local_x, mat.local_y, mat.local_z, mat.halo_x, mat.halo_y, mat.halo_z);
    U = new seq_structMatrix<idx_t, data_t, calc_t>(*L);
#ifdef COMPRESS
	L->compressed = U->compressed = compressed;
#endif

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

                    // D : 2, 3, 4
                    // 依据更新中心点时，该点是否已被更新来分类L和U
                    // L : 0, 1
                    // U : 5, 6
                    L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                    L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];

                    U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 5];
                    U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 6];

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
                    D : 8, 9, 10
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
				
                U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 11];
				U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 12];
				U->data[U_jik_loc + 2] = mat.data[A_jik_loc + 13];
				U->data[U_jik_loc + 3] = mat.data[A_jik_loc + 14];
				U->data[U_jik_loc + 4] = mat.data[A_jik_loc + 15];
				U->data[U_jik_loc + 5] = mat.data[A_jik_loc + 16];
				U->data[U_jik_loc + 6] = mat.data[A_jik_loc + 17];
				U->data[U_jik_loc + 7] = mat.data[A_jik_loc + 18];

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
                    D : 12, 13, 14
                    依据更新中心点时，该点是否已被更新来分类L和U
                    L : 0 ,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11
                    U : 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
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

                U->data[U_jik_loc + 0] = mat.data[A_jik_loc +15];
                U->data[U_jik_loc + 1] = mat.data[A_jik_loc +16];
                U->data[U_jik_loc + 2] = mat.data[A_jik_loc +17];
                U->data[U_jik_loc + 3] = mat.data[A_jik_loc +18];
                U->data[U_jik_loc + 4] = mat.data[A_jik_loc +19];
                U->data[U_jik_loc + 5] = mat.data[A_jik_loc +20];
                U->data[U_jik_loc + 6] = mat.data[A_jik_loc +21];
                U->data[U_jik_loc + 7] = mat.data[A_jik_loc +22];
                U->data[U_jik_loc + 8] = mat.data[A_jik_loc +23];
                U->data[U_jik_loc + 9] = mat.data[A_jik_loc +24];
                U->data[U_jik_loc +10] = mat.data[A_jik_loc +25];
                U->data[U_jik_loc +11] = mat.data[A_jik_loc +26];

                A_jik_loc += mat.num_diag;
                L_jik_loc += L->num_diag;
                U_jik_loc += U->num_diag;
            }// k loop
        }
	}
	else if (mat.num_diag == 9 ) {
		#pragma omp parallel for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
			idx_t A_jik_loc = j * mat.slice_dki_size + i * mat.slice_dk_size + kbeg * mat.num_diag;
            idx_t L_jik_loc = j * L->slice_dki_size + i * L->slice_dk_size + kbeg * L->num_diag;
            idx_t U_jik_loc = j * U->slice_dki_size + i * U->slice_dk_size + kbeg * U->num_diag;
            for (idx_t k = kbeg; k < kend; k++) {
                    // D : 3, 4, 5
                    // 依据更新中心点时，该点是否已被更新来分类L和U
                    // L : 0, 1, 2
                    // U : 6, 7, 8
                    L->data[L_jik_loc + 0] = mat.data[A_jik_loc + 0];
                    L->data[L_jik_loc + 1] = mat.data[A_jik_loc + 1];
					L->data[L_jik_loc + 2] = mat.data[A_jik_loc + 2];

                    U->data[U_jik_loc + 0] = mat.data[A_jik_loc + 6];
                    U->data[U_jik_loc + 1] = mat.data[A_jik_loc + 7];
					U->data[U_jik_loc + 2] = mat.data[A_jik_loc + 8];

                    A_jik_loc += mat.num_diag;
                    L_jik_loc += L->num_diag;
                    U_jik_loc += U->num_diag;
            }// k loop
		}
	}
    else {
        printf("LineGS::separate_LU: num_diag of %d not yet supported\n", mat.num_diag);
        MPI_Abort(MPI_COMM_WORLD, -4000);
    }
    LU_separated = true;
}

template<typename idx_t, typename data_t, typename calc_t>
void LineGS<idx_t, data_t, calc_t>::separate_diags() {
	assert(this->oper != nullptr);
	assert(!DiagGroups_separated);
	assert(sizeof(data_t) < sizeof(calc_t));

	const par_structMatrix<idx_t, calc_t, calc_t> & par_A = *((par_structMatrix<idx_t, calc_t, calc_t> *)(this->oper));
    const seq_structMatrix<idx_t, calc_t, calc_t> & seq_A = *(par_A.local_matrix);    

	// 如果calc_t和data_t不一样的话，在这里会被截断
    int my_pid; MPI_Comm_rank(par_A.comm_pkg->cart_comm, &my_pid);
    if (sizeof(calc_t) != sizeof(data_t)) {
        if (my_pid == 0) 
            printf(" Warning: LGS::separate_diags() truncate %d calc_t of %ld to data_t of %ld bytes\n", par_A.num_diag, sizeof(calc_t), sizeof(data_t));
	}

	switch (seq_A.num_diag)
	{
	case  9: DiagGroups_cnt = 3; break;// (0,1,2) || (6,7,8)
	case  7: DiagGroups_cnt = 2; break;// (0,1) || (5,6)
	case 19: DiagGroups_cnt = 4; break;// (0,1,2,3) (4,5,6,7) || (11,12,13,14) (15,16,17,18)
	case 27: DiagGroups_cnt = 6; break;// (0,1,2,3) (4,5,6,7) (8,9,10,11) || (15,16,17,18) (19,20,21,22) (23,24,25,26)
	default: MPI_Abort(par_A.comm_pkg->cart_comm, -70892);
	}
	
	DiagGroups = new seq_structMatrix<idx_t, data_t, calc_t>* [DiagGroups_cnt];
	const idx_t hx = seq_A.halo_x,  hy = seq_A.halo_y,  hz = seq_A.halo_z,
                lx = seq_A.local_x, ly = seq_A.local_y, lz = seq_A.local_z;
    const idx_t tot_elems = (lx + hx*2) * (ly + hy*2) * (lz + hz*2);
	if (seq_A.num_diag == 7) {
		DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(2, lx, ly, lz, hx, hy, hz);
        DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(2, lx, ly, lz, hx, hy, hz);
		#pragma omp parallel for schedule(static)
        for (idx_t e = 0; e < tot_elems; e++) {
            const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
            data_t * L_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * U_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
            L_ptr[0] = aos_ptrs[0]; L_ptr[1] = aos_ptrs[1];
            U_ptr[0] = aos_ptrs[5]; U_ptr[1] = aos_ptrs[6];
        }
	}
	else if (seq_A.num_diag == 19) {
		DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[2] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[3] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        #pragma omp parallel for schedule(static)
        for (idx_t e = 0; e < tot_elems; e++) {
            const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
            data_t * G0_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * G1_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
            data_t * G2_ptr = DiagGroups[2]->data + e * DiagGroups[2]->num_diag;
            data_t * G3_ptr = DiagGroups[3]->data + e * DiagGroups[3]->num_diag;
            G0_ptr[0] = aos_ptrs[0]; G0_ptr[1] = aos_ptrs[1]; G0_ptr[2] = aos_ptrs[2]; G0_ptr[3] = aos_ptrs[3];
            G1_ptr[0] = aos_ptrs[4]; G1_ptr[1] = aos_ptrs[5]; G1_ptr[2] = aos_ptrs[6]; G1_ptr[3] = aos_ptrs[7];
            G2_ptr[0] = aos_ptrs[11]; G2_ptr[1] = aos_ptrs[12]; G2_ptr[2] = aos_ptrs[13]; G2_ptr[3] = aos_ptrs[14];
            G3_ptr[0] = aos_ptrs[15]; G3_ptr[1] = aos_ptrs[16]; G3_ptr[2] = aos_ptrs[17]; G3_ptr[3] = aos_ptrs[18];
        }
	}
	else if (seq_A.num_diag == 27) {
		DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[2] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[3] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[4] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        DiagGroups[5] = new seq_structMatrix<idx_t, data_t, calc_t>(4, lx, ly, lz, hx, hy, hz);
        #pragma omp parallel for schedule(static)
        for (idx_t e = 0; e < tot_elems; e++) {
            const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
            data_t * G0_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * G1_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
            data_t * G2_ptr = DiagGroups[2]->data + e * DiagGroups[2]->num_diag;
            data_t * G3_ptr = DiagGroups[3]->data + e * DiagGroups[3]->num_diag;
            data_t * G4_ptr = DiagGroups[4]->data + e * DiagGroups[4]->num_diag;
            data_t * G5_ptr = DiagGroups[5]->data + e * DiagGroups[5]->num_diag;
            G0_ptr[0] = aos_ptrs[0]; G0_ptr[1] = aos_ptrs[1]; G0_ptr[2] = aos_ptrs[2]; G0_ptr[3] = aos_ptrs[3];
            G1_ptr[0] = aos_ptrs[4]; G1_ptr[1] = aos_ptrs[5]; G1_ptr[2] = aos_ptrs[6]; G1_ptr[3] = aos_ptrs[7];
            G2_ptr[0] = aos_ptrs[8]; G2_ptr[1] = aos_ptrs[9]; G2_ptr[2] = aos_ptrs[10];G2_ptr[3] = aos_ptrs[11];
            G3_ptr[0] = aos_ptrs[15]; G3_ptr[1] = aos_ptrs[16]; G3_ptr[2] = aos_ptrs[17]; G3_ptr[3] = aos_ptrs[18];
            G4_ptr[0] = aos_ptrs[19]; G4_ptr[1] = aos_ptrs[20]; G4_ptr[2] = aos_ptrs[21]; G4_ptr[3] = aos_ptrs[22];
            G5_ptr[0] = aos_ptrs[23]; G5_ptr[1] = aos_ptrs[24]; G5_ptr[2] = aos_ptrs[25]; G5_ptr[3] = aos_ptrs[26];
        }
	}
	else if (seq_A.num_diag == 9) {
		DiagGroups[0] = new seq_structMatrix<idx_t, data_t, calc_t>(3, lx, ly, lz, hx, hy, hz);
		DiagGroups[1] = new seq_structMatrix<idx_t, data_t, calc_t>(3, lx, ly, lz, hx, hy, hz);
		#pragma omp parallel for schedule(static)
		for (idx_t e = 0; e < tot_elems; e++) {
			const calc_t * aos_ptrs = seq_A.data + e * seq_A.num_diag;
			data_t * G0_ptr = DiagGroups[0]->data + e * DiagGroups[0]->num_diag;
            data_t * G1_ptr = DiagGroups[1]->data + e * DiagGroups[1]->num_diag;
			G0_ptr[0] = aos_ptrs[0]; G0_ptr[1] = aos_ptrs[1]; G0_ptr[2] = aos_ptrs[2];
            G1_ptr[0] = aos_ptrs[6]; G1_ptr[1] = aos_ptrs[7]; G1_ptr[2] = aos_ptrs[8];
		}
	}
#ifdef COMPRESS
	for (idx_t g = 0; g < DiagGroups_cnt; g++)
		DiagGroups[g]->compressed = compressed;
#endif
	DiagGroups_separated = true;
}

template<typename idx_t, typename data_t, typename calc_t>
void LineGS<idx_t, data_t, calc_t>::SOA_ForwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
	const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
	const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper);
	CHECK_LOCAL_HALO(x_vec, b_vec);
	assert(DiagGroups_separated);
	const idx_t local_x = DiagGroups[0]->local_x;// 不管有无压缩矩阵数据都能用
    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;

	const int num_threads = omp_get_max_threads();
	const bool scaled = par_A->scaled;
    const idx_t col_height = kend - kbeg;
	const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t**, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	idx_t num_arrs;
	if (this->zero_guess) {
		kernel = scaled ? SOA_forward_scaled_zero : SOA_forward_zero;
		num_arrs = DiagGroups_cnt >> 1;
	} else {
		kernel = scaled ? SOA_forward_scaled_ALL : SOA_forward_ALL;
		num_arrs = DiagGroups_cnt;
	}
	const idx_t beg_arrId = 0;// 前扫时总是从第0个开始
	assert(kernel);
	// printf("LGS-F kernel: %p\n", kernel);

	const data_t weight = this->weight;
	const data_t one_minus_weight = 1.0 - weight;
	if (num_threads > 1 && par_A->stencil != stencil_offset_2d9) {// 2d9不需要y方向的依赖
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
			calc_t buf[(x_vec.slice_k_size + 2) * 2];
			calc_t * const sol = buf + 1, * const rhs = buf + (x_vec.slice_k_size + 2) + 1;
			calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
			int tid = omp_get_thread_num();
			int nt = omp_get_num_threads();
			const data_t * A_jik[num_arrs];
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
					idx_t i_to_wait = (i == iend - 1) ? i_lev : (i_lev + wait_offi);
					const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
					for (idx_t id = 0; id < num_arrs; id++) {
                        idx_t gid = beg_arrId + id;
#ifdef COMPRESS
                        A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + (compressed ? 0 : i * DiagGroups[gid]->slice_dk_size) + kbeg * DiagGroups[gid]->num_diag;
#else
						A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + i * DiagGroups[gid]->slice_dk_size + kbeg * DiagGroups[gid]->num_diag;
#endif
                    }
					const calc_t * b_jik = b_data + vec_off;
					calc_t * x_jik = x_data + vec_off;
					const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf;
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
					TridiagSolver<idx_t, data_t, calc_t> * tridSolver = this->tri_solver[sid];
					{// 预取本柱的三对角系数
						__builtin_prefetch(tridSolver->Get_a(), 0, 0);
						__builtin_prefetch(tridSolver->Get_b(), 0, 0);
						__builtin_prefetch(tridSolver->Get_c(), 0, 0);
					}
					// 线程边界处等待
					if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
					if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }

					kernel(col_height, vec_k_size, vec_ki_size, A_jik, b_jik, x_jik, sqD_jik, rhs);
					tridSolver->Solve_neon_prft(rhs, sol);// 注意这个偏移！！kbeg == 1
					for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];

					if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
					else flag[j_lev+1] = i_lev;
				}
			}
		}
	}
	else {
		#pragma omp parallel
		{
		calc_t buf[x_vec.slice_k_size << 1];
		calc_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
		calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
		const data_t * A_jik[num_arrs];
		#pragma omp for collapse(2) schedule(static)
		for (idx_t j = jbeg; j < jend; j++)
		for (idx_t i = ibeg; i < iend; i++) {
			const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
			for (idx_t id = 0; id < num_arrs; id++) {
				idx_t gid = beg_arrId + id;
#ifdef COMPRESS
                        A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + (compressed ? 0 : i * DiagGroups[gid]->slice_dk_size) + kbeg * DiagGroups[gid]->num_diag;
#else
						A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + i * DiagGroups[gid]->slice_dk_size + kbeg * DiagGroups[gid]->num_diag;
#endif
			}
			const calc_t * b_jik = b_data + vec_off;
			calc_t * x_jik = x_data + vec_off;
			const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf;
			// printf("j %d i %d sqD_jik %p\n", j, i, sqD_jik);
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
			TridiagSolver<idx_t, data_t, calc_t> * tridSolver = this->tri_solver[sid];
			{// 预取本柱的三对角系数
				__builtin_prefetch(tridSolver->Get_a(), 0, 0);
				__builtin_prefetch(tridSolver->Get_b(), 0, 0);
				__builtin_prefetch(tridSolver->Get_c(), 0, 0);
			}
			kernel(col_height, vec_k_size, vec_ki_size, A_jik, b_jik, x_jik, sqD_jik, rhs);
			tridSolver->Solve_neon_prft(rhs, sol);// 注意这个偏移！！kbeg == 1
			for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];
		}
		}
	}

}

template<typename idx_t, typename data_t, typename calc_t>
void LineGS<idx_t, data_t, calc_t>::SOA_BackwardPass(const par_structVector<idx_t, calc_t> & b, par_structVector<idx_t, calc_t> & x) const
{
	const seq_structVector<idx_t, calc_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, calc_t> & x_vec = *(x.local_vector);
	const par_structMatrix<idx_t, calc_t, calc_t> * par_A = (par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper);
	CHECK_LOCAL_HALO(x_vec, b_vec);
	assert(DiagGroups_separated);
	const idx_t local_x = DiagGroups[0]->local_x;// 不管有无压缩矩阵数据都能用
    const calc_t * b_data = b_vec.data;
          calc_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;
    
	const int num_threads = omp_get_max_threads();
	const bool scaled = par_A->scaled;
    const idx_t col_height = kend - kbeg;
	const calc_t * sqD_data = scaled ? par_A->sqrt_D->data : nullptr;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t**, const calc_t*, const calc_t*, const calc_t*, calc_t*) = nullptr;
	idx_t num_arrs, beg_arrId;
	if (this->zero_guess) {
		kernel = scaled ? SOA_backward_scaled_zero : SOA_backward_zero;
		num_arrs = DiagGroups_cnt >> 1;
		beg_arrId= DiagGroups_cnt >> 1;
	} else {
		kernel = scaled ? SOA_backward_scaled_ALL : SOA_backward_ALL;
		num_arrs = DiagGroups_cnt;
		beg_arrId= 0;
	}
	assert(kernel);
	// printf("LGS-B kernel: %p\n", kernel);
	
	const calc_t weight = this->weight;
	const calc_t one_minus_weight = 1.0 - weight;
	if (num_threads > 1 && par_A->stencil != stencil_offset_2d9) {
		const idx_t slope = (par_A->num_diag == 7 || par_A->num_diag == 15) ? 1 : 2;
        idx_t dim_0 = jend - jbeg, dim_1 = iend - ibeg;
        idx_t flag[dim_0 + 1];
        flag[dim_0] = 0;// 边界标记已完成
        for (idx_t j = 0; j < dim_0; j++) 
            flag[j] = dim_1;// 初始化
        const idx_t wait_offi = - (slope - 1);
		// printf("slope %d wait_offi %d\n", slope, wait_offi);
		#pragma omp parallel
		{
			calc_t buf[x_vec.slice_k_size << 1];
			calc_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
			calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
			int tid = omp_get_thread_num();
			int nt = omp_get_num_threads();
			const data_t * A_jik[num_arrs];
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
					idx_t i_to_wait = (i == ibeg) ? i_lev : (i_lev + wait_offi);
					const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kend;
					for (idx_t id = 0; id < num_arrs; id++) {
                        idx_t gid = beg_arrId + id;
#ifdef COMPRESS
                        A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + (compressed ? 0 : i * DiagGroups[gid]->slice_dk_size) + kend * DiagGroups[gid]->num_diag;
#else
						A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + i * DiagGroups[gid]->slice_dk_size + kend * DiagGroups[gid]->num_diag;
#endif
                    }
					const calc_t * b_jik = b_data + vec_off;
					calc_t * x_jik = x_data + vec_off;
					const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf + col_height;// 注意这里的偏移
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
					TridiagSolver<idx_t, data_t, calc_t> * tridSolver = this->tri_solver[sid];
					{// 预取本柱的三对角系数
						__builtin_prefetch(tridSolver->Get_a(), 0, 0);
						__builtin_prefetch(tridSolver->Get_b(), 0, 0);
						__builtin_prefetch(tridSolver->Get_c(), 0, 0);
					}
					// 线程边界处等待
					if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
					if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
					// 中间的不需等待
					kernel(col_height, vec_k_size, vec_ki_size, A_jik, b_jik, x_jik, sqD_jik, rhs + col_height);
					tridSolver->Solve_neon_prft(rhs, sol);
					// printf("j %d i % d rhs sol\n", j, i);
					// for (idx_t k = 0; k < col_height; k++)
					// 	printf("%d %.5e %.5e\n", k, rhs[k], sol[k]);
					// printf("\n");
					x_jik -= col_height;// 重新将写回的位置移动到柱底
					sqD_jik -= col_height;
					for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];
					// 写回
					if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
					else flag[j_lev] = i_lev;
				}
			}
		}
	}
	else {
		#pragma omp parallel
		{
		calc_t buf[x_vec.slice_k_size << 1];
		calc_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
		calc_t sqD_buf[col_height]; for (idx_t k = 0; k < col_height; k++) sqD_buf[k] = 1.0;
		const data_t * A_jik[num_arrs];
		#pragma omp for collapse(2) schedule(static)
		for (idx_t j = jend - 1; j >= jbeg; j--)
		for (idx_t i = iend - 1; i >= ibeg; i--) {
			const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kend;
			for (idx_t id = 0; id < num_arrs; id++) {
				idx_t gid = beg_arrId + id;
#ifdef COMPRESS
                        A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + (compressed ? 0 : i * DiagGroups[gid]->slice_dk_size) + kend * DiagGroups[gid]->num_diag;
#else
						A_jik[id] = DiagGroups[gid]->data + j * DiagGroups[gid]->slice_dki_size
                            + i * DiagGroups[gid]->slice_dk_size + kend * DiagGroups[gid]->num_diag;
#endif
			}
			const calc_t * b_jik = b_data + vec_off;
			calc_t * x_jik = x_data + vec_off;
			const calc_t * sqD_jik = sqD_data ? sqD_data + vec_off : sqD_buf + col_height;
#ifdef COMPRESS
					idx_t sid = local_x * (j - jbeg) + (compressed ? 0 : i - ibeg);
#else
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
#endif
			TridiagSolver<idx_t, data_t, calc_t> * tridSolver = this->tri_solver[sid];
			{// 预取本柱的三对角系数
				__builtin_prefetch(tridSolver->Get_a(), 0, 0);
				__builtin_prefetch(tridSolver->Get_b(), 0, 0);
				__builtin_prefetch(tridSolver->Get_c(), 0, 0);
			}
			kernel(col_height, vec_k_size, vec_ki_size, A_jik, b_jik, x_jik, sqD_jik, rhs + col_height);
			
			tridSolver->Solve_neon_prft(rhs, sol);

			x_jik -= col_height;// 重新将写回的位置移动到柱底
			sqD_jik -= col_height;
			for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k] / sqD_jik[k];
		}
		}
	}
}

#endif