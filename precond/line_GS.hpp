#ifndef SMG_LINE_GS_HPP
#define SMG_LINE_GS_HPP

#include "line_Solver.hpp"

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
class LineGS : public LineSolver<idx_t, data_t, oper_t, res_t> {
public:
    // 对称GS：0 for sym, 1 for forward, -1 backward
    const SCAN_TYPE scan_type = SYMMETRIC;

	bool LU_separated = false;
    seq_structMatrix<idx_t, data_t, res_t> * L = nullptr;
    seq_structMatrix<idx_t, data_t, res_t> * U = nullptr;
	void separate_LU();

	bool Diags_separated = false;
	idx_t Diags_cnt = 0;
	seq_structVector<idx_t, data_t> ** Diags = nullptr;
	// AOS => SOA
	void separate_diags();

	void (*AOS_forward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const data_t*, const data_t*, data_t*) = nullptr;
    void (*AOS_backward_zero)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const data_t*, const data_t*, data_t*) = nullptr;
    void (*AOS_ALL)
        (const idx_t, const idx_t, const idx_t, const data_t, const data_t*, const data_t*, const data_t*, const data_t*, data_t*) = nullptr;


    LineGS(SCAN_TYPE type, LINE_DIRECTION line_dir, const StructCommPackage & comm_pkg) : 
		LineSolver<idx_t, data_t, oper_t, res_t>(line_dir, comm_pkg), scan_type(type) {  }
	void Mult(const par_structVector<idx_t, res_t> & b,
                    par_structVector<idx_t, res_t> & x) const;
    void Mult(const par_structVector<idx_t, res_t> & b,
                    par_structVector<idx_t, res_t> & x, bool use_zero_guess) const {
        this->zero_guess = use_zero_guess;
        Mult(b, x);
        this->zero_guess = false;// reset for safety concern
    }

	void truncate() {
		int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        if (my_pid == 0) printf("Warning: LGS truncated\n");
		if (Diags_separated) {
#ifdef __aarch64__
			for (idx_t id = 0; id < Diags_cnt; id++) {
				if (Diags[id] != nullptr) {
					idx_t tot_len = (Diags[id]->halo_y * 2 + Diags[id]->local_y)
                                *   (Diags[id]->halo_x * 2 + Diags[id]->local_x) * (Diags[id]->halo_z * 2 + Diags[id]->local_z);
					for (idx_t i = 0; i < tot_len; i++) {
						__fp16 tmp = (__fp16) Diags[id]->data[i];
						Diags[id]->data[i] = (data_t) tmp;
					}
				}
			}
#else
        	printf("architecture not support truncated to fp16\n");
#endif
		}
		if (LU_separated) {
			L->truncate();
			U->truncate();
		}
    }
	void post_setup() {
		if (sizeof(oper_t) != sizeof(data_t)) {
			assert(sizeof(data_t) < sizeof(oper_t));
            MPI_Abort(MPI_COMM_WORLD, -10300);
			// separate_diags();
		}
		else {
			separate_LU();
			const idx_t num_diag = ((const par_structMatrix<idx_t, oper_t, res_t>*)this->oper)->num_diag;
			switch(num_diag)
            {
            case 7:
                AOS_forward_zero    = AOS_line_forward_zero_3d7<idx_t, data_t>;
                AOS_backward_zero   = AOS_line_backward_zero_3d7<idx_t, data_t>;
                AOS_ALL        		= AOS_line_ALL_3d7<idx_t, data_t>;
                break;
			case 19:
				AOS_forward_zero	= AOS_line_forward_zero_3d19<idx_t, data_t>;
				AOS_backward_zero   = AOS_line_backward_zero_3d19<idx_t, data_t>;
				AOS_ALL				= AOS_line_ALL_3d19<idx_t, data_t>;
				break;
			case 27:
				AOS_forward_zero	= AOS_line_forward_zero_3d27<idx_t, data_t>;
				AOS_backward_zero   = AOS_line_backward_zero_3d27<idx_t, data_t>;
				AOS_ALL				= AOS_line_ALL_3d27<idx_t, data_t>;
				break;
            default:
                MPI_Abort(MPI_COMM_WORLD, -10303);
            }
		}
	}
	
    void ForwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
    void BackwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
	void BackwardPass_FFW0(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;

	void ForwardPass_neon_prft(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
    void BackwardPass_neon_prft(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
	void BackwardPass_FFW0_neon_prft(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const;
	virtual ~LineGS();
};

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
LineGS<idx_t, data_t, oper_t, res_t>::~LineGS() {
	if (Diags_separated) {
        for (idx_t id = 0; id < Diags_cnt; id++) {
            delete Diags[id];
            Diags[id] = nullptr;
        }
        delete [] Diags;
        Diags = nullptr;
    }
	if (LU_separated) {
		delete L;
		delete U;
	}
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineGS<idx_t, data_t, oper_t, res_t>::Mult(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const
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
#endif
	if (sizeof(data_t) == sizeof(oper_t)) {
        switch (scan_type)
        {
        case SYMMETRIC:
			if (this->zero_guess)
				x.set_val(0.0, true);
			else
	            x.update_halo();	
#ifdef PROFILE
			MPI_Barrier(MPI_COMM_WORLD);
			t = wall_time();
#endif
			ForwardPass(b, x);
#ifdef PROFILE
			t = wall_time() - t;
			MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, b.comm_pkg->cart_comm);
			MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, b.comm_pkg->cart_comm);
			if (my_pid == 0) {
				int num_diag = L->num_diag + 3 + (this->zero_guess ? 0 : U->num_diag);
				double bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
							 * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
				bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
				bytes *= num_proc;
				bytes /= (1024 * 1024 * 1024);// GB
				printf("Forw LGS data_t %ld oper_t %ld diag %d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
					sizeof(data_t), sizeof(oper_t), num_diag, bytes, mint, maxt, bytes/maxt, bytes/mint);
			}
#endif
            // 一般在前扫和后扫中间加一次通信，有利于减少迭代数，次数和访存量的减少需要权衡
            this->zero_guess = false;
			x.update_halo();// 通信完之后halo区是非零的，用普通版本的后扫
#ifdef PROFILE
			MPI_Barrier(b.comm_pkg->cart_comm);
            t = wall_time();
#endif
			BackwardPass(b, x);
#ifdef PROFILE
            t = wall_time() - t;
			MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, b.comm_pkg->cart_comm);
			MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, b.comm_pkg->cart_comm);
            if (my_pid == 0) {
				int num_diag = U->num_diag + 3 + (this->zero_guess ? 0 : L->num_diag);
				double bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
							 * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
				bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
				bytes *= num_proc;
				bytes /= (1024 * 1024 * 1024);// GB
				printf("Back LGS data_t %ld oper_t %ld diag %d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
					sizeof(data_t), sizeof(oper_t), num_diag, bytes, mint, maxt, bytes/maxt, bytes/mint);
            }
#endif
            break;
        case FORWARD:
			if (this->zero_guess)
				x.set_val(0.0, true);
			else
	            x.update_halo();
            ForwardPass(b, x);
            break;
        case BACKWARD:
			if (this->zero_guess)
				x.set_val(0.0, true);
			else
	            x.update_halo();
            BackwardPass(b, x);
            break;
        default:// do nothing, just act as an identity operator
            vec_copy(b, x);
            break;
        }
	}
	else {
        // 务必从上面正常精度的抄，有重要修改！！！
        /*
		assert(sizeof(data_t) == 2);
		assert(scan_type == SYMMETRIC);
		switch (scan_type)
		{
		case SYMMETRIC:
			if (this->zero_guess)
				x.set_val(0.0, true);
			else
	            x.update_halo();
#ifdef PROFILE
			MPI_Barrier(MPI_COMM_WORLD);
			t = wall_time();
#endif
			ForwardPass_neon_prft(b, x);
#ifdef PROFILE
			MPI_Barrier(MPI_COMM_WORLD);
			t = wall_time() - t;
			if (my_pid == 0) {
				int num_diag;
				if (this->zero_guess)
					num_diag = (LU_separated) ? (L->num_diag + 3) : ((Diags_cnt-3)/2 + 3);
				else
					num_diag = (LU_separated) ? (L->num_diag + U->num_diag + 3) : (Diags_cnt);
				double bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
							 * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
				bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
				bytes *= num_proc;
				bytes /= (1024 * 1024 * 1024);// GB
				printf("Forw LGS data_t %d oper_t %d diag %d total %.2f GB time %.5f s BW %.2f GB/s\n",
					sizeof(data_t), sizeof(oper_t), num_diag, bytes, t, bytes/t);
			}
#endif
            x.update_halo();
			// x.set_halo(0.0);
#ifdef PROFILE
			MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time();
#endif
			if (this->zero_guess)
				BackwardPass_FFW0_neon_prft(b, x);
			else 
				BackwardPass_neon_prft(b, x);
#ifdef PROFILE
			MPI_Barrier(MPI_COMM_WORLD);
            t = wall_time() - t;
            if (my_pid == 0) {
                int num_diag;
				if (this->zero_guess)
					num_diag = (LU_separated) ? (U->num_diag + 3) : ((Diags_cnt-3)/2 + 3);
				else
					num_diag = (LU_separated) ? (L->num_diag + U->num_diag + 3) : (Diags_cnt);
				double bytes = (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
							 * (x.local_vector->local_z + x.local_vector->halo_z * 2) * sizeof(res_t) * 2;// 向量的数据量
				bytes += x.local_vector->local_x * x.local_vector->local_y * x.local_vector->local_z * num_diag * sizeof(data_t);
				bytes *= num_proc;
				bytes /= (1024 * 1024 * 1024);// GB
				printf("Back LGS data_t %d oper_t %d diag %d total %.2f GB time %.5f s BW %.2f GB/s\n",
					sizeof(data_t), sizeof(oper_t), num_diag, bytes, t, bytes/t);
            }
#endif
            break;
		default:// do nothing, just act as an identity operator
            vec_copy(b, x);
            break;
		}
        */
       MPI_Abort(MPI_COMM_WORLD, -10330);
	}
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineGS<idx_t, data_t, oper_t, res_t>::ForwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const
{
    const seq_structVector<idx_t, res_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, res_t> & x_vec = *(x.local_vector);
	const par_structMatrix<idx_t, oper_t, res_t> * par_A = (par_structMatrix<idx_t, oper_t, res_t>*)(this->oper);
	CHECK_LOCAL_HALO(x_vec, b_vec);
	assert(LU_separated);
	const idx_t local_x = x_vec.local_x;
    const res_t * b_data = b_vec.data;
          res_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;

	const res_t weight = this->weight;
	const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const data_t * L_data = L->data, * U_data = (this->zero_guess) ? nullptr : U->data;
    const idx_t slice_dki_size = L->slice_dki_size, slice_dk_size = L->slice_dk_size;
	const idx_t num_diag = L->num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t, 
                    const data_t*, const data_t*, const data_t*, const data_t*, data_t*)
		= (this->zero_guess) ? AOS_forward_zero : AOS_ALL;

	const res_t one_minus_weight = 1.0 - weight;
	if (num_threads > 1) {
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
			res_t buf[(x_vec.slice_k_size + 2) * 2];
			res_t * const sol = buf + 1, * const rhs = buf + (x_vec.slice_k_size + 2) + 1;
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
					const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
					const data_t* L_jik = L_data + mat_off,
								* U_jik = (U_data) ? (U_data + mat_off) : nullptr;
					const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
					res_t * x_jik = x_data + vec_off;
					const res_t * b_jik = b_data + vec_off;
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
					// 线程边界处等待
					if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) < i_lev - 1) {  } // 只需检查W依赖
					if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) < i_to_wait) {  }

					kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, rhs);
					this->tri_solver[sid]->Solve(rhs, sol);// 注意这个偏移！！kbeg == 1
					for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k];

					if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev+1], i_lev, __ATOMIC_RELEASE);
					else flag[j_lev+1] = i_lev;
				}
			}
		}
	}
	else {
		res_t buf[x_vec.slice_k_size << 1];
		res_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
		for (idx_t j = jbeg; j < jend; j++)
		for (idx_t i = ibeg; i < iend; i++) {
			const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
			const data_t* L_jik = L_data + mat_off,
						* U_jik = (U_data) ? (U_data + mat_off) : nullptr;
			const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
			res_t * x_jik = x_data + vec_off;
			const res_t * b_jik = b_data + vec_off;
			idx_t sid = local_x * (j - jbeg) + i - ibeg;

			kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, rhs);
			this->tri_solver[sid]->Solve(rhs, sol);// 注意这个偏移！！kbeg == 1
			for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k];
		}
	}
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineGS<idx_t, data_t, oper_t, res_t>::BackwardPass(const par_structVector<idx_t, res_t> & b, par_structVector<idx_t, res_t> & x) const
{
    const seq_structVector<idx_t, res_t> & b_vec = *(b.local_vector);
          seq_structVector<idx_t, res_t> & x_vec = *(x.local_vector);
	const par_structMatrix<idx_t, oper_t, res_t> * par_A = (par_structMatrix<idx_t, oper_t, res_t>*)(this->oper);
	CHECK_LOCAL_HALO(x_vec, b_vec);
	assert(LU_separated);
	const idx_t local_x = x_vec.local_x;
    const res_t * b_data = b_vec.data;
          res_t * x_data = x_vec.data;

    const idx_t ibeg = b_vec.halo_x, iend = ibeg + b_vec.local_x,
                jbeg = b_vec.halo_y, jend = jbeg + b_vec.local_y,
                kbeg = b_vec.halo_z, kend = kbeg + b_vec.local_z;
    const idx_t vec_k_size = x_vec.slice_k_size, vec_ki_size = x_vec.slice_ki_size;
    
	const res_t weight = this->weight;
	const int num_threads = omp_get_max_threads();
    const idx_t col_height = kend - kbeg;
    const data_t * L_data = (this->zero_guess) ? nullptr : L->data, * U_data = U->data;
    const idx_t slice_dki_size = U->slice_dki_size, slice_dk_size = U->slice_dk_size;
	const idx_t num_diag = U->num_diag;
    void (*kernel) (const idx_t, const idx_t, const idx_t, const data_t, 
                    const data_t*, const data_t*, const data_t*, const data_t*, data_t*)
		= (this->zero_guess) ? AOS_backward_zero : AOS_ALL;
	
	const res_t one_minus_weight = 1.0 - weight;
	if (num_threads > 1) {
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
			res_t buf[x_vec.slice_k_size << 1];
			res_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
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
					const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
					const data_t* U_jik = U_data + mat_off,
								* L_jik = (L_data) ? (L_data + mat_off) : nullptr;
					const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
					res_t * x_jik = x_data + vec_off;
					const res_t * b_jik = b_data + vec_off;
					idx_t sid = local_x * (j - jbeg) + i - ibeg;
					// 线程边界处等待
					if (it == t_beg) while ( __atomic_load_n(&flag[j_lev+1], __ATOMIC_ACQUIRE) > i_to_wait) {  }
					if (it == t_end - 1) while (__atomic_load_n(&flag[j_lev  ], __ATOMIC_ACQUIRE) > i_lev + 1) {  }
					// 中间的不需等待
					kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, rhs);
					this->tri_solver[sid]->Solve(rhs, sol);
					for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k];
					// 写回
					if (it == t_beg || it == t_end - 1) __atomic_store_n(&flag[j_lev], i_lev, __ATOMIC_RELEASE);
					else flag[j_lev] = i_lev;
				}
			}
		}
	}
	else {
		res_t buf[x_vec.slice_k_size << 1];
		res_t * const sol = buf + kbeg, * const rhs = buf + (x_vec.slice_k_size) + kbeg;
		for (idx_t j = jend - 1; j >= jbeg; j--)
		for (idx_t i = iend - 1; i >= ibeg; i--) {
			const idx_t mat_off = j * slice_dki_size + i * slice_dk_size + kbeg * num_diag;
			const data_t* U_jik = U_data + mat_off,
						* L_jik = (L_data) ? (L_data + mat_off) : nullptr;
			const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;
			res_t * x_jik = x_data + vec_off;
			const res_t * b_jik = b_data + vec_off;
			idx_t sid = local_x * (j - jbeg) + i - ibeg;
			
			kernel(col_height, vec_k_size, vec_ki_size, weight, L_jik, U_jik, b_jik, x_jik, rhs);
			this->tri_solver[sid]->Solve(rhs, sol);
			for (idx_t k = 0; k < col_height; k++) x_jik[k] = one_minus_weight * x_jik[k] + weight * sol[k];
		}
	}
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineGS<idx_t, data_t, oper_t, res_t>::separate_LU() {
	assert(this->oper != nullptr);
	assert(!LU_separated);
    // 提取矩阵对角元到向量，提取L和U到另一个矩阵
    assert(this->oper->input_dim[0] == this->oper->output_dim[0] &&
           this->oper->input_dim[1] == this->oper->output_dim[1] &&
           this->oper->input_dim[2] == this->oper->output_dim[2] );

    const seq_structMatrix<idx_t, oper_t, res_t> & mat = *(((par_structMatrix<idx_t, oper_t, res_t>*)(this->oper))->local_matrix);
    // 因为本进程的上下边界需要纳入邻居的贡献，所以也需要存下来上下位置的对角线
    const idx_t diag_block_width = 3;
	assert((mat.num_diag - diag_block_width) % 2 ==0);

    L = new seq_structMatrix<idx_t, data_t, res_t>( (mat.num_diag - diag_block_width) / 2, // 不包含对角线所在的一柱
                                            mat.local_x, mat.local_y, mat.local_z, mat.halo_x, mat.halo_y, mat.halo_z);
    U = new seq_structMatrix<idx_t, data_t, res_t>(*L);

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
    else {
        printf("LineGS::separate_LU: num_diag of %d not yet supported\n", mat.num_diag);
        MPI_Abort(MPI_COMM_WORLD, -4000);
    }
    LU_separated = true;
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineGS<idx_t, data_t, oper_t, res_t>::separate_diags() {
	assert(this->oper != nullptr);
    assert(this->oper->input_dim[0] == this->oper->output_dim[0] &&
           this->oper->input_dim[1] == this->oper->output_dim[1] &&
           this->oper->input_dim[2] == this->oper->output_dim[2] );
	const seq_structMatrix<idx_t, oper_t, res_t> & mat = *(((par_structMatrix<idx_t, oper_t, res_t>*)(this->oper))->local_matrix);
    
	// 如果oper_t和data_t不一样的话，在这里会被截断
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    if (sizeof(oper_t) != sizeof(data_t)) {
        if (my_pid == 0) 
            printf(" Warning: LGS::separate_diags() truncate oper_t of %ld to data_t of %ld bytes\n", sizeof(oper_t), sizeof(data_t));
	}
	if (Diags != nullptr) {
		for (idx_t i = 0; i < Diags_cnt; i++)
			delete Diags[i];
		delete Diags;
	}

	Diags_cnt = mat.num_diag;
	Diags = new seq_structVector<idx_t, data_t> * [Diags_cnt];
	for (idx_t id = 0; id < mat.num_diag; id++) {
		Diags[id] = nullptr;
		if (mat.num_diag ==  7 && (id ==  3)) continue;
		if (mat.num_diag == 27 && (id == 13)) continue;

		Diags[id] = new seq_structVector<idx_t, data_t>(mat.local_x, mat.local_y, mat.local_z, mat.halo_x, mat.halo_y, mat.halo_z);
		data_t * dst = Diags[id]->data;
		oper_t * src = mat.data;

		idx_t tot_elem = (mat.local_x + mat.halo_x * 2) * (mat.local_y + mat.halo_y * 2) * (mat.local_z + mat.halo_z * 2);
		for (idx_t i = 0; i < tot_elem; i++)
			dst[i] = src[i * mat.num_diag + id];
	}
	Diags_separated = true;
}

#endif