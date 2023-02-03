#ifndef SMG_PAR_STRUCT_MV_HPP
#define SMG_PAR_STRUCT_MV_HPP

#include "common.hpp"
#include "par_struct_vec.hpp"
#include "operator.hpp"

// data_t是矩阵数据存储的精度，calc_t是矩阵运算时的精度
template<typename idx_t, typename data_t, typename calc_t>
class par_structMatrix : public Operator<idx_t, data_t, calc_t>  {
public:
    idx_t num_diag;
    idx_t offset_x     , offset_y     , offset_z     ;// 该矩阵在全局中的偏移
    bool scaled = false;
    seq_structVector<idx_t, data_t> * sqrt_D = nullptr;

    seq_structMatrix<idx_t, data_t, calc_t> * local_matrix;

    // 对角线打包 每四个一组
    mutable bool DiagGroups_separated = false;
    mutable idx_t DiagGroups_cnt = 0;
    mutable seq_structMatrix<idx_t, data_t, calc_t> ** DiagGroups = nullptr;
    // spmv的函数指针
    void (* SOA_spmv)(const idx_t, const idx_t, const idx_t, const data_t **, const calc_t *, calc_t *) = nullptr;
    const idx_t * stencil = nullptr;

    // 通信相关的
    StructCommPackage * comm_pkg = nullptr;
    bool own_comm_pkg = false;

    par_structMatrix(MPI_Comm comm, idx_t num_d, idx_t gx, idx_t gy, idx_t gz, idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z);
    // 按照model的规格生成一个结构化向量，浅拷贝通信包
    par_structMatrix(const par_structMatrix & model);
    ~par_structMatrix();

    void setup_cart_comm(MPI_Comm comm, idx_t px, idx_t py, idx_t pz, bool unblk);
    void setup_comm_pkg(bool need_corner=true);

    void truncate() {
        int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
        if (my_pid == 0) printf("Warning: parMat truncated and sqrt_D to __fp16 (Consider to use f32 if effect bad!)\n");
        local_matrix->truncate();
        if (sqrt_D != nullptr) {
            const idx_t sqD_len = (sqrt_D->local_x + sqrt_D->halo_x * 2) * (sqrt_D->local_y + sqrt_D->halo_y * 2)
                                * (sqrt_D->local_z + sqrt_D->halo_z * 2);
#ifdef __aarch64__
            for (idx_t p = 0; p < sqD_len; p++) {
                __fp16 tmp = (__fp16) sqrt_D->data[p];
                // if (p == sqD_len / 3) printf("parMat::sqD truncate %.20e to", sqrt_D->data[p]);
                sqrt_D->data[p] = (data_t) tmp;
                // if (p == sqD_len / 3) printf("%.20e\n", sqrt_D->data[p]);
            }
#else
            printf("architecture not support truncated to fp16\n");
#endif
        }
    }
    void separate_Diags() const;
    void separate_truncate_Diags(const seq_structMatrix<idx_t, calc_t, calc_t> & A_high);
    void update_halo();
    void Mult(const par_structVector<idx_t, calc_t> & x, 
                    par_structVector<idx_t, calc_t> & y, bool use_zero_guess/* ignored */) const;
protected:
    void SOA_Mult(const seq_structVector<idx_t, calc_t> & x, seq_structVector<idx_t, calc_t> & y) const ;
    void AOS_Mult(const seq_structVector<idx_t, calc_t> & x, seq_structVector<idx_t, calc_t> & y) const {
        local_matrix->Mult(x, y, (scaled ? sqrt_D : nullptr));
    }

public:
    void read_data(const std::string pathname);
    void write_data(const std::string pathname, const std::string prefix);
    void set_val(data_t val, bool halo_set=false);
    void set_diag_val(idx_t d, data_t val);
    void init_random_base2(idx_t max_power = 6);

    void mul(const data_t factor) {
        local_matrix->mul(factor);
    } 
    bool check_Dirichlet();
    void set_boundary();
    void scale(const data_t scaled_diag);
    bool check_scaling(const data_t scaled_diag);
    void write_CSR_bin(const std::string pathname) const;
    void write_struct_AOS_bin(const std::string pathname, const std::string file) const;
};


/*
 * * * * * par_structMatrix * * * * *  
 */

template<typename idx_t, typename data_t, typename calc_t>
par_structMatrix<idx_t, data_t, calc_t>::par_structMatrix(MPI_Comm comm, idx_t num_d,
    idx_t global_size_x, idx_t global_size_y, idx_t global_size_z, idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z)
    : Operator<idx_t, data_t, calc_t>(global_size_x, global_size_y, global_size_z, global_size_x, global_size_y, global_size_z), 
        num_diag(num_d)
{
    // for GMG concern: must be fully divided by processors
    assert(global_size_x % num_proc_x == 0);
    assert(global_size_y % num_proc_y == 0);
    assert(global_size_z % num_proc_z == 0);

    setup_cart_comm(comm, num_proc_x, num_proc_y, num_proc_z, false);

    int (&cart_ids)[3] = comm_pkg->cart_ids;
    offset_y = cart_ids[0] * global_size_y / num_proc_y;
    offset_x = cart_ids[1] * global_size_x / num_proc_x;
    offset_z = cart_ids[2] * global_size_z / num_proc_z;

    // 建立本地数据的内存
    local_matrix = new seq_structMatrix<idx_t, data_t, calc_t>
        (num_diag, global_size_x / num_proc_x, global_size_y / num_proc_y, global_size_z / num_proc_z, 1, 1, 1);
    
    setup_comm_pkg();// 3d7的时候不需要角上的数据通信

    switch (num_diag)
    {
    case 7:
        stencil = stencil_offset_3d7;
#ifdef __aarch64__
        if constexpr (sizeof(data_t) != sizeof(calc_t)) SOA_spmv = SOA_spmv_3d7_Cal32Stg16;
#endif
        break;
    case 19:
        stencil = stencil_offset_3d19;
#ifdef __aarch64__
        if constexpr (sizeof(data_t) != sizeof(calc_t)) SOA_spmv = SOA_spmv_3d19_Cal32Stg16;
#endif
        break;
    case 27:
        stencil = stencil_offset_3d27;
#ifdef __aarch64__
        if constexpr (sizeof(data_t) != sizeof(calc_t)) SOA_spmv = SOA_spmv_3d27_Cal32Stg16;
#endif
        break;
    default:
        printf("not supported number of diagonals %d\n", num_diag);
        MPI_Abort(comm, -5577);
    }
}

template<typename idx_t, typename data_t, typename calc_t>
par_structMatrix<idx_t, data_t, calc_t>::par_structMatrix(const par_structMatrix & model) 
    : Operator<idx_t, data_t, calc_t>(  model.input_dim[0], model.input_dim[1], model.input_dim[2], 
                                        model.output_dim[0], model.output_dim[1], model.output_dim[2]),
        num_diag(model.num_diag), offset_x(model.offset_x), offset_y(model.offset_y), offset_z(model.offset_z)
{
    local_matrix = new seq_structMatrix<idx_t, data_t, calc_t>(*(model.local_matrix));
    // 浅拷贝
    comm_pkg = model.comm_pkg;
    own_comm_pkg = false;
    SOA_spmv = model.SOA_spmv;
    stencil = model.stencil;
}

template<typename idx_t, typename data_t, typename calc_t>
par_structMatrix<idx_t, data_t, calc_t>::~par_structMatrix()
{
    delete local_matrix;
    local_matrix = nullptr;
    if (own_comm_pkg) {
        delete comm_pkg;
        comm_pkg = nullptr;
    }
    if (DiagGroups != nullptr) {
        for (idx_t i = 0; i < DiagGroups_cnt; i++) {
            delete DiagGroups[i];
            DiagGroups[i] = nullptr;
        }
        delete [] DiagGroups;
    }
    if (scaled) {
        assert(sqrt_D != nullptr);
        delete sqrt_D; sqrt_D = nullptr;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::setup_cart_comm(MPI_Comm comm, 
    idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z, bool unblk)
{
    // bool relay_mode = unblk ? false : true;
    bool relay_mode = true;
    comm_pkg = new StructCommPackage(relay_mode);
    own_comm_pkg = true;
    // 对comm_pkg内变量的引用，免得写太麻烦了
    MPI_Comm & cart_comm                         = comm_pkg->cart_comm;
    int (&cart_ids)[3]                           = comm_pkg->cart_ids;
    int (&ngbs_pid)[NUM_NEIGHBORS]               = comm_pkg->ngbs_pid;
    int & my_pid                                 = comm_pkg->my_pid;

    // create 2D distributed grid
    int dims[3] = {num_proc_y, num_proc_x, num_proc_z};
    int periods[3] = {0, 0, 0};

    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
    assert(cart_comm != MPI_COMM_NULL);

    MPI_Cart_shift(cart_comm, 0, 1, &ngbs_pid[J_L], &ngbs_pid[J_U]);
    MPI_Cart_shift(cart_comm, 1, 1, &ngbs_pid[I_L], &ngbs_pid[I_U]);
    MPI_Cart_shift(cart_comm, 2, 1, &ngbs_pid[K_L], &ngbs_pid[K_U]);

    MPI_Comm_rank(cart_comm, &my_pid);
    MPI_Cart_coords(cart_comm, my_pid, 3, cart_ids);

#ifdef DEBUG
    printf("proc %3d cart_ids (%3d,%3d,%3d) IL %3d IU %3d JL %3d JU %3d KL %3d KU %3d\n",
        my_pid, cart_ids[0], cart_ids[1], cart_ids[2],  
        ngbs_pid[I_L], ngbs_pid[I_U], ngbs_pid[J_L], ngbs_pid[J_U], ngbs_pid[K_L], ngbs_pid[K_U]);
#endif
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::setup_comm_pkg(bool need_corner)
{
    MPI_Datatype (&send_subarray)[NUM_NEIGHBORS] = comm_pkg->send_subarray;
    MPI_Datatype (&recv_subarray)[NUM_NEIGHBORS] = comm_pkg->recv_subarray;
    MPI_Datatype & mpi_scalar_type               = comm_pkg->mpi_scalar_type;
    // 建立通信结构：注意data的排布从内到外依次为diag(3)->k(2)->i(1)->j(0)，按照C-order
    if      constexpr (sizeof(data_t) == 16)  comm_pkg->mpi_scalar_type = MPI_LONG_DOUBLE;
    else if constexpr (sizeof(data_t) == 8)   comm_pkg->mpi_scalar_type = MPI_DOUBLE;
    else if constexpr (sizeof(data_t) == 4)   comm_pkg->mpi_scalar_type = MPI_FLOAT;
    else if constexpr (sizeof(data_t) == 2)   comm_pkg->mpi_scalar_type = MPI_SHORT;
    else { printf("INVALID data_t when creating subarray, sizeof %ld bytes\n", sizeof(data_t)); MPI_Abort(MPI_COMM_WORLD, -2001); }

    idx_t size[4] = {   local_matrix->local_y + 2 * local_matrix->halo_y,
                        local_matrix->local_x + 2 * local_matrix->halo_x,
                        local_matrix->local_z + 2 * local_matrix->halo_z,
                        local_matrix->num_diag  };
    idx_t subsize[4], send_start[4], recv_start[4];
    for (idx_t ingb = 0; ingb < NUM_NEIGHBORS; ingb++) {
        switch (ingb)
        {
        // 最先传的
        case K_L:
        case K_U:
            subsize[0] = local_matrix->local_y;
            subsize[1] = local_matrix->local_x;
            subsize[2] = local_matrix->halo_z;
            break;
        case I_L:
        case I_U:
            subsize[0] = local_matrix->local_y;
            subsize[1] = local_matrix->halo_x;
            subsize[2] = local_matrix->local_z + (need_corner ? 2 * local_matrix->halo_z : 0);
            break;
        case J_L:
        case J_U:
            subsize[0] = local_matrix->halo_y;
            subsize[1] = local_matrix->local_x + (need_corner ? 2 * local_matrix->halo_x : 0);
            subsize[2] = local_matrix->local_z + (need_corner ? 2 * local_matrix->halo_z : 0);
            break;
        default:
            printf("INVALID NEIGHBOR ID of %d!\n", ingb);
            MPI_Abort(MPI_COMM_WORLD, -2000);
        }
        // 最内维的高度层和矩阵元素层的通信长度不变
        subsize[3] = local_matrix->num_diag;

        switch (ingb)
        {
        case K_L:// 向K下发的内halo
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = recv_start[1] = local_matrix->halo_x;
            send_start[2] = local_matrix->halo_z;           recv_start[2] = 0;
            break;
        case K_U:
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = recv_start[1] = local_matrix->halo_x;
            send_start[2] = local_matrix->local_z;          recv_start[2] = local_matrix->local_z + local_matrix->halo_z;
            break;
        case I_L:
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = local_matrix->halo_x;           recv_start[1] = 0;
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        case I_U:
            send_start[0] = recv_start[0] = local_matrix->halo_y;
            send_start[1] = local_matrix->local_x;          recv_start[1] = local_matrix->local_x + local_matrix->halo_x;
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        case J_L:
            send_start[0] = local_matrix->halo_y;           recv_start[0] = 0;
            send_start[1] = recv_start[1] = (need_corner ? 0 : local_matrix->halo_x);
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        case J_U:
            send_start[0] = local_matrix->local_y;          recv_start[0] = local_matrix->local_y + local_matrix->halo_y;
            send_start[1] = recv_start[1] = (need_corner ? 0 : local_matrix->halo_x);
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_matrix->halo_z);
            break;
        default:
            printf("INVALID NEIGHBOR ID of %d!\n", ingb);
            MPI_Abort(MPI_COMM_WORLD, -2000);
        }
        // 最内维的高度层的通信起始位置不变
        send_start[3] = recv_start[3] = 0;

        MPI_Type_create_subarray(4, size, subsize, send_start, MPI_ORDER_C, mpi_scalar_type, &send_subarray[ingb]);
        MPI_Type_commit(&send_subarray[ingb]);
        MPI_Type_create_subarray(4, size, subsize, recv_start, MPI_ORDER_C, mpi_scalar_type, &recv_subarray[ingb]);
        MPI_Type_commit(&recv_subarray[ingb]);
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::update_halo()
{
#ifdef DEBUG
    local_matrix->init_debug(offset_x, offset_y, offset_z);
    if (my_pid == 1) {
        local_matrix->print_level_diag(1, 3);
    }
#endif

    comm_pkg->exec_comm(local_matrix->data);

#ifdef DEBUG
    if (my_pid == 1) {
        local_matrix->print_level_diag(1, 3);
    }
#endif
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::Mult(const par_structVector<idx_t, calc_t> & x, par_structVector<idx_t, calc_t> & y,
    bool use_zero_guess/* ignored */) const
{
    assert( this->input_dim[0] == x.global_size_x && this->output_dim[0] == y.global_size_x &&
            this->input_dim[1] == x.global_size_y && this->output_dim[1] == y.global_size_y &&
            this->input_dim[2] == x.global_size_z && this->output_dim[2] == y.global_size_z    );

    // lazy halo updated: only done when needed 
    x.update_halo();

#ifdef PROFILE
    int my_pid; MPI_Comm_rank(y.comm_pkg->cart_comm, &my_pid);
    int num_procs; MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double bytes = local_matrix->local_x * local_matrix->local_y
                 * local_matrix->local_z * num_diag * sizeof(data_t);
    bytes += (x.local_vector->local_x + x.local_vector->halo_x * 2) * (x.local_vector->local_y + x.local_vector->halo_y * 2)
           * (x.local_vector->local_z + x.local_vector->halo_z * 2) * 2 * sizeof(calc_t);
    bytes *= num_procs;
    bytes /= (1024 * 1024 * 1024);// GB
    MPI_Barrier(y.comm_pkg->cart_comm);
    int warm_cnt = 5;
    for (int te = 0; te < warm_cnt; te++) {
        if constexpr (sizeof(data_t) != sizeof(calc_t))
            SOA_Mult(*(x.local_vector), *(y.local_vector));
        else
            AOS_Mult(*(x.local_vector), *(y.local_vector));
    }
    MPI_Barrier(y.comm_pkg->cart_comm);
    double t = wall_time();
    int test_cnt = 20;
    for (int te = 0; te < test_cnt; te++) {
#endif
    // do computation
    if constexpr (sizeof(data_t) != sizeof(calc_t))
        SOA_Mult(*(x.local_vector), *(y.local_vector));
    else
        AOS_Mult(*(x.local_vector), *(y.local_vector));
#ifdef PROFILE
    }
    t = wall_time() - t;
    t /= test_cnt;
    double mint, maxt;
    MPI_Allreduce(&t, &maxt, 1, MPI_DOUBLE, MPI_MAX, y.comm_pkg->cart_comm);
    MPI_Allreduce(&t, &mint, 1, MPI_DOUBLE, MPI_MIN, y.comm_pkg->cart_comm);
    // mint = maxt = t;
    if (my_pid == 0) printf("SpMv data_t %ld calc_t %ld diag %d total %.2f GB time %.5f/%.5f s BW %.2f/%.2f GB/s\n",
                sizeof(data_t), sizeof(calc_t), num_diag, bytes, mint, maxt, bytes/maxt, bytes/mint);
#endif
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::SOA_Mult(const seq_structVector<idx_t, calc_t> & x, seq_structVector<idx_t, calc_t> & y) const
{
    CHECK_LOCAL_HALO(*local_matrix, x);
    CHECK_LOCAL_HALO(x , y);
    assert(DiagGroups_separated);
    assert(SOA_spmv);
    const calc_t * x_data = x.data;
    calc_t * y_data = y.data;

    const idx_t ibeg = x.halo_x, iend = ibeg + x.local_x,
                jbeg = x.halo_y, jend = jbeg + x.local_y,
                kbeg = x.halo_z, kend = kbeg + x.local_z;
    const idx_t vec_k_size = x.slice_k_size, vec_ki_size = x.slice_ki_size;
    const idx_t col_height = kend - kbeg;

    #pragma omp parallel
    {
        const data_t * A_jik[DiagGroups_cnt];
        #pragma omp for collapse(2) schedule(static)
        for (idx_t j = jbeg; j < jend; j++)
        for (idx_t i = ibeg; i < iend; i++) {
            const idx_t vec_off = j * vec_ki_size + i * vec_k_size + kbeg;// 一定要有kbeg
            const calc_t * x_jik = x_data + vec_off;
            calc_t * y_jik = y_data + vec_off;
            for (idx_t g = 0; g < DiagGroups_cnt; g++)
                // A_jik[g] = DG_data[g] + j * slice_dki[g]
                //     + i * slice_dk[g] + kbeg * nd[g];
                A_jik[g] = DiagGroups[g]->data + j * DiagGroups[g]->slice_dki_size
                        + i * DiagGroups[g]->slice_dk_size + kbeg * DiagGroups[g]->num_diag;
            SOA_spmv(col_height, vec_k_size, vec_ki_size, A_jik, x_jik, y_jik);
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::read_data(const std::string pathname) {
    seq_structMatrix<idx_t, data_t, calc_t> & A_local = *local_matrix;

    assert( (sizeof(data_t) ==16 && comm_pkg->mpi_scalar_type == MPI_LONG_DOUBLE) || 
            (sizeof(data_t) == 4 && comm_pkg->mpi_scalar_type == MPI_FLOAT)  ||
            (sizeof(data_t) == 8 && comm_pkg->mpi_scalar_type == MPI_DOUBLE) ||
            (sizeof(data_t) == 2 && comm_pkg->mpi_scalar_type == MPI_SHORT )    );
    
    idx_t lx = A_local.local_x, ly = A_local.local_y, lz = A_local.local_z;
    idx_t tot_len = lx * ly * lz;
    if (strstr(pathname.c_str(), "LASER")) {
        assert(sizeof(data_t) == 8);
    } else if (strstr(pathname.c_str(), "GRAPES")) {
        assert(sizeof(data_t) == 4);
    }

    size_t bytes = sizeof(data_t) * tot_len;
    void * buf = malloc(bytes);// 读入缓冲区（给定的数据是双精度的）

    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = (sizeof(data_t) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Datatype read_type = MPI_DATATYPE_NULL;// 读取时的类型

    idx_t size[3], subsize[3], start[3];
    size[0] = this->input_dim[1];// y 方向的全局大小
    size[1] = this->input_dim[0];// x 方向的全局大小
    size[2] = this->input_dim[2];// z 方向的全局大小
    subsize[0] = ly;
    subsize[1] = lx;
    subsize[2] = lz;
    start[0] = offset_y;
    start[1] = offset_x;
    start[2] = offset_z;
    
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, etype, &read_type);
    MPI_Type_commit(&read_type);

    MPI_Offset displacement = 0;
    displacement *= sizeof(data_t);// 位移要以字节为单位！
    MPI_Status status;

    // 依次读入A的各条对角线
    for (idx_t idiag = 0; idiag < num_diag; idiag++) {
        const std::string filename = pathname + "/array_a." + std::to_string(idiag);
        MPI_File_open(comm_pkg->cart_comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        MPI_File_set_view(fh, displacement, etype, read_type, "native", MPI_INFO_NULL);
        MPI_File_read_all(fh, buf, tot_len, etype, &status);
        for (idx_t j = 0; j < ly; j++)
        for (idx_t i = 0; i < lx; i++)
        for (idx_t k = 0; k < lz; k++)
            A_local.data[idiag +
                        (k + A_local.halo_z) * A_local.num_diag + 
                        (i + A_local.halo_x) * A_local.slice_dk_size +
                        (j + A_local.halo_y) * A_local.slice_dki_size]
                    = ((data_t *)buf)[k + lz * (i + lx * j)];// other
        MPI_File_close(&fh);
    }
    free(buf);
    // 矩阵需要填充halo区（目前仅此一次）
    update_halo();
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::write_data(const std::string pathname, const std::string prefix) {
    seq_structMatrix<idx_t, data_t, calc_t> & A_local = *local_matrix;
    assert( (sizeof(data_t) ==16 && comm_pkg->mpi_scalar_type == MPI_LONG_DOUBLE) || 
            (sizeof(data_t) == 4 && comm_pkg->mpi_scalar_type == MPI_FLOAT)  ||
            (sizeof(data_t) == 8 && comm_pkg->mpi_scalar_type == MPI_DOUBLE) ||
            (sizeof(data_t) == 2 && comm_pkg->mpi_scalar_type == MPI_SHORT )    );
    
    idx_t lx = A_local.local_x, ly = A_local.local_y, lz = A_local.local_z;
    idx_t tot_len = lx * ly * lz;
    if (strstr(pathname.c_str(), "LASER")) {
        assert(sizeof(data_t) == 8);
    } else if (strstr(pathname.c_str(), "GRAPES")) {
        assert(sizeof(data_t) == 4);
    }

    size_t bytes = sizeof(data_t) * tot_len;
    void * buf = malloc(bytes);// 读入缓冲区（给定的数据是双精度的）

    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = (sizeof(data_t) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Datatype read_type = MPI_DATATYPE_NULL;// 读取时的类型

    idx_t size[3], subsize[3], start[3];
    size[0] = this->input_dim[1];// y 方向的全局大小
    size[1] = this->input_dim[0];// x 方向的全局大小
    size[2] = this->input_dim[2];// z 方向的全局大小
    subsize[0] = ly;
    subsize[1] = lx;
    subsize[2] = lz;
    start[0] = offset_y;
    start[1] = offset_x;
    start[2] = offset_z;
    
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, etype, &read_type);
    MPI_Type_commit(&read_type);

    MPI_Offset displacement = 0;
    displacement *= sizeof(data_t);// 位移要以字节为单位！
    MPI_Status status;

    // 依次读入A的各条对角线
    for (idx_t idiag = 0; idiag < num_diag; idiag++) {
        const std::string filename = pathname + "/" + prefix + ".array_a." + std::to_string(idiag);
        for (idx_t j = 0; j < ly; j++)
        for (idx_t i = 0; i < lx; i++)
        for (idx_t k = 0; k < lz; k++)
            ((data_t *)buf)[k + lz * (i + lx * j)] = 
                A_local.data[idiag +
                        (k + A_local.halo_z) * A_local.num_diag + 
                        (i + A_local.halo_x) * A_local.slice_dk_size +
                        (j + A_local.halo_y) * A_local.slice_dki_size];
        MPI_File_open(comm_pkg->cart_comm, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        MPI_File_set_view(fh, displacement, etype, read_type, "native", MPI_INFO_NULL);
        MPI_File_write_all(fh, buf, tot_len, etype, &status);
        MPI_File_close(&fh);
    }
    free(buf);
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::set_val(data_t val, bool halo_set) {
    if (halo_set) {
        const idx_t tot_len = (local_matrix->local_x + local_matrix->halo_x * 2)
                            * (local_matrix->local_y + local_matrix->halo_y * 2)
                            * (local_matrix->local_z + local_matrix->halo_z * 2) * num_diag;
        for (idx_t p = 0; p < tot_len; p++)
            local_matrix->data[p] = 0.0;
    }
    else
        *(local_matrix) = val;
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::set_diag_val(idx_t d, data_t val) {
    local_matrix->set_diag_val(d, val);
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::init_random_base2(idx_t max_power)
{
    seq_structMatrix<idx_t, data_t, calc_t> & mat = *(local_matrix);
    const idx_t jbeg = mat.halo_y, jend = jbeg + mat.local_y,
                ibeg = mat.halo_x, iend = ibeg + mat.local_x,
                kbeg = mat.halo_z, kend = kbeg + mat.local_z;
    // idx_t num_options = 10;
    // data_t options[num_options] = {1.0, 0.5, 0.25, 0.};
    idx_t power;
    srand(time(0));
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++)
    for (idx_t d = 0; d < num_diag; d++)
    {
        idx_t loc = d + k * num_diag + i * mat.slice_dk_size + j * mat.slice_dki_size;
        power = rand() % max_power;
        idx_t denom = 0x1 << power;
        power = rand() % max_power;
        idx_t numer = 0x1 << power;
        mat.data[loc] = (data_t) numer / (data_t) denom;

        // mat.data[loc] = (data_t) rand() / (data_t) RAND_MAX;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::separate_Diags() const
{
    assert(DiagGroups_separated == false && DiagGroups == nullptr);

    DiagGroups_cnt = num_diag / NEON_MAX_STRIDE;// number of groups
    idx_t last_group = num_diag - NEON_MAX_STRIDE * DiagGroups_cnt;
    if (last_group > 0) DiagGroups_cnt ++;// additional group to contain remaining diags

    DiagGroups = new seq_structMatrix<idx_t, data_t, calc_t> * [DiagGroups_cnt];
    for (idx_t id = 0; id < DiagGroups_cnt; id++) {
        if (id == DiagGroups_cnt - 1 && last_group > 0)
            DiagGroups[id] = new seq_structMatrix<idx_t, data_t, calc_t>
                (last_group   , local_matrix->local_x, local_matrix->local_y, local_matrix->local_z,
                                local_matrix->halo_x , local_matrix->halo_y , local_matrix->halo_z);
        else
            DiagGroups[id] = new seq_structMatrix<idx_t, data_t, calc_t>
                (NEON_MAX_STRIDE,local_matrix->local_x, local_matrix->local_y, local_matrix->local_z,
                                local_matrix->halo_x , local_matrix->halo_y , local_matrix->halo_z);
    }

    const idx_t tot_elems = (local_matrix->local_y + local_matrix->halo_y * 2)
                        *   (local_matrix->local_x + local_matrix->halo_x * 2)
                        *   (local_matrix->local_z + local_matrix->halo_z * 2);
    #pragma omp parallel for schedule(static)
    for (idx_t e = 0; e < tot_elems; e++) {
        const data_t * aos_ptr = local_matrix->data + e * local_matrix->num_diag;
        data_t * soa_ptrs[DiagGroups_cnt];
        for (idx_t g = 0; g < DiagGroups_cnt; g++)
            soa_ptrs[g] = DiagGroups[g]->data + e * DiagGroups[g]->num_diag;

        for (idx_t d = 0; d < num_diag; d++) {
            // continuous
            idx_t gid = d / NEON_MAX_STRIDE;// group id
            idx_t lid = d % NEON_MAX_STRIDE;// land id

            // cyclic
            // idx_t gid = d % DiagGroups_cnt;// group id
            // idx_t lid = d / DiagGroups_cnt;// land id
            soa_ptrs[gid][lid] = aos_ptr[d];
        }
    }
    DiagGroups_separated = true;
    if (num_diag == 7) assert(DiagGroups_cnt == 2);
    else if (num_diag == 19) assert(DiagGroups_cnt == 5);
    else if (num_diag == 27) assert(DiagGroups_cnt == 7);
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::separate_truncate_Diags(const seq_structMatrix<idx_t, calc_t, calc_t> & A_high)
{
    assert(DiagGroups_separated == false && DiagGroups == nullptr);
    assert(A_high.num_diag == num_diag);
    CHECK_LOCAL_HALO(A_high, *local_matrix);
    DiagGroups_cnt = num_diag / NEON_MAX_STRIDE;// number of groups
    idx_t last_group = num_diag - NEON_MAX_STRIDE * DiagGroups_cnt;
    if (last_group > 0) DiagGroups_cnt ++;// additional group to contain remaining diags

    DiagGroups = new seq_structMatrix<idx_t, data_t, calc_t> * [DiagGroups_cnt];
    for (idx_t id = 0; id < DiagGroups_cnt; id++) {
        if (id == DiagGroups_cnt - 1 && last_group > 0)
            DiagGroups[id] = new seq_structMatrix<idx_t, data_t, calc_t>
                (last_group   , A_high.local_x, A_high.local_y, A_high.local_z,
                                A_high.halo_x , A_high.halo_y , A_high.halo_z);
        else
            DiagGroups[id] = new seq_structMatrix<idx_t, data_t, calc_t>
                (NEON_MAX_STRIDE,A_high.local_x, A_high.local_y, A_high.local_z,
                                A_high.halo_x , A_high.halo_y , A_high.halo_z);
    }

    const idx_t tot_elems = (A_high.local_y + A_high.halo_y * 2)
                        *   (A_high.local_x + A_high.halo_x * 2)
                        *   (A_high.local_z + A_high.halo_z * 2);
    #pragma omp parallel for schedule(static)
    for (idx_t e = 0; e < tot_elems; e++) {
        const calc_t * aos_ptr = A_high.data + e * num_diag;
        data_t * soa_ptrs[DiagGroups_cnt];
        for (idx_t g = 0; g < DiagGroups_cnt; g++)
            soa_ptrs[g] = DiagGroups[g]->data + e * DiagGroups[g]->num_diag;

        for (idx_t d = 0; d < num_diag; d++) {
            // continuous
            idx_t gid = d / NEON_MAX_STRIDE;// group id
            idx_t lid = d % NEON_MAX_STRIDE;// land id

            // cyclic
            // idx_t gid = d % DiagGroups_cnt;// group id
            // idx_t lid = d / DiagGroups_cnt;// land id
            soa_ptrs[gid][lid] = aos_ptr[d];
        }
    }
    DiagGroups_separated = true;
    if (num_diag == 7) assert(DiagGroups_cnt == 2);
    else if (num_diag == 19) assert(DiagGroups_cnt == 5);
    else if (num_diag == 27) assert(DiagGroups_cnt == 7);
}

template<typename idx_t, typename data_t, typename calc_t>
bool par_structMatrix<idx_t, data_t, calc_t>::check_Dirichlet()
{
    // 确定各维上是否是边界
    const bool x_lbdr = comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        const data_t * ptr = local_matrix->data + j * local_matrix->slice_dki_size + 
            i * local_matrix->slice_dk_size + k * local_matrix->num_diag;
        if (num_diag == 27) {
            if (i == ibeg   && x_lbdr) {
                assert(ptr[ 0] == 0.0); assert(ptr[ 1] == 0.0); assert(ptr[ 2] == 0.0);
                assert(ptr[ 9] == 0.0); assert(ptr[10] == 0.0); assert(ptr[11] == 0.0);
                assert(ptr[18] == 0.0); assert(ptr[19] == 0.0); assert(ptr[20] == 0.0);
            }
            if (i == iend-1 && x_ubdr) {
                assert(ptr[ 6] == 0.0); assert(ptr[ 7] == 0.0); assert(ptr[ 8] == 0.0);
                assert(ptr[15] == 0.0); assert(ptr[16] == 0.0); assert(ptr[17] == 0.0);
                assert(ptr[24] == 0.0); assert(ptr[25] == 0.0); assert(ptr[26] == 0.0);
            }
            if (j == jbeg   && y_lbdr) {
                assert(ptr[ 0] == 0.0); assert(ptr[ 1] == 0.0); assert(ptr[ 2] == 0.0);
                assert(ptr[ 3] == 0.0); assert(ptr[ 4] == 0.0); assert(ptr[ 5] == 0.0);
                assert(ptr[ 6] == 0.0); assert(ptr[ 7] == 0.0); assert(ptr[ 8] == 0.0);
            }
            if (j == jend-1 && y_ubdr) {
                assert(ptr[18] == 0.0); assert(ptr[19] == 0.0); assert(ptr[20] == 0.0);
                assert(ptr[21] == 0.0); assert(ptr[22] == 0.0); assert(ptr[23] == 0.0);
                assert(ptr[24] == 0.0); assert(ptr[25] == 0.0); assert(ptr[26] == 0.0);
            }
            if (k == kbeg   && z_lbdr) {
                assert(ptr[ 0] == 0.0); assert(ptr[ 3] == 0.0); assert(ptr[ 6] == 0.0);
                assert(ptr[ 9] == 0.0); assert(ptr[12] == 0.0); assert(ptr[15] == 0.0);
                assert(ptr[18] == 0.0); assert(ptr[21] == 0.0); assert(ptr[24] == 0.0);
            }
            if (k == kend-1 && z_ubdr) {
                assert(ptr[ 2] == 0.0); assert(ptr[ 5] == 0.0); assert(ptr[ 8] == 0.0);
                assert(ptr[11] == 0.0); assert(ptr[14] == 0.0); assert(ptr[17] == 0.0);
                assert(ptr[20] == 0.0); assert(ptr[23] == 0.0); assert(ptr[26] == 0.0);
            }
        }
        else if (num_diag == 19) {
            if (i == ibeg   && x_lbdr) {
                                        assert(ptr[ 0] == 0.0);
                assert(ptr[ 5] == 0.0); assert(ptr[ 6] == 0.0); assert(ptr[ 7] == 0.0);
                                        assert(ptr[14] == 0.0);
            }
            if (i == iend-1 && x_ubdr) {
                                        assert(ptr[ 4] == 0.0);
                assert(ptr[11] == 0.0); assert(ptr[12] == 0.0); assert(ptr[13] == 0.0);
                                        assert(ptr[18] == 0.0);
            }
            if (j == jbeg   && y_lbdr) {
                                        assert(ptr[ 0] == 0.0);
                assert(ptr[ 1] == 0.0); assert(ptr[ 2] == 0.0); assert(ptr[ 3] == 0.0);
                                        assert(ptr[ 4] == 0.0);
            }
            if (j == jend-1 && y_ubdr) {
                                        assert(ptr[14] == 0.0);
                assert(ptr[15] == 0.0); assert(ptr[16] == 0.0); assert(ptr[17] == 0.0);
                                        assert(ptr[18] == 0.0);
            }
            if (k == kbeg   && z_lbdr) {
                                        assert(ptr[ 1] == 0.0);
                assert(ptr[ 5] == 0.0); assert(ptr[ 8] == 0.0); assert(ptr[11] == 0.0);
                                        assert(ptr[15] == 0.0);
            }
            if (k == kend-1 && z_ubdr) {
                                        assert(ptr[ 3] == 0.0);
                assert(ptr[ 7] == 0.0); assert(ptr[10] == 0.0); assert(ptr[13] == 0.0);
                                        assert(ptr[17] == 0.0);
            }
        }
        else if (num_diag == 7) {
            if (i == ibeg   && x_lbdr) assert(ptr[1] == 0.0);
            if (i == iend-1 && x_ubdr) assert(ptr[5] == 0.0);
            if (j == jbeg   && y_lbdr) assert(ptr[0] == 0.0);
            if (j == jend-1 && y_ubdr) assert(ptr[6] == 0.0);
            if (k == kbeg   && z_lbdr) assert(ptr[2] == 0.0);
            if (k == kend-1 && z_ubdr) assert(ptr[4] == 0.0);
        } 
        else {
            assert(false);
        }
    }

    return true;
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::set_boundary()
{
    // 确定各维上是否是边界
    const bool x_lbdr = comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        data_t * ptr = local_matrix->data + j * local_matrix->slice_dki_size + 
            i * local_matrix->slice_dk_size + k * local_matrix->num_diag;
        if (num_diag == 27) {
            if (i == ibeg   && x_lbdr) {
                ptr[ 0] =  0.0; ptr[ 1] =  0.0; ptr[ 2] =  0.0;
                ptr[ 9] =  0.0; ptr[10] =  0.0; ptr[11] =  0.0;
                ptr[18] =  0.0; ptr[19] =  0.0; ptr[20] =  0.0;
            }
            if (i == iend-1 && x_ubdr) {
                ptr[ 6] =  0.0; ptr[ 7] =  0.0; ptr[ 8] =  0.0;
                ptr[15] =  0.0; ptr[16] =  0.0; ptr[17] =  0.0;
                ptr[24] =  0.0; ptr[25] =  0.0; ptr[26] =  0.0;
            }
            if (j == jbeg   && y_lbdr) {
                ptr[ 0] =  0.0; ptr[ 1] =  0.0; ptr[ 2] =  0.0;
                ptr[ 3] =  0.0; ptr[ 4] =  0.0; ptr[ 5] =  0.0;
                ptr[ 6] =  0.0; ptr[ 7] =  0.0; ptr[ 8] =  0.0;
            }
            if (j == jend-1 && y_ubdr) {
                ptr[18] =  0.0; ptr[19] =  0.0; ptr[20] =  0.0;
                ptr[21] =  0.0; ptr[22] =  0.0; ptr[23] =  0.0;
                ptr[24] =  0.0; ptr[25] =  0.0; ptr[26] =  0.0;
            }
            if (k == kbeg   && z_lbdr) {
                ptr[ 0] =  0.0; ptr[ 3] =  0.0; ptr[ 6] =  0.0;
                ptr[ 9] =  0.0; ptr[12] =  0.0; ptr[15] =  0.0;
                ptr[18] =  0.0; ptr[21] =  0.0; ptr[24] =  0.0;
            }
            if (k == kend-1 && z_ubdr) {
                ptr[ 2] =  0.0; ptr[ 5] =  0.0; ptr[ 8] =  0.0;
                ptr[11] =  0.0; ptr[14] =  0.0; ptr[17] =  0.0;
                ptr[20] =  0.0; ptr[23] =  0.0; ptr[26] =  0.0;
            }
        }
        else if (num_diag == 19) {
            if (i == ibeg   && x_lbdr) {
                                ptr[ 0] =  0.0;
                ptr[ 5] =  0.0; ptr[ 6] =  0.0; ptr[ 7] =  0.0;
                                ptr[14] =  0.0;
            }
            if (i == iend-1 && x_ubdr) {
                                ptr[ 4] =  0.0;
                ptr[11] =  0.0; ptr[12] =  0.0; ptr[13] =  0.0;
                                ptr[18] =  0.0;
            }
            if (j == jbeg   && y_lbdr) {
                                ptr[ 0] =  0.0;
                ptr[ 1] =  0.0; ptr[ 2] =  0.0; ptr[ 3] =  0.0;
                                ptr[ 4] =  0.0;
            }
            if (j == jend-1 && y_ubdr) {
                                ptr[14] =  0.0;
                ptr[15] =  0.0; ptr[16] =  0.0; ptr[17] =  0.0;
                                ptr[18] =  0.0;
            }
            if (k == kbeg   && z_lbdr) {
                                ptr[ 1] =  0.0;
                ptr[ 5] =  0.0; ptr[ 8] =  0.0; ptr[11] =  0.0;
                                ptr[15] =  0.0;
            }
            if (k == kend-1 && z_ubdr) {
                                ptr[ 3] =  0.0;
                ptr[ 7] =  0.0; ptr[10] =  0.0; ptr[13] =  0.0;
                                ptr[17] =  0.0;
            }
        }
        else if (num_diag == 7) {
            if (i == ibeg   && x_lbdr) ptr[1] =  0.0;
            if (i == iend-1 && x_ubdr) ptr[5] =  0.0;
            if (j == jbeg   && y_lbdr) ptr[0] =  0.0;
            if (j == jend-1 && y_ubdr) ptr[6] =  0.0;
            if (k == kbeg   && z_lbdr) ptr[2] =  0.0;
            if (k == kend-1 && z_ubdr) ptr[4] =  0.0;
        } 
        else {
            assert(false);
        }
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::scale(const data_t scaled_diag)
{
    assert(scaled == false);
    sqrt_D = new seq_structVector<idx_t, data_t>(
        local_matrix->local_x, local_matrix->local_y, local_matrix->local_z,
        local_matrix->halo_x , local_matrix->halo_y , local_matrix->halo_z );
    sqrt_D->set_halo(0.0);
    // 确定各维上是否是边界
    const bool x_lbdr = comm_pkg->ngbs_pid[I_L] == MPI_PROC_NULL, x_ubdr = comm_pkg->ngbs_pid[I_U] == MPI_PROC_NULL;
    const bool y_lbdr = comm_pkg->ngbs_pid[J_L] == MPI_PROC_NULL, y_ubdr = comm_pkg->ngbs_pid[J_U] == MPI_PROC_NULL;
    const bool z_lbdr = comm_pkg->ngbs_pid[K_L] == MPI_PROC_NULL, z_ubdr = comm_pkg->ngbs_pid[K_U] == MPI_PROC_NULL;

    idx_t jbeg = (y_lbdr ? sqrt_D->halo_y : 0), jend = sqrt_D->halo_y + sqrt_D->local_y + (y_ubdr ? 0 : sqrt_D->halo_y);
    idx_t ibeg = (x_lbdr ? sqrt_D->halo_x : 0), iend = sqrt_D->halo_x + sqrt_D->local_x + (x_ubdr ? 0 : sqrt_D->halo_x);
    idx_t kbeg = (z_lbdr ? sqrt_D->halo_z : 0), kend = sqrt_D->halo_z + sqrt_D->local_z + (z_ubdr ? 0 : sqrt_D->halo_z);

    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    if (my_pid == 0) printf("parMat scaled => diagonal as %.2e\n", scaled_diag);

    CHECK_LOCAL_HALO(*sqrt_D, *local_matrix);
    const idx_t diag_id = num_diag >> 1;// 7(3), 19(9), 27(13)
    const idx_t vec_ki_size = sqrt_D->slice_ki_size, vec_k_size = sqrt_D->slice_k_size;
    const idx_t slice_dki_size = local_matrix->slice_dki_size, slice_dk_size = local_matrix->slice_dk_size;
    // 提取对角线元素，开方
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        data_t tmp = local_matrix->data[j * slice_dki_size + i * slice_dk_size + k * num_diag + diag_id];
        assert(tmp > 0.0);
        tmp /= scaled_diag;
        sqrt_D->data[j * vec_ki_size + i * vec_k_size + k] = sqrt(tmp);
    }

    // 矩阵元素的scaling
    jbeg = local_matrix->halo_y; jend = jbeg + local_matrix->local_y;
    ibeg = local_matrix->halo_x; iend = ibeg + local_matrix->local_x;
    kbeg = local_matrix->halo_z; kend = kbeg + local_matrix->local_z;
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        data_t * mat_ptr = local_matrix->data + j * slice_dki_size + i * slice_dk_size + k * num_diag;
        for (idx_t d = 0; d < num_diag; d++) {
            data_t & tmp = mat_ptr[d];
            idx_t ngb_j = j + stencil[d * 3 + 0];
            idx_t ngb_i = i + stencil[d * 3 + 1];
            idx_t ngb_k = k + stencil[d * 3 + 2];
            if ((x_lbdr && ngb_i <  ibeg) || (x_ubdr && ngb_i >= iend) ||
                (y_lbdr && ngb_j <  jbeg) || (y_ubdr && ngb_j >= jend) ||
                (z_lbdr && ngb_k <  kbeg) || (z_ubdr && ngb_k >= kend)) {
                assert(tmp == 0.0);
            }
            else {
                data_t my_sqrt_Dval = sqrt_D->data[    j * vec_ki_size +     i * vec_k_size +     k];
                data_t ngb_sqrt_Dval= sqrt_D->data[ngb_j * vec_ki_size + ngb_i * vec_k_size + ngb_k];
                assert(my_sqrt_Dval > 0.0 && ngb_sqrt_Dval > 0.0);
                tmp /= (my_sqrt_Dval * ngb_sqrt_Dval);
            }
        }
    }
    update_halo();// 更新一下scaling之后的矩阵元素
    assert(check_Dirichlet());
    assert(check_scaling(scaled_diag));
    scaled = true;
}

template<typename idx_t, typename data_t, typename calc_t>
bool par_structMatrix<idx_t, data_t, calc_t>::check_scaling(const data_t scaled_diag)
{
    const idx_t ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    const idx_t id = num_diag >> 1;

    assert(id != -1);
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);

    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        const data_t * ptr = local_matrix->data + j * local_matrix->slice_dki_size + 
            i * local_matrix->slice_dk_size + k * local_matrix->num_diag;
        
        assert(abs(ptr[id] - scaled_diag) < 1e-4);
        // if (abs(ptr[id] - 1.0) >= 1e-4) {
        //     printf(" proc %d j %d i %d k %d val %.5e\n", my_pid, j, i, k, ptr[id]);
        // }
    }

    return true;

}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::write_CSR_bin(const std::string pathname) const {
    int num_proc; MPI_Comm_size(comm_pkg->cart_comm, &num_proc);
    assert(num_proc == 1);
    if (strstr(pathname.c_str(), "GRAPES")) assert(sizeof(data_t) == 4);
    else                                    assert(sizeof(data_t) == 8); 
    assert(sizeof(idx_t)  == 4);// int only
    assert(this->input_dim[0] == this->output_dim[0] && this->input_dim[1] == this->output_dim[1] && this->input_dim[2] == this->output_dim[2]);

    const idx_t gx = this->input_dim[0], gy = this->input_dim[1], gz = this->input_dim[2];
    const idx_t nrows = gx * gy * gz;

    std::vector<idx_t> row_ptr(nrows + 1, 0);
    std::vector<idx_t> col_idx;
    std::vector<data_t> vals;
    const idx_t jbeg = local_matrix->halo_y, jend = jbeg + local_matrix->local_y,
                ibeg = local_matrix->halo_x, iend = ibeg + local_matrix->local_x,
                kbeg = local_matrix->halo_z, kend = kbeg + local_matrix->local_z;
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        idx_t row_idx = ((j-jbeg)*gx + i-ibeg)*gz + k-kbeg; assert(row_idx < nrows);
        for (idx_t d = 0; d < num_diag; d++) {
            const idx_t ngb_j = j + this->stencil[d*3  ],
                        ngb_i = i + this->stencil[d*3+1],
                        ngb_k = k + this->stencil[d*3+2];
            if (ngb_j < jbeg || ngb_j >= jend ||
                ngb_i < ibeg || ngb_i >= iend ||
                ngb_k < kbeg || ngb_k >= kend   ) continue;
            idx_t ngb_row_idx = ((ngb_j-jbeg)*gx + ngb_i-ibeg)*gz + ngb_k-kbeg; assert(ngb_row_idx < nrows);
            data_t val = local_matrix->data[
                        j * local_matrix->slice_dki_size
                    +   i * local_matrix->slice_dk_size
                    +   k * local_matrix->num_diag
                    +   d];
            if (val != 0.0) {// 非零元
                row_ptr[row_idx+1] ++;
                col_idx.push_back(ngb_row_idx);
                vals.push_back(val);
            }
        }
    }
    
    for (idx_t i = 0; i < nrows; i++)
        row_ptr[i+1] += row_ptr[i];
    assert(row_ptr[0] == 0);
    assert(row_ptr[nrows] == col_idx.size());
    assert(row_ptr[nrows] == vals.size());

    std::string filename;
    filename = pathname + "/Ai.bin";
    printf("writing to %s...\n", filename.c_str());
    FILE * fp = fopen(filename.c_str(), "wb");
    size_t size = fwrite(row_ptr.data(), sizeof(idx_t), nrows + 1, fp); assert(size == row_ptr.size());
    fclose(fp);

    filename = pathname + "/Aj.bin";
    printf("writing to %s...\n", filename.c_str());
    fp = fopen(filename.c_str(), "wb");
    size = fwrite(col_idx.data(), sizeof(idx_t), row_ptr[nrows], fp); assert(size == col_idx.size());
    fclose(fp);

    filename = pathname + "/Av.bin";
    printf("writing to %s...\n", filename.c_str());
    fp = fopen(filename.c_str(), "wb");
    size = fwrite(vals.data()   , sizeof(data_t),row_ptr[nrows], fp); assert(size == vals.size());
    fclose(fp);
}

template<typename idx_t, typename data_t, typename calc_t>
void par_structMatrix<idx_t, data_t, calc_t>::write_struct_AOS_bin(const std::string pathname, const std::string file) const {
    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    seq_structMatrix<idx_t, data_t, calc_t> & mat = *local_matrix;

    idx_t lx = mat.local_x, ly = mat.local_y, lz = mat.local_z;
    idx_t tot_len = lx * ly * lz * num_diag;
    if (strstr(pathname.c_str(), "GRAPES")) assert(sizeof(data_t) == 4);
    else                                    assert(sizeof(data_t) == 8); 
    size_t bytes = sizeof(data_t) * tot_len;
    void * buf = malloc(bytes);
    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = (sizeof(data_t) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Datatype write_type = MPI_DATATYPE_NULL;// 写出类型

    idx_t size[4], subsize[4], start[4];
    size[0] = this->input_dim[1];// global_size_y;
    size[1] = this->input_dim[0];// global_size_x;
    size[2] = this->input_dim[2];// global_size_z;
    size[3] = num_diag;
    subsize[0] = ly;
    subsize[1] = lx;
    subsize[2] = lz;
    subsize[3] = num_diag;
    start[0] = offset_y;
    start[1] = offset_x;
    start[2] = offset_z;
    start[3] = 0;

    MPI_Type_create_subarray(4, size, subsize, start, MPI_ORDER_C, etype, &write_type);
    MPI_Type_commit(&write_type);

    MPI_Offset displacement = 0;
    displacement *= sizeof(data_t);// 位移要以字节为单位！
    MPI_Status status;

    // 拷入缓冲区
    for (idx_t j = 0; j < ly; j++)
    for (idx_t i = 0; i < lx; i++)
    for (idx_t k = 0; k < lz; k++)
    for (idx_t d = 0; d < num_diag; d++)
        ((data_t *)buf)[d + num_diag * (k + lz * (i + lx * j))]
            = mat.data[d + 
                (k + mat.halo_z) *      num_diag +
                (i + mat.halo_x) * mat.slice_dk_size +
                (j + mat.halo_y) * mat.slice_dki_size];

    const std::string filename = pathname + "/" + file;
    if (my_pid == 0) printf("writing to %s\n", filename.c_str());

    // 写出
    int ret;
    ret = MPI_File_open(comm_pkg->cart_comm, filename.c_str(),
                        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        printf("Could not open file: ret %d\n", ret);
    }
    MPI_File_set_view(fh, displacement, etype, write_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, buf, tot_len, etype, &status);
    
    MPI_File_close(&fh);
    MPI_Type_free(&write_type);

    free(buf);
}

#endif