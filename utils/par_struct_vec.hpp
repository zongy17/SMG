#ifndef SMG_PAR_STRUCT_VEC_HPP
#define SMG_PAR_STRUCT_VEC_HPP

#include "common.hpp"
#include "comm_pkg.hpp"
#include "seq_struct_mv.hpp"

template<typename idx_t, typename data_t>
struct irrgPts_vec {
    idx_t gid;// global idx
    data_t val;
};

template<typename idx_t, typename data_t>
class par_structVector {
public:
    idx_t global_size_x, global_size_y, global_size_z;// 全局方向的格点数
    idx_t offset_x     , offset_y     , offset_z     ;// 该向量在全局中的偏移
    seq_structVector<idx_t, data_t> * local_vector = nullptr;

    // 通信相关的
    // 真的需要在每个向量里都有一堆通信的东西吗？虽然方便了多重网格的搞法，但能不能多个同样规格的向量实例共用一套？
    // 可以参考hypre等的实现，某个向量own了这些通信的，然后可以在拷贝构造函数中“外借”出去，析构时由拥有者进行销毁
    StructCommPackage * comm_pkg = nullptr;
    bool own_comm_pkg = false;

    idx_t num_irrgPts = 0;// 非规则点的数目
    irrgPts_vec<idx_t, data_t> * irrgPts = nullptr;

    par_structVector(MPI_Comm comm, idx_t gx, idx_t gy, idx_t gz, idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z,
        bool need_corner);
    // 按照model的规格生成一个结构化向量，浅拷贝通信包
    par_structVector(const par_structVector & model);
    ~par_structVector();

    void setup_cart_comm(MPI_Comm comm, idx_t px, idx_t py, idx_t pz, bool unblk);
    void setup_comm_pkg(bool need_corner=false);

    void init_irrgPts(idx_t num, const idx_t * gids, const char * filename=nullptr);
    void update_halo() const;
    void set_halo(data_t val) const;
    void set_val(data_t val, bool halo_set=false);
    void write_CSR_bin(const std::string pathname, const std::string file) const;
    void read_data(const std::string pathname, const std::string file);
    void write_data(const std::string pathname, const std::string file);
};

/*
 * * * * * par_structVector * * * * *  
 */

template<typename idx_t, typename data_t>
par_structVector<idx_t, data_t>::par_structVector(MPI_Comm comm,
    idx_t gx, idx_t gy, idx_t gz, idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z, bool need_corner)
    : global_size_x(gx), global_size_y(gy), global_size_z(gz)
{
    // for GMG concern: must be fully divided by processors
    assert(global_size_x % num_proc_x == 0);
    assert(global_size_y % num_proc_y == 0);
    assert(global_size_z % num_proc_z == 0);

    bool unblk = need_corner ? false : true;
    setup_cart_comm(comm, num_proc_x, num_proc_y, num_proc_z, unblk);

    int (&cart_ids)[3] = comm_pkg->cart_ids;
    offset_y = cart_ids[0] * global_size_y / num_proc_y;
    offset_x = cart_ids[1] * global_size_x / num_proc_x;
    offset_z = cart_ids[2] * global_size_z / num_proc_z;

    // 建立本地数据的内存
    local_vector = new seq_structVector<idx_t, data_t>
        (global_size_x / num_proc_x, global_size_y / num_proc_y, global_size_z / num_proc_z, 1, 1, 1);
    
    setup_comm_pkg(need_corner);
}

template<typename idx_t, typename data_t>
par_structVector<idx_t, data_t>::par_structVector(const par_structVector & model)
    : global_size_x(model.global_size_x), global_size_y(model.global_size_y), global_size_z(model.global_size_z),
      offset_x(model.offset_x), offset_y(model.offset_y), offset_z(model.offset_z)
{
    local_vector = new seq_structVector<idx_t, data_t>(*(model.local_vector));
    // 浅拷贝
    comm_pkg = model.comm_pkg;
    own_comm_pkg = false;
    
    if (model.num_irrgPts > 0) {
        num_irrgPts = model.num_irrgPts;
        irrgPts = new irrgPts_vec<idx_t, data_t> [num_irrgPts];
        for (idx_t ir = 0; ir < num_irrgPts; ir++) {
            irrgPts[ir] = model.irrgPts[ir];
        }
    }
}

template<typename idx_t, typename data_t>
par_structVector<idx_t, data_t>::~par_structVector()
{
    delete local_vector;
    local_vector = nullptr;
    if (own_comm_pkg) {
        delete comm_pkg;
        comm_pkg = nullptr;
    }
    if (num_irrgPts > 0) {
        delete irrgPts; irrgPts = nullptr;
    }   
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::setup_cart_comm(MPI_Comm comm, 
    idx_t num_proc_x, idx_t num_proc_y, idx_t num_proc_z, bool unblk)
{
    bool relay_mode = unblk ? false : true;// 是否为接力传递数据的（二踢脚）通信方式
    comm_pkg = new StructCommPackage(relay_mode);
    own_comm_pkg = true;
    // 对comm_pkg内变量的引用，免得写太麻烦了
    MPI_Comm & cart_comm                         = comm_pkg->cart_comm;
    int (&cart_ids)[3]                           = comm_pkg->cart_ids;
    int (&ngbs_pid)[NUM_NEIGHBORS]               = comm_pkg->ngbs_pid;
    int & my_pid                                 = comm_pkg->my_pid;

    // create 3D distributed grid
    idx_t dims[3] = {num_proc_y, num_proc_x, num_proc_z};
    idx_t periods[3] = {0, 0, 0};

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

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::setup_comm_pkg(bool need_corner)
{
    MPI_Datatype (&send_subarray)[NUM_NEIGHBORS] = comm_pkg->send_subarray;
    MPI_Datatype (&recv_subarray)[NUM_NEIGHBORS] = comm_pkg->recv_subarray;
    MPI_Datatype & mpi_scalar_type               = comm_pkg->mpi_scalar_type;
    // 建立通信结构：注意data的排布从内到外依次为k(2)->i(1)->j(0)，按照C-order
    if      constexpr (sizeof(data_t) == 16)  mpi_scalar_type = MPI_LONG_DOUBLE;
    else if constexpr (sizeof(data_t) == 8)   mpi_scalar_type = MPI_DOUBLE;
    else if constexpr (sizeof(data_t) == 4)   mpi_scalar_type = MPI_FLOAT;
    else if constexpr (sizeof(data_t) == 2)   mpi_scalar_type = MPI_SHORT;
    else { printf("INVALID data_t when creating subarray, sizeof %ld bytes\n", sizeof(data_t)); MPI_Abort(MPI_COMM_WORLD, -2001); }
    
    idx_t size[3] = {   local_vector->local_y + 2 * local_vector->halo_y,
                        local_vector->local_x + 2 * local_vector->halo_x,
                        local_vector->local_z + 2 * local_vector->halo_z  };
    idx_t subsize[3], send_start[3], recv_start[3];
    for (idx_t ingb = 0; ingb < NUM_NEIGHBORS; ingb++) {
        switch (ingb)
        {
        // 最先传的
        case K_L:
        case K_U:
            subsize[0] = local_vector->local_y;
            subsize[1] = local_vector->local_x;
            subsize[2] = local_vector->halo_z;
            break;
        case I_L:
        case I_U:
            subsize[0] = local_vector->local_y;
            subsize[1] = local_vector->halo_x;
            subsize[2] = local_vector->local_z + (need_corner ? 2 * local_vector->halo_z : 0);
            break;
        case J_L:
        case J_U:
            subsize[0] = local_vector->halo_y;
            subsize[1] = local_vector->local_x + (need_corner ? 2 * local_vector->halo_x : 0);
            subsize[2] = local_vector->local_z + (need_corner ? 2 * local_vector->halo_z : 0);
            break;
        default:
            printf("INVALID NEIGHBOR ID of %d!\n", ingb);
            MPI_Abort(MPI_COMM_WORLD, -2000);
        }
        
        switch (ingb)
        {
        case K_L:// 向K下发的内halo
            send_start[0] = recv_start[0] = local_vector->halo_y;
            send_start[1] = recv_start[1] = local_vector->halo_x;
            send_start[2] = local_vector->halo_z;           recv_start[2] = 0;
            break;
        case K_U:
            send_start[0] = recv_start[0] = local_vector->halo_y;
            send_start[1] = recv_start[1] = local_vector->halo_x;
            send_start[2] = local_vector->local_z;          recv_start[2] = local_vector->local_z + local_vector->halo_z;
            break;
        case I_L:
            send_start[0] = recv_start[0] = local_vector->halo_y;
            send_start[1] = local_vector->halo_x;           recv_start[1] = 0;
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_vector->halo_z);
            break;
        case I_U:
            send_start[0] = recv_start[0] = local_vector->halo_y;
            send_start[1] = local_vector->local_x;          recv_start[1] = local_vector->local_x + local_vector->halo_x;
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_vector->halo_z);
            break;
        case J_L:
            send_start[0] = local_vector->halo_y;           recv_start[0] = 0;
            send_start[1] = recv_start[1] = (need_corner ? 0 : local_vector->halo_x);
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_vector->halo_z);
            break;
        case J_U:
            send_start[0] = local_vector->local_y;          recv_start[0] = local_vector->local_y + local_vector->halo_y;
            send_start[1] = recv_start[1] = (need_corner ? 0 : local_vector->halo_x);
            send_start[2] = recv_start[2] = (need_corner ? 0 : local_vector->halo_z);
            break;
        default:
            printf("INVALID NEIGHBOR ID of %d!\n", ingb);
            MPI_Abort(MPI_COMM_WORLD, -2000);
        }

        MPI_Type_create_subarray(3, size, subsize, send_start, MPI_ORDER_C, mpi_scalar_type, &send_subarray[ingb]);
        MPI_Type_commit(&send_subarray[ingb]);
        MPI_Type_create_subarray(3, size, subsize, recv_start, MPI_ORDER_C, mpi_scalar_type, &recv_subarray[ingb]);
        MPI_Type_commit(&recv_subarray[ingb]);
    }
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::init_irrgPts(idx_t num, const idx_t * gids, const char* filename)
{
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);

    num_irrgPts = num;
    if (num_irrgPts > 0) {
        if (irrgPts != nullptr) delete irrgPts;

        irrgPts = new irrgPts_vec<idx_t, data_t> [num_irrgPts];
        for (idx_t ir = 0; ir < num_irrgPts; ir++) {
            irrgPts[ir].gid = gids[ir];
            irrgPts[ir].val = 0.0;// init as 0
        }
        
        if (filename != nullptr) {
            FILE * fp = fopen(filename, "r");
            idx_t id; double v;
            
            idx_t read_cnt = 0;
            while (fscanf(fp, "%d %lf", &id, &v) != EOF) {
                for (idx_t j = 0; j < num_irrgPts; j++) {
                    if (irrgPts[j].gid == id) {
                        irrgPts[j].val = v;
                        read_cnt ++;
                    }
                }
            }
            fclose(fp);
            assert(read_cnt == num_irrgPts);
        }
    }
#ifdef DEBUG
    for (idx_t i = 0; i < num_irrgPts; i++)
        printf(" proc %d pt%d idx %d val %.10e\n", my_pid, i, irrgPts[i].gid, irrgPts[i].val);
#endif
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::update_halo() const
{
#ifdef DEBUG
    local_vector->init_debug(offset_x, offset_y, offset_z);
    if (my_pid == 1) {
        local_vector->print_level(1);
    }
#endif

    comm_pkg->exec_comm(local_vector->data);

#ifdef DEBUG
    if (my_pid == 1) {
        local_vector->print_level(1);
    }
#endif
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::set_halo(data_t val) const
{
    local_vector->set_halo(val);
}

// val -> v[...]
template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::set_val(const data_t val, bool halo_set) {
    if (halo_set) {
        seq_structVector<idx_t, data_t> & vec = *local_vector;
        idx_t tot_len = (vec.local_x + vec.halo_x * 2) * (vec.local_y + vec.halo_y * 2) * (vec.local_z + vec.halo_z * 2);
        #pragma omp parallel for schedule(static) 
        for (idx_t i = 0; i < tot_len; i++)
            vec.data[i] = val;
    }
    else {// 只将内部的值设置 
        *(local_vector) = val;
    }

    // 非规则点
    for (idx_t ir = 0; ir < num_irrgPts; ir++)
        irrgPts[ir].val = val;
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::write_CSR_bin(const std::string pathname, const std::string file) const {
    int num_proc; MPI_Comm_size(comm_pkg->cart_comm, &num_proc);
    assert(num_proc == 1);
    if (strstr(pathname.c_str(), "GRAPES")) assert(sizeof(data_t) == 4);
    else                                    assert(sizeof(data_t) == 8); 
    assert(sizeof(idx_t)  == 4);// int only

    const idx_t & gx = global_size_x, & gy = global_size_y, & gz = global_size_z;
    const idx_t nrows = gx * gy * gz;
    size_t bytes = nrows * sizeof(data_t);
    void * vec_vals = malloc(bytes);

    const idx_t jbeg = local_vector->halo_y, jend = jbeg + local_vector->local_y,
                ibeg = local_vector->halo_x, iend = ibeg + local_vector->local_x,
                kbeg = local_vector->halo_z, kend = kbeg + local_vector->local_z;
    #pragma omp parallel for collapse(3) schedule(static)
    for (idx_t j = jbeg; j < jend; j++)
    for (idx_t i = ibeg; i < iend; i++)
    for (idx_t k = kbeg; k < kend; k++) {
        idx_t row_idx = ((j-jbeg)*gx + i-ibeg)*gz + k-kbeg;
        assert(row_idx < nrows);
        ((data_t *)vec_vals)[row_idx] 
            = local_vector->data[j * local_vector->slice_ki_size + i * local_vector->slice_k_size + k];
    }
    const std::string filename = pathname + "/" + file;
    printf("writing to %s\n", filename.c_str());
    FILE * fp = fopen(filename.c_str(), "wb");
    int size = fwrite(vec_vals, sizeof(data_t), nrows, fp); assert(size == nrows);
    fclose(fp);
    free(vec_vals);
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::read_data(const std::string pathname, const std::string file)
{
    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    seq_structVector<idx_t, data_t> & vec = *local_vector;

    idx_t lx = vec.local_x, ly = vec.local_y, lz = vec.local_z;
    idx_t tot_len = lx * ly * lz;
    if (strstr(pathname.c_str(), "GRAPES")) assert(sizeof(data_t) == 4);
    else                                    assert(sizeof(data_t) == 8); 

    size_t bytes = sizeof(data_t) * tot_len;
    void * buf = malloc(bytes);// 读入缓冲区（给定的数据是双精度的）
    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = (sizeof(data_t) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Datatype read_type = MPI_DATATYPE_NULL;// 读取时的类型

    idx_t size[3], subsize[3], start[3];
    size[0] = global_size_y;
    size[1] = global_size_x;
    size[2] = global_size_z;
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

    const std::string filename = pathname + "/" + file;
    if (my_pid == 0) printf("reading from %s\n", filename.c_str());

    // 读入右端向量
    int ret;
    ret = MPI_File_open(comm_pkg->cart_comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        printf("Could not open file: ret %d\n", ret);
    }
    MPI_File_set_view(fh, displacement, etype, read_type, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, buf, tot_len, etype, &status);

    for (idx_t j = 0; j < ly; j++)
    for (idx_t i = 0; i < lx; i++)
    for (idx_t k = 0; k < lz; k++)
        vec.data[k + vec.halo_z +
                (i + vec.halo_x) * vec.slice_k_size +
                (j + vec.halo_y) * vec.slice_ki_size]
                = ((data_t *) buf)[k + lz * (i + lx * j)];// other

    
    MPI_File_close(&fh);
    MPI_Type_free(&read_type);

    free(buf);
}

template<typename idx_t, typename data_t>
void par_structVector<idx_t, data_t>::write_data(const std::string pathname, const std::string file)
{
    int my_pid; MPI_Comm_rank(comm_pkg->cart_comm, &my_pid);
    seq_structVector<idx_t, data_t> & vec = *local_vector;

    idx_t lx = vec.local_x, ly = vec.local_y, lz = vec.local_z;
    idx_t tot_len = lx * ly * lz;
    if (strstr(pathname.c_str(), "GRAPES")) assert(sizeof(data_t) == 4);
    else                                    assert(sizeof(data_t) == 8); 
    size_t bytes = sizeof(data_t) * tot_len;
    void * buf = malloc(bytes);
    MPI_File fh = MPI_FILE_NULL;// 文件句柄
    MPI_Datatype etype = (sizeof(data_t) == 4) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Datatype write_type = MPI_DATATYPE_NULL;// 写出类型

    idx_t size[3], subsize[3], start[3];
    size[0] = global_size_y;
    size[1] = global_size_x;
    size[2] = global_size_z;
    subsize[0] = ly;
    subsize[1] = lx;
    subsize[2] = lz;
    start[0] = offset_y;
    start[1] = offset_x;
    start[2] = offset_z;

    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, etype, &write_type);
    MPI_Type_commit(&write_type);

    MPI_Offset displacement = 0;
    displacement *= sizeof(data_t);// 位移要以字节为单位！
    MPI_Status status;

    const std::string filename = pathname + "/" + file;
    if (my_pid == 0) printf("writing to %s\n", filename.c_str());
    // 拷入缓冲区
    
    for (idx_t j = 0; j < ly; j++)
    for (idx_t i = 0; i < lx; i++)
    for (idx_t k = 0; k < lz; k++)
        ((data_t *) buf)[k + lz * (i + lx * j)]
            = vec.data[k + vec.halo_z +
                (i + vec.halo_x) * vec.slice_k_size +
                (j + vec.halo_y) * vec.slice_ki_size];
    
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

/*
 * * * * * Vector Ops * * * * *  
 */
// TODO: 向量点积也可以放在iterative solver里实现
template<typename idx_t, typename data_t, typename res_t>
res_t vec_dot(par_structVector<idx_t, data_t> const & x, par_structVector<idx_t, data_t> const & y)
{   
    CHECK_VEC_GLOBAL(x, y);
    CHECK_OFFSET(x, y);
    assert(sizeof(res_t) == 8);

    res_t loc_prod, glb_prod;
    loc_prod = seq_vec_dot<idx_t, data_t, res_t>(*(x.local_vector), *(y.local_vector));

    // 累计上本进程内部的非规则点
    assert(x.num_irrgPts == y.num_irrgPts);
    for (idx_t ir = 0; ir < x.num_irrgPts; ir++) {
        assert(x.irrgPts[ir].gid == y.irrgPts[ir].gid);
        loc_prod += x.irrgPts[ir].val * y.irrgPts[ir].val;
    }

#ifdef DEBUG
    printf("proc %3d halo_x/y/z %d %d %d local_x/y/z %d %d %d loc_proc %10.7e\n", 
        x.comm_pkg->my_pid,
        x.local_vector->halo_x, x.local_vector->halo_y, x.local_vector->halo_z,
        x.local_vector->local_x, x.local_vector->local_y, x.local_vector->local_z, loc_prod);
#endif

    // 这里不能用x.comm_pkg->mpi_scalar_type，因为点积希望向上保留精度
    if (sizeof(res_t) == 8)
        MPI_Allreduce(&loc_prod, &glb_prod, 1, MPI_DOUBLE, MPI_SUM, x.comm_pkg->cart_comm);
    else 
        MPI_Allreduce(&loc_prod, &glb_prod, 1, MPI_FLOAT , MPI_SUM, x.comm_pkg->cart_comm);
    
    return glb_prod;
}

// v1 + alpha * v2 -> v
template<typename idx_t, typename data_t, typename scalar_t>
void vec_add(const par_structVector<idx_t, data_t> & v1, scalar_t alpha, 
             const par_structVector<idx_t, data_t> & v2, par_structVector<idx_t, data_t> & v)
{
    CHECK_VEC_GLOBAL(v1, v2);

    seq_vec_add(*(v1.local_vector), alpha, *(v2.local_vector), *(v.local_vector));    

    // 非规则点
    assert(v1.num_irrgPts == v2.num_irrgPts && v1.num_irrgPts == v.num_irrgPts);
    for (idx_t ir = 0; ir < v1.num_irrgPts; ir++) {
        assert(v1.irrgPts[ir].gid == v2.irrgPts[ir].gid && v1.irrgPts[ir].gid == v.irrgPts[ir].gid);
        v.irrgPts[ir].val = v1.irrgPts[ir].val + alpha * v2.irrgPts[ir].val;
    }
}



// src -> dst
template<typename idx_t, typename data_t>
void vec_copy(const par_structVector<idx_t, data_t> & src, par_structVector<idx_t, data_t> & dst) {
    CHECK_VEC_GLOBAL(src, dst);

    seq_vec_copy(*(src.local_vector), *(dst.local_vector));

    // 非规则点
    assert(src.num_irrgPts == dst.num_irrgPts);
    for (idx_t ir = 0; ir < src.num_irrgPts; ir++) {
        assert(src.irrgPts[ir].gid == dst.irrgPts[ir].gid);
        dst.irrgPts[ir].val = src.irrgPts[ir].val;
    }
}

// coeff * src -> dst
template<typename idx_t, typename data_t, typename scalar_t>
void vec_mul_by_scalar(const scalar_t coeff, const par_structVector<idx_t, data_t> & src, par_structVector<idx_t, data_t> & dst) {
    CHECK_VEC_GLOBAL(src, dst);

    seq_vec_mul_by_scalar(coeff, *(src.local_vector), *(dst.local_vector));

    // 非规则点
    assert(src.num_irrgPts == dst.num_irrgPts);
    for (idx_t ir = 0; ir < src.num_irrgPts; ir++) {
        assert(src.irrgPts[ir].gid == dst.irrgPts[ir].gid);
        dst.irrgPts[ir].val = coeff * src.irrgPts[ir].val;
    }
}

// vec *= coeff
template<typename idx_t, typename data_t, typename scalar_t>
void vec_scale(const scalar_t coeff, par_structVector<idx_t, data_t> & vec) {
    seq_vec_scale(coeff, *(vec.local_vector));

    // 非规则点
    for (idx_t ir = 0; ir < vec.num_irrgPts; ir++)
        vec.irrgPts[ir].val *= coeff;
}

#endif