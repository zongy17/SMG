#ifndef SMG_COMMON_HPP
#define SMG_COMMON_HPP

#ifndef KSP_BIT
#define KSP_BIT 32
#endif
#ifndef PC_DATA_BIT
#define PC_DATA_BIT 16
#endif
#ifndef PC_CALC_BIT
#define PC_CALC_BIT 32
#endif

#if KSP_BIT==80
#define KSP_TYPE long double
#define KSP_MPI_TYPE MPI_LONG_DOUBLE
#elif KSP_BIT==64
#define KSP_TYPE double
#define KSP_MPI_TYPE MPI_DOUBLE
#elif KSP_BIT==32
#define KSP_TYPE float
#define KSP_MPI_TYPE MPI_FLOAT
#endif

#if PC_DATA_BIT==80
#define PC_DATA_TYPE long double 
#elif PC_DATA_BIT==64
#define PC_DATA_TYPE double 
#elif PC_DATA_BIT==32
#define PC_DATA_TYPE float
#elif PC_DATA_BIT==16
#define PC_DATA_TYPE __fp16
#endif

#if PC_CALC_BIT==80
#define PC_CALC_TYPE long double 
#elif PC_CALC_BIT==64
#define PC_CALC_TYPE double 
#elif PC_CALC_BIT==32
#define PC_CALC_TYPE float
#elif PC_CALC_BIT==16
#define PC_CALC_TYPE __fp16
#endif

#define IDX_TYPE int

#define NEON_MAX_STRIDE 4
// #define NDEBUG

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <string.h>
#include <string>
#include <vector>
#include <unordered_map>

enum NEIGHBOR_ID {I_L, I_U, J_L, J_U, K_L, K_U, NUM_NEIGHBORS};
typedef enum {VERT} LINE_DIRECTION;
typedef enum {XZ} PLANE_DIRECTION;

#define MIN(a, b) ((a) > (b)) ? (b) : (a)

template<typename T>
bool check_power2(T num) {
    while (num > 1) {
        T new_num = num >> 1;
        if (num != new_num * 2) return false;
        num = new_num;
    }
    return true;
}

typedef enum {HALO, ALL, SPMV, SPTRSV, VEC, NUM_PROFILE} PROFILE_ITEMS;
double profile[NUM_PROFILE];

#include <limits.h>
#include <unistd.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <libgen.h>

__thread char _affinity_buffer[256];
const char *get_affinity() {
    int ncores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if ((int)ncores > (int)sizeof(_affinity_buffer))
        return NULL;

    cpu_set_t cpu_set;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set) == -1)
        return NULL;

    for (int i = 0; i < ncores; i++) {
        _affinity_buffer[i] = CPU_ISSET(i, &cpu_set) ? '1' : '0';
}
    _affinity_buffer[ncores] = '\0';
    return _affinity_buffer;
}

void print_affinity(FILE *fd)
{
    int myrank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    for (int rank = 0; rank < nranks; rank++) {
#ifdef _OPENMP
        #pragma omp parallel
#endif
        if (myrank == rank) {
#ifdef _OPENMP
            int nthreads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();
            #pragma omp for ordered schedule(static,1)
#else
            int nthreads = 1;
            int thread_id = 0;
#endif
            for (int t = 0; t < nthreads; t++) {
#ifdef _OPENMP
                #pragma omp ordered
#endif
                {
                    int rank_digits = trunc(log10(nranks)) + 1;
                    int thread_digits = trunc(log10(nthreads)) + 1;
                    rank_digits = rank_digits > 3 ? rank_digits : 3;
                    thread_digits = thread_digits > 1 ? thread_digits : 1;
                    const char *affinity = get_affinity();
                    fprintf(fd, "rank=%0*d:%0*d affinity=%s\n", rank_digits, myrank, thread_digits, thread_id, affinity);
                    fflush(fd);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void stick_affinity()
{
    int myrank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int thread_id = omp_get_thread_num();
#else
        int thread_id = 0;
#endif
        const char *affinity = get_affinity();
        int len = strnlen(affinity, INT_MAX);
        int count = 0;
        for (int i = 0; i < len; i++) {
            if (affinity[i] == '0')
                continue;
            if (count == thread_id) {
                cpu_set_t cpu_set;
                CPU_ZERO(&cpu_set);
                CPU_SET(i, &cpu_set);
                sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
                break;
            }
            count += 1;
        }
    }
}

const IDX_TYPE stencil_offset_3d27[27 * 3] = {
    // y , x , z
    -1, -1, -1,
    -1, -1,  0,
    -1, -1,  1,
    -1,  0, -1,
    -1,  0,  0,
    -1,  0,  1,
    -1,  1, -1,
    -1,  1,  0,
    -1,  1,  1,

    0, -1, -1,
    0, -1,  0,
    0, -1,  1,
    0,  0, -1,
    0,  0,  0,
    0,  0,  1,
    0,  1, -1,
    0,  1,  0,
    0,  1,  1,

    1, -1, -1,
    1, -1,  0,
    1, -1,  1,
    1,  0, -1,
    1,  0,  0,
    1,  0,  1,
    1,  1, -1,
    1,  1,  0,
    1,  1,  1
};

const IDX_TYPE map_storage_to_memory_diag[19] = {
 // 0  1   2  3   4   5  6  7  8   9 10  11 12  13  14 15  16 17  18 
    9, 6, 12, 2, 16, 18, 4, 0, 14, 8, 5, 11, 1, 15, 10, 7, 13, 3, 17
};
const IDX_TYPE stencil_offset_3d19[19 * 3] = {
    // y , x , z 从外到内维
    -1, -1,  0,// 0
    -1,  0, -1,// 1
    -1,  0,  0,// 2
    -1,  0,  1,// 3
    -1,  1,  0,// 4

    0, -1, -1,// 5
    0, -1,  0,// 6
    0, -1,  1,// 7
    0,  0, -1,// 8
    0,  0,  0,// 9
    0,  0,  1,// 10
    0,  1, -1,// 11
    0,  1,  0,// 12
    0,  1,  1,// 13

    1, -1,  0,// 14
    1,  0, -1,
    1,  0,  0,
    1,  0,  1,
    1,  1,  0 // 18
};

const IDX_TYPE stencil_offset_3d15[15 * 3] = {
    -1,  0, -1,
    -1,  0,  0,
    -1,  0,  1,

    0, -1, -1,
    0, -1,  0,
    0, -1,  1,
    0,  0, -1,
    0,  0,  0,
    0,  0,  1,
    0,  1, -1,
    0,  1,  0,
    0,  1,  1,

    1,  0, -1,
    1,  0,  0,
    1,  0,  1,
};

const IDX_TYPE stencil_offset_3d7[7 * 3] = {
    // y , x , z
    -1,  0,  0, // 0
    0, -1,  0, // 1
    0,  0, -1, // 2
    0,  0,  0, // 3
    0,  0,  1, // 4
    0,  1,  0, // 5
    1,  0,  0  // 6
};

const IDX_TYPE stencil_offset_2d9[9 * 2] = {
    // x, z
    -1, -1, // 0     
    -1,  0, // 1            2---5---8
    -1,  1, // 2            |       |
    0 , -1, // 3        z   1   4   7
    0 ,  0, // 4        ^   |       |
    0 ,  1, // 5        |   0---3---6
    1 , -1, // 6        O ---> x 
    1 ,  0, // 7        数据排布z内维先变
    1 ,  1  // 8
};

#define CHECK_HALO(x , y) \
    assert((x).halo_x == (y).halo_x  &&  (x).halo_y == (y).halo_y  &&  (x).halo_z == (y).halo_z);

#define CHECK_LOCAL_HALO(x , y) \
    assert((x).local_x == (y).local_x && (x).local_y == (y).local_y && (x).local_z == (y).local_z && \
           (x).halo_x == (y).halo_x  &&  (x).halo_y == (y).halo_y  &&  (x).halo_z == (y).halo_z);

#define CHECK_INOUT_DIM(x , y) \
    assert( (x).input_dim[0] == (y).input_dim[0] && (x).input_dim[1] == (y).input_dim[1] && \
            (x).input_dim[2] == (y).input_dim[2] && \
            (x).output_dim[0] == (y).output_dim[0] && (x).output_dim[1] == (y).output_dim[1] && \
            (x).output_dim[2] == (y).output_dim[2] );

#define CHECK_OFFSET(x , y) \
    assert((x).offset_x == (y).offset_x && (x).offset_y == (y).offset_y && (x).offset_z == (y).offset_z);

#define CHECK_INPUT_DIM(x , y) \
    assert( (x).input_dim[0] == (y).global_size_x && (x).input_dim[1] == (y).global_size_y && \
            (x).input_dim[2] == (y).global_size_z );

#define CHECK_OUTPUT_DIM(x , y) \
    assert( (x).output_dim[0] == (y).global_size_x && (x).output_dim[1] == (y).global_size_y && \
            (x).output_dim[2] == (y).global_size_z );

#define CHECK_VEC_GLOBAL(x, y) \
    assert( (x).global_size_x == (y).global_size_x && (x).global_size_y == (y).global_size_y && \
            (x).global_size_z == (y).global_size_z );

double wall_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1. * t.tv_sec + 1.e-9 * t.tv_nsec;
}

#if defined(__aarch64__)
#define barrier() __asm__ __volatile__("dmb" ::: "memory")
#define smp_mb()  __asm__ __volatile__("dmb ish" ::: "memory")
#define smp_wmb() __asm__ __volatile__("dmb ishst" ::: "memory")
#define smp_rmb() __asm__ __volatile__("dmb ishld" ::: "memory")
#else
// #error No architecture detected.
#endif

#include <unistd.h>
#include <sched.h>
#include <pthread.h>

void print_affinity() {
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);// 这个节点上on_line的有多少个cpu核
    printf("affinity_cpu=%02d of %ld ", sched_getcpu(), nproc);// sched_getcpu()返回这个线程绑定的核心id
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity error");
        exit(1);
    }
    printf("affinity_mask=");
    for (int i = 0; i < nproc; i++) printf("%d", CPU_ISSET(i, &mask));
    printf("\n");
}

template<typename idx_t>
void uniformly_distributed_integer(idx_t M, idx_t N, idx_t * arr)
{// assert arr's space allocated of N
    assert(M >= N);
    idx_t gap = M / N;
    idx_t rem = M - gap * N;

    arr[0] = 0;
    for (idx_t i = 0; i < N - 1; i++)
        arr[i + 1] = arr[i] + gap + ((i<rem) ? 1 : 0);    
}

#endif