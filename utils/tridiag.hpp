#ifndef SMG_TRIDIAG_HPP
#define SMG_TRIDIAG_HPP

#include "seq_struct_mv.hpp"

/* Fortran style (1-based):
       _                                                _   _      _     _      _
      |  b(1)  c(1)                                      | | x(1  ) |   | d(1  ) |
      |  a(2)  b(2)  c(2)                                | | x(2  ) |   | d(2  ) |
      |        a(3)  b(3)  c(3)                          | | x(3  ) |   | d(3  ) |
      |           ...  ...  ...                          | | ...    | = | ...    |
      |                ...  ...  ...                     | | ...    |   | ...    |
      |                          a(n-1)  b(n-1)  c(n-1)  | | x(n-1) |   | d(n-1) |
      |_                                 a(n  )  b(n  ) _| |_x(n  )_|   |_d(n  )_|
    */

template<typename idx_t, typename data_t, typename oper_t>
class TridiagSolver {
protected:
    idx_t n_size;
    data_t * a = nullptr, * b = nullptr, * c = nullptr;
public:
    bool periodic = false;
    TridiagSolver(bool is_periodic = false) : periodic(is_periodic) { }
    // 执行分解过程
    virtual void Setup(const idx_t n_size, oper_t * a_buf, oper_t * b_buf, oper_t * c_buf);
    void truncate() {
#ifdef __aarch64__
        for (idx_t i = 0; i < n_size; i++) {
            __fp16 tmp;
            tmp = (__fp16) a[i]; a[i] = (data_t) tmp;
            tmp = (__fp16) b[i]; b[i] = (data_t) tmp;
            tmp = (__fp16) c[i]; c[i] = (data_t) tmp;
        }
#else
        printf("architecture not support truncated to fp16\n");
#endif
    }
    void Solve(oper_t * rhs, oper_t * sol);
    void Solve_neon_prft(oper_t * rhs, oper_t * sol);
    data_t * Get_a() {return a;}
    data_t * Get_b() {return b;}
    data_t * Get_c() {return c;}
    idx_t Get_n_size() {return n_size;}
    virtual ~TridiagSolver();
};

template<typename idx_t, typename data_t, typename oper_t>
TridiagSolver<idx_t, data_t, oper_t>::~TridiagSolver() {
    // if (a) {
    //     delete a; a = nullptr;
    // }
    // if (b) {
    //     delete b; b = nullptr;
    // }
    // if (c) {
    //     delete c; c = nullptr;
    // }
    if (a || b || c) {
        assert(a);
        delete a; a = nullptr;
        b = c = nullptr;
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void TridiagSolver<idx_t, data_t, oper_t>::Setup(const idx_t len, oper_t * a_buf, oper_t * b_buf, oper_t * c_buf)
{
    n_size = len;
    assert(a_buf[   0      ] == 0.0);
    assert(c_buf[n_size - 1] == 0.0);
    // 系数分解
    c_buf[0] /= b_buf[0];
    for (idx_t i = 1; i < n_size; i++) {
        b_buf[i] = b_buf[i] - a_buf[i] * c_buf[i - 1];
        c_buf[i] = c_buf[i] / b_buf[i];
    }

    assert(a == nullptr && b == nullptr && c == nullptr);// 避免realloc
    a = new data_t [n_size * 3];
    b = a +  n_size;
    c = a + (n_size << 1);
    // 拷贝
    for (idx_t i = 0; i < n_size; i++) {
        a[i] = a_buf[i];
        b[i] = b_buf[i];
        c[i] = c_buf[i];
    }
}

template<typename idx_t, typename data_t, typename oper_t>
void TridiagSolver<idx_t, data_t, oper_t>::Solve(oper_t * rhs, oper_t * sol) {
    // 前代
    rhs[0] /= b[0];
    for (idx_t i = 1; i < n_size; i++) 
        rhs[i] = (rhs[i] - a[i] * rhs[i - 1]) / b[i];
    
    // 回代
    sol[n_size - 1] = rhs[n_size - 1];
    for (idx_t i = n_size - 2; i >= 0; i--)
        sol[i] = rhs[i] - c[i] * sol[i + 1];
}


#ifdef __aarch64__
#define NEON_LEN 4

template<typename idx_t, typename data_t, typename oper_t>
void TridiagSolver<idx_t, data_t, oper_t>::Solve_neon_prft(oper_t * __restrict__ rhs, oper_t * __restrict__ sol)
{
    // for (idx_t i = 0; i < n_size; i++)// 会引用到rhs[-1]，由于一定有a[0]=0，故要求其不为nan
    //     rhs[i] = (rhs[i] - a[i] * rhs[i - 1]) / b[i];
    // 注意为了使用向量化的写法，需要提前padding！！！
    rhs[-1] = 0.0;
    sol[n_size] = 0.0;

    // 前代
    idx_t k = 0, max_k = ((n_size)&(~(NEON_LEN-1)));
    const data_t * a_ptr = a, * b_ptr = b;

    // 为前代的后半部分准备预取
    __builtin_prefetch(a_ptr + 32, 0, 0);
    __builtin_prefetch(b_ptr + 32, 0, 0);
    __builtin_prefetch(rhs + 16, 1); __builtin_prefetch(rhs + 32, 1); __builtin_prefetch(rhs + 48, 1);
    __builtin_prefetch(rhs + 64, 1);
    // 为回代的前半部分准备预取
    __builtin_prefetch(sol + 64, 1); __builtin_prefetch(sol + 48, 1);
    __builtin_prefetch(c + 32, 0, 0);
    
    // 等价变换 rhs[i] = rhs[i]/b[i] - a[i]/b[i] * rhs[i-1]
    for (; k < max_k; k += NEON_LEN) {
        float16x4_t a_16 = vld1_f16((__fp16*) a_ptr), b_16 = vld1_f16((__fp16*) b_ptr);
        float32x4_t a_32 = vcvt_f32_f16(a_16), b_32 = vcvt_f32_f16(b_16);

        float32x4_t rhs_32 = vld1q_f32(rhs);
        float32x4_t rhs_div_b = vdivq_f32(rhs_32, b_32);
        float32x4_t   a_div_b = vdivq_f32(  a_32, b_32);

        rhs[0] = vgetq_lane_f32(rhs_div_b, 0) - vgetq_lane_f32(a_div_b, 0) * rhs[-1];
        rhs[1] = vgetq_lane_f32(rhs_div_b, 1) - vgetq_lane_f32(a_div_b, 1) * rhs[ 0];
        rhs[2] = vgetq_lane_f32(rhs_div_b, 2) - vgetq_lane_f32(a_div_b, 2) * rhs[ 1];
        rhs[3] = vgetq_lane_f32(rhs_div_b, 3) - vgetq_lane_f32(a_div_b, 3) * rhs[ 2];

        a_ptr += NEON_LEN; b_ptr += NEON_LEN;
        rhs += NEON_LEN;
    }// 结束时 k == max_k，且k和a_ptr, b_ptr, rhs等偏移相同
    for (k = 0; k < n_size - max_k; k++) {
        rhs[k] = (rhs[k] - a_ptr[k] * rhs[k - 1]) / b_ptr[k];
    }

    // 回代
    sol += max_k;
    const data_t * c_ptr = c + max_k;
    for (k = n_size - max_k - 1; k >= 0; k--) {
        sol[k] = rhs[k] - c_ptr[k] * sol[k + 1];
    }
    
    // 为前代的后半部分准备预取
    __builtin_prefetch(c_ptr - 32, 0, 0);
    __builtin_prefetch(sol - 64, 1); __builtin_prefetch(sol - 48, 1);
    __builtin_prefetch(rhs - 32, 0, 0);

    for (k = max_k; k > 0; ) {
        // 指针先往后移，腾出载入寄存器的长度
        c_ptr -= NEON_LEN;
        rhs -= NEON_LEN; sol -= NEON_LEN;

        float16x4_t c_16 = vld1_f16((__fp16*) c_ptr);
        float32x4_t c_32 = vcvt_f32_f16(c_16);
        
        sol[3] = rhs[3] - vgetq_lane_f32(c_32, 3) * sol[4];
        sol[2] = rhs[2] - vgetq_lane_f32(c_32, 2) * sol[3];
        sol[1] = rhs[1] - vgetq_lane_f32(c_32, 1) * sol[2];
        sol[0] = rhs[0] - vgetq_lane_f32(c_32, 0) * sol[1];

        k -= NEON_LEN;
    }

    // rhs -= max_k;// 恢复原来的指向
    // __builtin_prefetch(c, 0, 0);
    // __builtin_prefetch(sol, 1);
    // __builtin_prefetch(rhs, 0, 0);
    // for (idx_t i = n_size - 1; i >= 0; i--)// 会引用到sol[n_size]，由于一定有c[n_size-1]=0，故要求其不为nan
    //     sol[i] = rhs[i] - c[i] * sol[i + 1];
}

#undef NEON_LEN
#endif

template<typename idx_t, typename data_t>
void tridiag_thomas(data_t * a, data_t * b, data_t * c, data_t * d, data_t * sol, idx_t n_size)
{
    /* Fortran style (1-based):
       _                                                _   _      _     _      _
      |  b(1)  c(1)                                      | | x(1  ) |   | d(1  ) |
      |  a(2)  b(2)  c(2)                                | | x(2  ) |   | d(2  ) |
      |        a(3)  b(3)  c(3)                          | | x(3  ) |   | d(3  ) |
      |           ...  ...  ...                          | | ...    | = | ...    |
      |                ...  ...  ...                     | | ...    |   | ...    |
      |                          a(n-1)  b(n-1)  c(n-1)  | | x(n-1) |   | d(n-1) |
      |_                                 a(n  )  b(n  ) _| |_x(n  )_|   |_d(n  )_|
    */

    c[0] /= b[0];
    d[0] /= b[0];
    for (idx_t i = 1; i < n_size; i++) {
        // 要消掉a[i]这个非零元会对b[i]造成的影响
        data_t denom = b[i] - a[i] * c[i - 1];
        c[i] /= denom;
        d[i] = (d[i] - a[i] * d[i - 1]) / denom;
    }

    sol[n_size - 1] = d[n_size - 1];
    for (idx_t i = n_size - 2; i >= 0; i--)
        sol[i] = d[i] - c[i] * sol[i + 1];
}

// 尽可能使用向量化指令
// 提前算好a[i]/b[i] 和 rhs[i]/b[i]

template<typename idx_t, typename data_t>
void tridiag_thomos_factored(const data_t * __restrict__ a, const data_t * __restrict__ b,
    const data_t * __restrict__ c, data_t * __restrict__ rhs, data_t * __restrict__ sol, idx_t n_size)
{
    // 前代
    rhs[0] /= b[0];
    for (idx_t i = 1; i < n_size; i++) 
        rhs[i] = (rhs[i] - a[i] * rhs[i - 1]) / b[i];

    // 回代
    sol[n_size - 1] = rhs[n_size - 1];
    for (idx_t i = n_size - 2; i >= 0; i--)
        sol[i] = rhs[i] - c[i] * sol[i + 1];
}

#endif