#ifndef SMG_KERNELS_2D9_HPP
#define SMG_KERNELS_2D9_HPP

// ===================================================================
// ========================  2d9  kernels  ===========================
// ===================================================================
        /*  
                  
               2-------5-------8
               |       |       |
   z           |       |       |
   ^           1-------4 ------7
   |           |       |       |
   O-------> x |       |       |
               0-------3-------6
        */
// =================================== SPMV =============================

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_spmv_2d9(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const data_t * dummy)
{
    const calc_t * x_iN = x_jik - vec_k_size, * x_iP = x_jik + vec_k_size;
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k] =
                A_jik[0] * x_iN [k-1]
            +   A_jik[1] * x_iN [k  ]
            +   A_jik[2] * x_iN [k+1]
            +   A_jik[3] * x_jik[k-1]
            +   A_jik[4] * x_jik[k  ]
            +   A_jik[5] * x_jik[k+1]
            +   A_jik[6] * x_iP [k-1]
            +   A_jik[7] * x_iP [k  ]
            +   A_jik[8] * x_iP [k+1];
        A_jik += 9;// move the ptr
    }
}

// ================================= LGS =======================================
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_forward_zero_2d9(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * dummy, const calc_t * b_jik, const calc_t * x_jik, const calc_t * sqD_jik, calc_t * rhs)
{
    const calc_t * x_iN = x_jik - vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp = 
            + L_jik[0] * x_iN [k-1]
            + L_jik[1] * x_iN [k  ]
            + L_jik[2] * x_iN [k+1];// L * x_{k+1}
        rhs[k] = b_jik[k] - tmp;// b - L*x_{k+1}
        L_jik += 3;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_backward_zero_2d9(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * dummy, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, const calc_t * sqD_jik, calc_t * rhs)
{
    const calc_t * x_iP = x_jik   + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
            + U_jik[0] * x_iP [k-1]
            + U_jik[1] * x_iP [k  ]
            + U_jik[2] * x_iP [k+1];
        rhs[k] = b_jik[k] - tmp;
        U_jik += 3;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_ALL_2d9(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, const calc_t * sqD_jik, calc_t * rhs)
{
    const calc_t * x_iN = x_jik - vec_k_size, * x_iP = x_jik + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
            + L_jik[0] * x_iN [k-1]
            + L_jik[1] * x_iN [k  ]
            + L_jik[2] * x_iN [k+1]
            + U_jik[0] * x_iP [k-1]
            + U_jik[1] * x_iP [k  ]
            + U_jik[2] * x_iP [k+1];
        rhs[k] = b_jik[k] - tmp;
        L_jik += 3;
        U_jik += 3;
    }
}

/* 
---------------------------------------------------

    Structure of Array: diagonals separated !!!

-------------------------------------------------------
*/
#ifdef __aarch64__
#define GROUP_LEN 8
#define NEON_LEN 4
// ============================ SPMV =================================

// ============================ PGS ==================================

// ========================= PGS (Scaled) ==============================

// ============================ LGS ===================================


#undef NEON_LEN
#undef GROUP_LEN
#endif

#endif