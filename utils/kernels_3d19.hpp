#ifndef SMG_KERNELS_3D19_HPP
#define SMG_KERNELS_3D19_HPP

// ===================================================================
// ========================  3d19  kernels  ==========================
// ===================================================================
 /*
                                                    内存读入后的格式
                    /----- 17 -----/                    
                  7 |    10      13|                   
                /---|-- 3 ------/  |
               |    |           |  |
               |    14---- 16 --|- 18
   z   y       |  6 |    9      |12|
   ^  ^        0 ---|-- 2 ----- 4  |
   | /         |    |           |  |
   |/          |    /----- 15 --|--/ 
   O-------> x |  5      8      |11 
               |/------ 1 ------|/
            */

// =================================== SPMV =============================

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_spmv_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const data_t * dummy)
{
    const calc_t * x_jNi = x_jik - vec_ki_size, * x_jPi = x_jik + vec_ki_size;
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k] = 
            + A_jik[ 0] * x_jNi[- vec_k_size + k  ]
            + A_jik[ 1] * x_jNi[               k-1]
            + A_jik[ 2] * x_jNi[               k  ]
            + A_jik[ 3] * x_jNi[               k+1]
            + A_jik[ 4] * x_jNi[  vec_k_size + k  ]

            + A_jik[ 5] * x_jik[- vec_k_size + k-1]
            + A_jik[ 6] * x_jik[- vec_k_size + k  ]
            + A_jik[ 7] * x_jik[- vec_k_size + k+1]
            + A_jik[ 8] * x_jik[               k-1]
            + A_jik[ 9] * x_jik[               k  ]
            + A_jik[10] * x_jik[               k+1]
            + A_jik[11] * x_jik[  vec_k_size + k-1]
            + A_jik[12] * x_jik[  vec_k_size + k  ]
            + A_jik[13] * x_jik[  vec_k_size + k+1]

            + A_jik[14] * x_jPi[- vec_k_size + k  ]
            + A_jik[15] * x_jPi[               k-1]
            + A_jik[16] * x_jPi[               k  ]
            + A_jik[17] * x_jPi[               k+1]
            + A_jik[18] * x_jPi[  vec_k_size + k  ];

        A_jik += 19;// move the ptr
    }
}

// template<typename idx_t, typename data_t, typename calc_t>
// void inline SOA_spmv_3d19(const idx_t num,
//     const idx_t vec_k_size , const idx_t vec_ki_size,
//     const data_t * A_jik[19], const calc_t * x_jik, calc_t * y_jik)
// {
//     // 要打桩查一下是否生成了向量化代码！！！！
//     const calc_t * x2 = x_jik - vec_ki_size, * x16 = x_jik + vec_ki_size,
//                 * x6 = x_jik - vec_k_size , * x12 = x_jik + vec_k_size,
//                 * x8 = x_jik - 1, * x10 = x_jik + 1;
//     const calc_t * x0 = x2 - vec_k_size, * x4 = x2 + vec_k_size,
//                 * x3 = x2 + 1, * x1 = x2 - 1,
//                 * x14= x16- vec_k_size, * x18= x16+ vec_k_size,
//                 * x17= x16+ 1, * x15= x16- 1,
//                 * x5 = x8 - vec_k_size, * x11 = x8 + vec_k_size,
//                 * x7 = x10- vec_k_size, * x13 = x10+ vec_k_size;
//     const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3],
//                 * A4 = A_jik[4], * A5 = A_jik[5], * A6 = A_jik[6], * A7 = A_jik[7],
//                 * A8 = A_jik[8], * A9 = A_jik[9], * A10= A_jik[10],* A11= A_jik[11],
//                 * A12= A_jik[12],* A13= A_jik[13],* A14= A_jik[14],* A15= A_jik[15],
//                 * A16= A_jik[16],* A17= A_jik[17],* A18= A_jik[18]; 
//     #pragma GCC unroll (4)
//     for (idx_t k = 0; k < num; k++) {
//         y_jik[k]= A0[k] * x0[k] + A1[k] * x1[k] + A2[k] * x2[k] + A3[k] * x3[k]
//                 + A4[k] * x4[k] + A5[k] * x5[k] + A6[k] * x6[k] + A7[k] * x7[k]
//                 + A8[k] *  x8[k]  + A9[k] * x_jik[k]+ A10[k] * x10[k]
//                 + A11[k] * x11[k] + A12[k] * x12[k] + A13[k] * x13[k]
//                 + A14[k] * x14[k] + A15[k] * x15[k] + A16[k] * x16[k]
//                 + A17[k] * x17[k] + A18[k] * x18[k];
//     }
// }

// =================================== PGS =============================

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_zero_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jZiN = x_jik - vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[9];
        calc_t tmp = 
            + L_jik[0] * x_jNiZ[- vec_k_size + k  ]
            + L_jik[1] * x_jNiZ[             + k-1]
            + L_jik[2] * x_jNiZ[             + k  ]
            + L_jik[3] * x_jNiZ[             + k+1]
            + L_jik[4] * x_jNiZ[  vec_k_size + k  ]
            + L_jik[5] * x_jZiN[               k-1]
            + L_jik[6] * x_jZiN[               k  ]
            + L_jik[7] * x_jZiN[               k+1]
            + L_jik[8] * x_jik [               k-1];// L * x_{k+1}
        tmp = b_jik[k] - tmp;// b - L*x_{k+1}
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 10;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_zero_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t* x_jPiZ = x_jik + vec_ki_size,
                * x_jZiP = x_jik + vec_k_size; 
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[9];
        calc_t tmp = 
            + U_jik[0] * x_jik [               k+1]
            + U_jik[1] * x_jZiP[               k-1]
            + U_jik[2] * x_jZiP[               k  ]
            + U_jik[3] * x_jZiP[               k+1]
            + U_jik[4] * x_jPiZ[- vec_k_size + k  ]
            + U_jik[5] * x_jPiZ[               k-1]
            + U_jik[6] * x_jPiZ[               k  ]
            + U_jik[7] * x_jPiZ[               k+1]
            + U_jik[8] * x_jPiZ[  vec_k_size + k  ];// U*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k+1}

        x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 10;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_ALL_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t* x_jNiZ = x_jik - vec_ki_size, * x_jPiZ = x_jik + vec_ki_size,
                * x_jZiN = x_jik - vec_k_size , * x_jZiP = x_jik + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[9];// = U_jik[3]
        calc_t tmp = 
        + L_jik[0] * x_jNiZ[- vec_k_size + k  ]
        + L_jik[1] * x_jNiZ[             + k-1]
        + L_jik[2] * x_jNiZ[             + k  ]
        + L_jik[3] * x_jNiZ[             + k+1]
        + L_jik[4] * x_jNiZ[  vec_k_size + k  ]
        + L_jik[5] * x_jZiN[               k-1]
        + L_jik[6] * x_jZiN[               k  ]
        + L_jik[7] * x_jZiN[               k+1]
        + L_jik[8] * x_jik [               k-1]
        // 
        + U_jik[0] * x_jik [               k+1]
        + U_jik[1] * x_jZiP[               k-1]
        + U_jik[2] * x_jZiP[               k  ]
        + U_jik[3] * x_jZiP[               k+1]
        + U_jik[4] * x_jPiZ[- vec_k_size + k  ]
        + U_jik[5] * x_jPiZ[               k-1]
        + U_jik[6] * x_jPiZ[               k  ]
        + U_jik[7] * x_jPiZ[               k+1]
        + U_jik[8] * x_jPiZ[  vec_k_size + k  ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik += 10;
        U_jik += 10;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_ALL_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t* x_jNiZ = x_jik - vec_ki_size, * x_jPiZ = x_jik + vec_ki_size,
                * x_jZiN = x_jik - vec_k_size , * x_jZiP = x_jik + vec_k_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[9];
        calc_t tmp = 
        + L_jik[0] * x_jNiZ[- vec_k_size + k  ]
        + L_jik[1] * x_jNiZ[             + k-1]
        + L_jik[2] * x_jNiZ[             + k  ]
        + L_jik[3] * x_jNiZ[             + k+1]
        + L_jik[4] * x_jNiZ[  vec_k_size + k  ]
        + L_jik[5] * x_jZiN[               k-1]
        + L_jik[6] * x_jZiN[               k  ]
        + L_jik[7] * x_jZiN[               k+1]
        + L_jik[8] * x_jik [               k-1]
        // 
        + U_jik[0] * x_jik [               k+1]
        + U_jik[1] * x_jZiP[               k-1]
        + U_jik[2] * x_jZiP[               k  ]
        + U_jik[3] * x_jZiP[               k+1]
        + U_jik[4] * x_jPiZ[- vec_k_size + k  ]
        + U_jik[5] * x_jPiZ[               k-1]
        + U_jik[6] * x_jPiZ[               k  ]
        + U_jik[7] * x_jPiZ[               k+1]
        + U_jik[8] * x_jPiZ[  vec_k_size + k  ];
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik -= 10;
        U_jik -= 10;
    }
}

// =================================== LGS =============================

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_forward_zero_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * dummy, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t* x_jNiZ = x_jik  - vec_ki_size,
                * x_jZiN = x_jik  - vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp = 
            + L_jik[0] * x_jNiZ[ - vec_k_size + k  ]
            + L_jik[1] * x_jNiZ[                k-1]
            + L_jik[2] * x_jNiZ[                k  ]
            + L_jik[3] * x_jNiZ[                k+1]
            + L_jik[4] * x_jNiZ[   vec_k_size + k  ]
            + L_jik[5] * x_jZiN[                k-1]
            + L_jik[6] * x_jZiN[                k  ]
            + L_jik[7] * x_jZiN[                k+1];// L * x_{k+1}
        rhs[k] = b_jik[k] - tmp;// b - L*x_{k+1}
        L_jik += 8;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_backward_zero_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * dummy, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t* x_jPiZ  = x_jik   + vec_ki_size,
                * x_jZiP  = x_jik   + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
            + U_jik[0] * x_jZiP[                k-1]
            + U_jik[1] * x_jZiP[                k  ]
            + U_jik[2] * x_jZiP[                k+1]
            + U_jik[3] * x_jPiZ[ - vec_k_size + k  ]
            + U_jik[4] * x_jPiZ[                k-1]
            + U_jik[5] * x_jPiZ[                k  ]
            + U_jik[6] * x_jPiZ[                k+1]
            + U_jik[7] * x_jPiZ[   vec_k_size + k  ];
        rhs[k] = b_jik[k] - tmp;
        U_jik += 8;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_ALL_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t* x_jNiZ  = x_jik - vec_ki_size, * x_jZiN = x_jik - vec_k_size,
                * x_jPiZ  = x_jik + vec_ki_size, * x_jZiP = x_jik + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
            + L_jik[0] * x_jNiZ[ - vec_k_size + k  ]
            + L_jik[1] * x_jNiZ[                k-1]
            + L_jik[2] * x_jNiZ[                k  ]
            + L_jik[3] * x_jNiZ[                k+1]
            + L_jik[4] * x_jNiZ[   vec_k_size + k  ]
            + L_jik[5] * x_jZiN[                k-1]
            + L_jik[6] * x_jZiN[                k  ]
            + L_jik[7] * x_jZiN[                k+1]
            + U_jik[0] * x_jZiP[                k-1]
            + U_jik[1] * x_jZiP[                k  ]
            + U_jik[2] * x_jZiP[                k+1]
            + U_jik[3] * x_jPiZ[ - vec_k_size + k  ]
            + U_jik[4] * x_jPiZ[                k-1]
            + U_jik[5] * x_jPiZ[                k  ]
            + U_jik[6] * x_jPiZ[                k+1]
            + U_jik[7] * x_jPiZ[   vec_k_size + k  ];
        rhs[k] = b_jik[k] - tmp;
        L_jik += 8;
        U_jik += 8;
    }
}

// =================================== BILU ==================================
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_forward_zero_3d19(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t* x_jNiZ = x_jik  - dim_1 * dim_2,
                * x_jZiN = x_jik  - dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        calc_t tmp = 
            + L_jik[0] * x_jNiZ[ - dim_2 + k  ]
            + L_jik[1] * x_jNiZ[           k-1]
            + L_jik[2] * x_jNiZ[           k  ]
            + L_jik[3] * x_jNiZ[           k+1]
            + L_jik[4] * x_jNiZ[   dim_2 + k  ]
            + L_jik[5] * x_jZiN[           k-1]
            + L_jik[6] * x_jZiN[           k  ]
            + L_jik[7] * x_jZiN[           k+1]
            + L_jik[8] * x_jik [           k-1];// L * x_{k+1}
        x_jik[k] = b_jik[k] - tmp;
        L_jik += 9;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_backward_zero_3d19(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t* x_jPiZ  = x_jik   + dim_1 * dim_2,
                * x_jZiP  = x_jik   + dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        calc_t para = U_jik[0];
        calc_t tmp = 
            + U_jik[1] * x_jik [           k+1]
            + U_jik[2] * x_jZiP[           k-1]
            + U_jik[3] * x_jZiP[           k  ]
            + U_jik[4] * x_jZiP[           k+1]
            + U_jik[5] * x_jPiZ[ - dim_2 + k  ]
            + U_jik[6] * x_jPiZ[           k-1]
            + U_jik[7] * x_jPiZ[           k  ]
            + U_jik[8] * x_jPiZ[           k+1]
            + U_jik[9] * x_jPiZ[   dim_2 + k  ];
        x_jik[k] = (b_jik[k] - tmp) / para;
        U_jik -= 10;
    }
}

#define NEON_LEN 4
// =========================================================================
// =========================== Structure Of Array ==========================
// =========================================================================

// ================================ SPMV ===================================
#define GROUP_LEN 8
void inline SOA_spmv_3d19_Cal32Stg16(const int num,
    const int vec_k_size , const int vec_ki_size,
    const __fp16 * Diags[5], const float * x9, float * y_jik)
{
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2], * A12_15 = Diags[3], * A16_18 = Diags[4];
    const float * x2 = x9 - vec_ki_size, * x16 = x9 + vec_ki_size,
                * x6 = x9 - vec_k_size , * x12 = x9 + vec_k_size ,
                * x8 = x9 - 1          , * x10 = x9 + 1          ;
    const float * x0 = x2 - vec_k_size , * x4  = x2 + vec_k_size ,
                * x1 = x2 - 1          , * x3  = x2 + 1          ,
                * x14= x16- vec_k_size , * x18 = x16+ vec_k_size ,
                * x15= x16- 1          , * x17 = x16+ 1          ,
                * x5 = x6 - 1          , * x7  = x6 + 1          ,
                * x11= x12- 1          , * x13 = x12+ 1          ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0,
                x0_32_1, x1_32_1, x2_32_1, x3_32_1;
    float32x4_t tmp_0, tmp_1;
    // A0_3每个GROUP会读取8*4=32个半精度数，即一条cacheline，x*等会读取8个单精度数 32B
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float16x8x3_t A16_18_16;
        tmp_0 = vdupq_n_f32(0.0); tmp_1 = vdupq_n_f32(0.0);
        // A0~A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlaq_f32(tmp_1, A3_32_1, x3_32_1);
        // A4~A7
        A0_3_16 = vld4q_f16(A4_7); A4_7 += GROUP_LEN * 4; __builtin_prefetch(A4_7 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7, 0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlaq_f32(tmp_1, A3_32_1, x3_32_1);
        // A8~A11
        A0_3_16 = vld4q_f16(A8_11); A8_11  += GROUP_LEN * 4; __builtin_prefetch(A8_11 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8) ; x8  += NEON_LEN; x0_32_1 = vld1q_f32(x8);  x8  += NEON_LEN; __builtin_prefetch(x8, 0);
        x1_32_0 = vld1q_f32(x9) ; x9  += NEON_LEN; x1_32_1 = vld1q_f32(x9);  x9  += NEON_LEN; __builtin_prefetch(x9, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10,0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlaq_f32(tmp_1, A3_32_1, x3_32_1);
        // A12~A15
        A0_3_16 = vld4q_f16(A12_15); A12_15 += GROUP_LEN * 4; __builtin_prefetch(A12_15 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x12); x12 += NEON_LEN; x0_32_1 = vld1q_f32(x12); x12 += NEON_LEN; __builtin_prefetch(x12,0);
        x1_32_0 = vld1q_f32(x13); x13 += NEON_LEN; x1_32_1 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13,0);
        x2_32_0 = vld1q_f32(x14); x14 += NEON_LEN; x2_32_1 = vld1q_f32(x14); x14 += NEON_LEN; __builtin_prefetch(x14,0);
        x3_32_0 = vld1q_f32(x15); x15 += NEON_LEN; x3_32_1 = vld1q_f32(x15); x15 += NEON_LEN; __builtin_prefetch(x15,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlaq_f32(tmp_1, A3_32_1, x3_32_1);
        // A16~A18
        A16_18_16 = vld3q_f16(A16_18);  A16_18  += GROUP_LEN * 3; __builtin_prefetch(A16_18 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A16_18_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A16_18_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A16_18_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A16_18_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A16_18_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A16_18_16.val[2]);
        x0_32_0 = vld1q_f32(x16); x16 += NEON_LEN; x0_32_1 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16,0);
        x1_32_0 = vld1q_f32(x17); x17 += NEON_LEN; x1_32_1 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17,0);
        x2_32_0 = vld1q_f32(x18); x18 += NEON_LEN; x2_32_1 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);

        vst1q_f32(y_jik           , tmp_0);
        vst1q_f32(y_jik + NEON_LEN, tmp_1);
        y_jik += GROUP_LEN; __builtin_prefetch(y_jik,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A16_18_16;
        tmp_0 = vdupq_n_f32(0.0);
        // A0~A3
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A4~A7
        A0_3_16 = vld4_f16(A4_7); A4_7 += NEON_LEN * 4; __builtin_prefetch(A4_7 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7, 0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A8~A11
        A0_3_16 = vld4_f16(A8_11); A8_11 += NEON_LEN * 4; __builtin_prefetch(A8_11 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8) ; x8  += NEON_LEN; __builtin_prefetch(x8, 0);
        x1_32_0 = vld1q_f32(x9) ; x9  += NEON_LEN; __builtin_prefetch(x9, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10,0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A12~A15
        A0_3_16 = vld4_f16(A12_15); A12_15 += NEON_LEN * 4; __builtin_prefetch(A12_15 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x12); x12 += NEON_LEN; __builtin_prefetch(x12,0);
        x1_32_0 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13,0);
        x2_32_0 = vld1q_f32(x14); x14 += NEON_LEN; __builtin_prefetch(x14,0);
        x3_32_0 = vld1q_f32(x15); x15 += NEON_LEN; __builtin_prefetch(x15,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A16~A18
        A16_18_16 = vld3_f16(A16_18); A16_18 += NEON_LEN * 3; __builtin_prefetch(A16_18 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A16_18_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A16_18_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A16_18_16.val[2]);
        x0_32_0 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16,0);
        x1_32_0 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17,0);
        x2_32_0 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);

        vst1q_f32(y_jik           , tmp_0);
        y_jik += NEON_LEN; __builtin_prefetch(y_jik,1);
    }
    for (k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        y_jik[k] = 
            A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        +   A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        +   A8_11[0]* x8[k] + A8_11[1]* x9[k] + A8_11[2]* x9[k] + A8_11[3]* x9[k]
        +   A12_15[0]*x12[k]+ A12_15[1]*x13[k]+ A12_15[2]*x14[k]+ A12_15[3]*x15[k]
        +   A16_18[0]*x16[k]+ A16_18[1]*x17[k]+ A16_18[2]*x18[k];
        A0_3 += 4; A4_7 += 4; A8_11+= 4; A12_15+= 4;
        A16_18+= 3;
    }
}


// =================================== PGS =============================
void inline SOA_point_forward_zero_3d19_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[3], const float * b9, float * x9, const float * dummy)
{// 0,1,2,3一组，4,5,6,7一组，8,9,10一组
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_10 = Diags[2];
    const float * x2 = x9 - vec_ki_size, * x6 = x9 - vec_k_size;
    const float * x0 = x2 - vec_k_size , * x4 = x2 + vec_k_size,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0,
                x0_32_1, x1_32_1, x2_32_1, x3_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float16x8x3_t A8_10_16;
        float32x4_t tmp_0, tmp_1;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; tmp_1   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + GROUP_LEN, 0);
        // A0 ~ A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A4 ~ A7
        A0_3_16 = vld4q_f16(A4_7); A4_7 += GROUP_LEN * 4; __builtin_prefetch(A4_7 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // 本柱的 A8 ~ A9
        A8_10_16 = vld3q_f16(A8_10); A8_10 += GROUP_LEN * 3; __builtin_prefetch(A8_10 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A8_10_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A8_10_16.val[1]);// A9
        A1_32_0 = vdivq_f32(vwgts, A1_32_0); A1_32_1 = vdivq_f32(vwgts, A1_32_1);
        vst1q_f32(x9, tmp_0);
        x9[0] = (x9[0] - vgetq_lane_f32(A0_32_0, 0) * x9[-1]) * vgetq_lane_f32(A1_32_0, 0);
        x9[1] = (x9[1] - vgetq_lane_f32(A0_32_0, 1) * x9[ 0]) * vgetq_lane_f32(A1_32_0, 1);
        x9[2] = (x9[2] - vgetq_lane_f32(A0_32_0, 2) * x9[ 1]) * vgetq_lane_f32(A1_32_0, 2);
        x9[3] = (x9[3] - vgetq_lane_f32(A0_32_0, 3) * x9[ 2]) * vgetq_lane_f32(A1_32_0, 3);
        x9 += NEON_LEN;
        vst1q_f32(x9, tmp_1);
        x9[0] = (x9[0] - vgetq_lane_f32(A0_32_1, 0) * x9[-1]) * vgetq_lane_f32(A1_32_1, 0);
        x9[1] = (x9[1] - vgetq_lane_f32(A0_32_1, 1) * x9[ 0]) * vgetq_lane_f32(A1_32_1, 1);
        x9[2] = (x9[2] - vgetq_lane_f32(A0_32_1, 2) * x9[ 1]) * vgetq_lane_f32(A1_32_1, 2);
        x9[3] = (x9[3] - vgetq_lane_f32(A0_32_1, 3) * x9[ 2]) * vgetq_lane_f32(A1_32_1, 3);
        x9 += NEON_LEN; __builtin_prefetch(x9 + GROUP_LEN,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A8_10_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + NEON_LEN, 0);
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);

        A0_3_16 = vld4_f16(A4_7); A4_7 += NEON_LEN * 4; __builtin_prefetch(A4_7 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);

        A8_10_16 = vld3_f16(A8_10); A8_10 += NEON_LEN * 3; __builtin_prefetch(A8_10 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A8_10_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(A8_10_16.val[1]);// A9
        A1_32_0 = vdivq_f32(vwgts, A1_32_0);
        vst1q_f32(x9, tmp_0);
        x9[0] = (x9[0] - vgetq_lane_f32(A0_32_0, 0) * x9[-1]) * vgetq_lane_f32(A1_32_0, 0);
        x9[1] = (x9[1] - vgetq_lane_f32(A0_32_0, 1) * x9[ 0]) * vgetq_lane_f32(A1_32_0, 1);
        x9[2] = (x9[2] - vgetq_lane_f32(A0_32_0, 2) * x9[ 1]) * vgetq_lane_f32(A1_32_0, 2);
        x9[3] = (x9[3] - vgetq_lane_f32(A0_32_0, 3) * x9[ 2]) * vgetq_lane_f32(A1_32_0, 3);
        x9 += NEON_LEN; __builtin_prefetch(x9 + NEON_LEN,1);
    }
    for ( k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        float diag_val = A8_10[1];
        float tmp = 
        + A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        + A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        + A8_10[0]* x9[k-1];
        tmp = b9[k] - tmp;// b - L*x_{k+1}
        x9[k] = weight * tmp / diag_val;
        A0_3 += 4; A4_7 += 4; A8_10 += 3;
    }
}

void inline SOA_point_forward_ALL_3d19_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[5], const float * b9, float * x9, const float * dummy)
{// 0,1,2,3一组，4,5,6,7一组，8,9,10一组，11,12,13,14一组，15,16,17,18一组
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_10 = Diags[2], * A11_14 = Diags[3], * A15_18 = Diags[4];
    const float * x2 = x9 - vec_ki_size, * x6 = x9 - vec_k_size,
                * x16= x9 + vec_ki_size, * x12= x9 + vec_k_size;
    const float * x0 = x2 - vec_k_size, * x4 = x2 + vec_k_size,
                * x14= x16- vec_k_size, * x18= x16+ vec_k_size,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1,
                * x11= x12- 1, * x13= x12+ 1, * x15= x16- 1, * x17= x16+ 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0,
                x0_32_1, x1_32_1, x2_32_1, x3_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float16x8x3_t A8_10_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; tmp_1   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + GROUP_LEN, 0);
        // A0~A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A4~A7
        A0_3_16 = vld4q_f16(A4_7); A4_7 += GROUP_LEN * 4; __builtin_prefetch(A4_7 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);

        // A11~A14
        A0_3_16 = vld4q_f16(A11_14); A11_14 += GROUP_LEN * 4; __builtin_prefetch(A11_14 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x0_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x12); x12 += NEON_LEN; x1_32_1 = vld1q_f32(x12); x12 += NEON_LEN; __builtin_prefetch(x12 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x13); x13 += NEON_LEN; x2_32_1 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x14); x14 += NEON_LEN; x3_32_1 = vld1q_f32(x14); x14 += NEON_LEN; __builtin_prefetch(x14 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A15~A18
        A0_3_16 = vld4q_f16(A15_18); A15_18 += GROUP_LEN * 4; __builtin_prefetch(A15_18 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x15); x15 += NEON_LEN; x0_32_1 = vld1q_f32(x15); x15 += NEON_LEN; __builtin_prefetch(x15 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x16); x16 += NEON_LEN; x1_32_1 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x17); x17 += NEON_LEN; x2_32_1 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x18); x18 += NEON_LEN; x3_32_1 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // 本柱的A8~A10
        A8_10_16 = vld3q_f16(A8_10); A8_10 += GROUP_LEN * 3; __builtin_prefetch(A8_10 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A8_10_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A8_10_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A8_10_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0); A1_32_1 = vdivq_f32(vwgts, A1_32_1);// 此时A1_32存着wgt/A9
        float A8_buf[GROUP_LEN], A10_buf[GROUP_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0); A0_32_1 = vmulq_f32(A0_32_1, A1_32_1);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0); A2_32_1 = vmulq_f32(A2_32_1, A1_32_1);
        vst1q_f32(A8_buf , A0_32_0); vst1q_f32(A8_buf  + NEON_LEN, A0_32_1);
        vst1q_f32(A10_buf, A2_32_0); vst1q_f32(A10_buf + NEON_LEN, A2_32_1);
        float * x_jik = x9;
        res_0 = vld1q_f32(x9); x9 += NEON_LEN     ; res_1 = vld1q_f32(x9); x9 += NEON_LEN; __builtin_prefetch(x9 + GROUP_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; res_1 = vmulq_f32(res_1, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0); res_1 = vmlaq_f32(res_1, tmp_1, A1_32_1);// 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A8_buf[0] * x_jik[-1] + A10_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A8_buf[1] * x_jik[ 0] + A10_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A8_buf[2] * x_jik[ 1] + A10_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A8_buf[3] * x_jik[ 2] + A10_buf[3] * x_jik[4]);
        x_jik[4] = vgetq_lane_f32(res_1, 0) - (A8_buf[4] * x_jik[ 3] + A10_buf[4] * x_jik[5]);
        x_jik[5] = vgetq_lane_f32(res_1, 1) - (A8_buf[5] * x_jik[ 4] + A10_buf[5] * x_jik[6]);
        x_jik[6] = vgetq_lane_f32(res_1, 2) - (A8_buf[6] * x_jik[ 5] + A10_buf[6] * x_jik[7]);
        x_jik[7] = vgetq_lane_f32(res_1, 3) - (A8_buf[7] * x_jik[ 6] + A10_buf[7] * x_jik[8]);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A8_10_16;
        float32x4_t tmp_0, res_0;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + NEON_LEN, 0);
        // A0~A3
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A4~A7
        A0_3_16 = vld4_f16(A4_7); A4_7 += NEON_LEN * 4; __builtin_prefetch(A4_7 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A11~A14
        A0_3_16 = vld4_f16(A11_14); A11_14 += NEON_LEN * 4; __builtin_prefetch(A11_14 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x12); x12 += NEON_LEN; __builtin_prefetch(x12 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x14); x14 += NEON_LEN; __builtin_prefetch(x14 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15~A18
        A0_3_16 = vld4_f16(A15_18); A15_18 += NEON_LEN * 4; __builtin_prefetch(A15_18 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x15); x15 += NEON_LEN; __builtin_prefetch(x15 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 本柱的A8~A10
        A8_10_16 = vld3_f16(A8_10); A8_10 += NEON_LEN * 3; __builtin_prefetch(A8_10 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A8_10_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(A8_10_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(A8_10_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0);// 此时A1_32存着wgt/A9
        float A8_buf[NEON_LEN], A10_buf[NEON_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0);
        vst1q_f32(A8_buf , A0_32_0);
        vst1q_f32(A10_buf, A2_32_0);
        float * x_jik = x9;
        res_0 = vld1q_f32(x9); x9 += NEON_LEN     ; __builtin_prefetch(x9 + NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0);// 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A8_buf[0] * x_jik[-1] + A10_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A8_buf[1] * x_jik[ 0] + A10_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A8_buf[2] * x_jik[ 1] + A10_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A8_buf[3] * x_jik[ 2] + A10_buf[3] * x_jik[4]);
    }
    for ( k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        float diag_val = A8_10[1];
        float tmp = 
        + A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        + A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        + A8_10[0]* x9[k-1] + A8_10[2] * x9[k+1]
        + A11_14[0] * x11[k] + A11_14[1] * x12[k] + A11_14[2] * x13[k] + A11_14[3] * x14[k]
        + A15_18[0] * x15[k] + A15_18[1] * x16[k] + A15_18[2] * x17[k] + A15_18[3] * x18[k];
        tmp = b9[k] - tmp;// b - L*x_{k+1}
        x9[k] *= (1.0 - weight);
        x9[k] += weight * tmp / diag_val;
        A0_3 += 4; A4_7 += 4; A8_10 += 3; A11_14 += 4; A15_18 += 4;
    }
}

void inline SOA_point_backward_ALL_3d19_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[5], const float * b9, float * x9, const float * dummy)
{// 0,1,2,3一组，4,5,6,7一组，8,9,10一组，11,12,13,14一组，15,16,17,18一组
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_10 = Diags[2], * A11_14 = Diags[3], * A15_18 = Diags[4];
    const float * x2 = x9 - vec_ki_size, * x6 = x9 - vec_k_size,
                * x16= x9 + vec_ki_size, * x12= x9 + vec_k_size;
    const float * x0 = x2 - vec_k_size, * x4 = x2 + vec_k_size,
                * x14= x16- vec_k_size, * x18= x16+ vec_k_size,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1,
                * x11= x12- 1, * x13= x12+ 1, * x15= x16- 1, * x17= x16+ 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float16x8x3_t A8_10_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        b9 -= NEON_LEN; tmp_1 = vld1q_f32(b9); b9 -= NEON_LEN; tmp_0 = vld1q_f32(b9); __builtin_prefetch(b9 - GROUP_LEN, 0);
        // A0~A3
        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0 -= NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - GROUP_LEN, 0);
        x1 -= NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - GROUP_LEN, 0);
        x2 -= NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 -= NEON_LEN; x2_32_0 = vld1q_f32(x2); __builtin_prefetch(x2 - GROUP_LEN, 0);
        x3 -= NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 -= NEON_LEN; x3_32_0 = vld1q_f32(x3); __builtin_prefetch(x3 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); 
        // A4~A7
        A4_7 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A4_7); __builtin_prefetch(A4_7 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x4 -= NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 -= NEON_LEN; x0_32_0 = vld1q_f32(x4); __builtin_prefetch(x4 - GROUP_LEN, 0);
        x5 -= NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - GROUP_LEN, 0);
        x6 -= NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x2_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - GROUP_LEN, 0);
        x7 -= NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 -= NEON_LEN; x3_32_0 = vld1q_f32(x7); __builtin_prefetch(x7 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A11~A14
        A11_14 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A11_14); __builtin_prefetch(A11_14 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x11 -= NEON_LEN; x0_32_1 = vld1q_f32(x11); x11 -= NEON_LEN; x0_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - GROUP_LEN, 0);
        x12 -= NEON_LEN; x1_32_1 = vld1q_f32(x12); x12 -= NEON_LEN; x1_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - GROUP_LEN, 0);
        x13 -= NEON_LEN; x2_32_1 = vld1q_f32(x13); x13 -= NEON_LEN; x2_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - GROUP_LEN, 0);
        x14 -= NEON_LEN; x3_32_1 = vld1q_f32(x14); x14 -= NEON_LEN; x3_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15~A18
        A15_18 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A15_18); __builtin_prefetch(A15_18 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x15 -= NEON_LEN; x0_32_1 = vld1q_f32(x15); x15 -= NEON_LEN; x0_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - GROUP_LEN, 0);
        x16 -= NEON_LEN; x1_32_1 = vld1q_f32(x16); x16 -= NEON_LEN; x1_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - GROUP_LEN, 0);
        x17 -= NEON_LEN; x2_32_1 = vld1q_f32(x17); x17 -= NEON_LEN; x2_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - GROUP_LEN, 0);
        x18 -= NEON_LEN; x3_32_1 = vld1q_f32(x18); x18 -= NEON_LEN; x3_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 本柱的A8~A10
        A8_10 -= GROUP_LEN * 3;
        A8_10_16 = vld3q_f16(A8_10); __builtin_prefetch(A8_10 - GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A8_10_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A8_10_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A8_10_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A8_10_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0); A1_32_1 = vdivq_f32(vwgts, A1_32_1);// 此时A1_32存着wgt/A9
        float A8_buf[GROUP_LEN], A10_buf[GROUP_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0); A0_32_1 = vmulq_f32(A0_32_1, A1_32_1);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0); A2_32_1 = vmulq_f32(A2_32_1, A1_32_1);
        vst1q_f32(A8_buf , A0_32_0); vst1q_f32(A8_buf  + NEON_LEN, A0_32_1);
        vst1q_f32(A10_buf, A2_32_0); vst1q_f32(A10_buf + NEON_LEN, A2_32_1);

        x9 -= NEON_LEN; res_1 = vld1q_f32(x9); x9 -= NEON_LEN; res_0 = vld1q_f32(x9); __builtin_prefetch(x9 - GROUP_LEN, 1);
        res_1 = vmulq_f32(res_1, vone_minus_wgts); res_0 = vmulq_f32(res_0, vone_minus_wgts);
        res_1 = vmlaq_f32(res_1, tmp_1, A1_32_1);  res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0);
        // 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x9[7] = vgetq_lane_f32(res_1, 3) - (A8_buf[7] * x9[ 6] + A10_buf[7] * x9[8]);
        x9[6] = vgetq_lane_f32(res_1, 2) - (A8_buf[6] * x9[ 5] + A10_buf[6] * x9[7]);
        x9[5] = vgetq_lane_f32(res_1, 1) - (A8_buf[5] * x9[ 4] + A10_buf[5] * x9[6]);
        x9[4] = vgetq_lane_f32(res_1, 0) - (A8_buf[4] * x9[ 3] + A10_buf[4] * x9[5]);
        x9[3] = vgetq_lane_f32(res_0, 3) - (A8_buf[3] * x9[ 2] + A10_buf[3] * x9[4]);
        x9[2] = vgetq_lane_f32(res_0, 2) - (A8_buf[2] * x9[ 1] + A10_buf[2] * x9[3]);
        x9[1] = vgetq_lane_f32(res_0, 1) - (A8_buf[1] * x9[ 0] + A10_buf[1] * x9[2]);
        x9[0] = vgetq_lane_f32(res_0, 0) - (A8_buf[0] * x9[-1] + A10_buf[0] * x9[1]);
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A8_10_16;
        float32x4_t tmp_0, res_0;
        b9 -= NEON_LEN; tmp_0 = vld1q_f32(b9); __builtin_prefetch(b9 - NEON_LEN, 0);
        // A0~A3
        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - NEON_LEN, 0);
        x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - NEON_LEN, 0);
        x2 -= NEON_LEN; x2_32_0 = vld1q_f32(x2); __builtin_prefetch(x2 - NEON_LEN, 0);
        x3 -= NEON_LEN; x3_32_0 = vld1q_f32(x3); __builtin_prefetch(x3 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0, x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0, x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0, x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0, x3_32_0);
        // A4~A7
        A4_7 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A4_7); __builtin_prefetch(A4_7 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x4 -= NEON_LEN; x0_32_0 = vld1q_f32(x4); __builtin_prefetch(x4 - NEON_LEN, 0);
        x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - NEON_LEN, 0);
        x6 -= NEON_LEN; x2_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - NEON_LEN, 0);
        x7 -= NEON_LEN; x3_32_0 = vld1q_f32(x7); __builtin_prefetch(x7 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A11~A14
        A11_14 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A11_14); __builtin_prefetch(A11_14 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x11 -= NEON_LEN; x0_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - NEON_LEN, 0);
        x12 -= NEON_LEN; x1_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - NEON_LEN, 0);
        x13 -= NEON_LEN; x2_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - NEON_LEN, 0);
        x14 -= NEON_LEN; x3_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15~A18
        A15_18 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A15_18); __builtin_prefetch(A15_18 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x15 -= NEON_LEN; x0_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - NEON_LEN, 0);
        x16 -= NEON_LEN; x1_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - NEON_LEN, 0);
        x17 -= NEON_LEN; x2_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - NEON_LEN, 0);
        x18 -= NEON_LEN; x3_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 本柱的A8~A10
        A8_10 -= NEON_LEN * 3;
        A8_10_16 = vld3_f16(A8_10); __builtin_prefetch(A8_10 - NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A8_10_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(A8_10_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(A8_10_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0);// 此时A1_32存着wgt/A9
        float A8_buf[NEON_LEN], A10_buf[NEON_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0);
        vst1q_f32(A8_buf , A0_32_0);
        vst1q_f32(A10_buf, A2_32_0);

        x9 -= NEON_LEN; res_0 = vld1q_f32(x9); __builtin_prefetch(x9 - NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0);
        // 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x9[3] = vgetq_lane_f32(res_0, 3) - (A8_buf[3] * x9[ 2] + A10_buf[3] * x9[4]);
        x9[2] = vgetq_lane_f32(res_0, 2) - (A8_buf[2] * x9[ 1] + A10_buf[2] * x9[3]);
        x9[1] = vgetq_lane_f32(res_0, 1) - (A8_buf[1] * x9[ 0] + A10_buf[1] * x9[2]);
        x9[0] = vgetq_lane_f32(res_0, 0) - (A8_buf[0] * x9[-1] + A10_buf[0] * x9[1]);
    }
    A0_3 -= 4; A4_7 -= 4; A8_10 -= 3; A11_14 -= 4; A15_18 -= 4;
    x0 -= min_nk; x1 -= min_nk; x2 -= min_nk; x3 -= min_nk;
    x4 -= min_nk; x5 -= min_nk; x6 -= min_nk; x7 -= min_nk;
    x9 -= min_nk;
    x11-= min_nk; x12-= min_nk; x13-= min_nk; x14-= min_nk;
    x15-= min_nk; x16-= min_nk; x17-= min_gk; x18-= min_nk; b9 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A8_10[1];
        float tmp = 
        + A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        + A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        + A8_10[0]* x9[k-1] + A8_10[2] * x9[k+1]
        + A11_14[0] * x11[k] + A11_14[1] * x12[k] + A11_14[2] * x13[k] + A11_14[3] * x14[k]
        + A15_18[0] * x15[k] + A15_18[1] * x16[k] + A15_18[2] * x17[k] + A15_18[3] * x18[k];
        tmp = b9[k] - tmp;// b - L*x_{k+1}
        x9[k] *= (1.0 - weight);
        x9[k] += weight * tmp / diag_val;
        A0_3 -= 4; A4_7 -= 4; A8_10 -= 3; A11_14 -= 4; A15_18 -= 4;
    }
}


// =================================== LGS =============================
void inline SOA_line_forward_zero_3d19_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[2], const float * b9, const float * x9, float * rhs)
{// (0,1,2,3) (4,5,6,7)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1];
    const float * x2 = x9 - vec_ki_size, * x6 = x9 - vec_k_size;
    const float * x0 = x2 - vec_k_size , * x4 = x2 + vec_k_size,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; tmp_1   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + GROUP_LEN, 0);
        // A0 ~ A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A4 ~ A7
        A0_3_16 = vld4q_f16(A4_7); A4_7 += GROUP_LEN * 4; __builtin_prefetch(A4_7 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; vst1q_f32(rhs, tmp_1); rhs += NEON_LEN; __builtin_prefetch(rhs + GROUP_LEN, 1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + NEON_LEN, 0);
        // A0 ~ A3
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A4 ~ A7
        A0_3_16 = vld4_f16(A4_7); A4_7 += NEON_LEN * 4; __builtin_prefetch(A4_7 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; __builtin_prefetch(rhs + NEON_LEN, 1);
    }
    for (k = 0; k < num - max_nk; k++) {
        float tmp = 
            + A0_3[0]*x0[k] + A0_3[1]*x1[k] + A0_3[2]*x2[k] + A0_3[3]*x3[k]
            + A4_7[0]*x4[k] + A4_7[1]*x5[k] + A4_7[2]*x6[k] + A4_7[3]*x7[k];
        rhs[k] = b9[k] - tmp;
        A0_3 += 4; A4_7 += 4;
    }
}

void inline SOA_line_forward_ALL_3d19_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[4], const float * b9, const float * x9, float * rhs)
{// (0,1,2,3) (4,5,6,7) || (11,12,13,14) (15,16,17,18)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A11_14 = Diags[2], * A15_18 = Diags[3];
    const float * x2 = x9 - vec_ki_size, * x6 = x9 - vec_k_size,
                * x16= x9 + vec_ki_size, * x12= x9 + vec_k_size;
    const float * x0 = x2 - vec_k_size , * x4 = x2 + vec_k_size,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1,
                * x14= x16- vec_k_size , * x18= x16 + vec_k_size,
                * x15= x16- 1, * x17= x16+ 1, * x11= x12- 1, * x13= x12+ 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; tmp_1   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + GROUP_LEN, 0);
        // A0 ~ A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A4 ~ A7
        A0_3_16 = vld4q_f16(A4_7); A4_7 += GROUP_LEN * 4; __builtin_prefetch(A4_7 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A11 ~ A14
        A0_3_16 = vld4q_f16(A11_14); A11_14 += GROUP_LEN * 4; __builtin_prefetch(A11_14 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x0_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x12); x12 += NEON_LEN; x1_32_1 = vld1q_f32(x12); x12 += NEON_LEN; __builtin_prefetch(x12 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x13); x13 += NEON_LEN; x2_32_1 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x14); x14 += NEON_LEN; x3_32_1 = vld1q_f32(x14); x14 += NEON_LEN; __builtin_prefetch(x14 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A15~A18
        A0_3_16 = vld4q_f16(A15_18); A15_18 += GROUP_LEN * 4; __builtin_prefetch(A15_18 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x15); x15 += NEON_LEN; x0_32_1 = vld1q_f32(x15); x15 += NEON_LEN; __builtin_prefetch(x15 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x16); x16 += NEON_LEN; x1_32_1 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x17); x17 += NEON_LEN; x2_32_1 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x18); x18 += NEON_LEN; x3_32_1 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; vst1q_f32(rhs, tmp_1); rhs += NEON_LEN; __builtin_prefetch(rhs + GROUP_LEN, 1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9 + NEON_LEN, 0);
        // A0 ~ A3
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A4 ~ A7
        A0_3_16 = vld4_f16(A4_7); A4_7 += NEON_LEN * 4; __builtin_prefetch(A4_7 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A11 ~ A14
        A0_3_16 = vld4_f16(A11_14); A11_14 += NEON_LEN * 4; __builtin_prefetch(A11_14 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x12); x12 += NEON_LEN; __builtin_prefetch(x12 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x14); x14 += NEON_LEN; __builtin_prefetch(x14 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15 ~ A18
        A0_3_16 = vld4_f16(A15_18); A15_18 += NEON_LEN * 4; __builtin_prefetch(A15_18 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x15); x15 += NEON_LEN; __builtin_prefetch(x15 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; __builtin_prefetch(rhs + NEON_LEN, 1);
    }
    for (k = 0; k < num - max_nk; k++) {
        float tmp = 
        + A0_3[0]*x0[k] + A0_3[1]*x1[k] + A0_3[2]*x2[k] + A0_3[3]*x3[k]
        + A4_7[0]*x4[k] + A4_7[1]*x5[k] + A4_7[2]*x6[k] + A4_7[3]*x7[k]
        + A11_14[0]*x11[k] + A11_14[1]*x12[k] + A11_14[2]*x13[k] + A11_14[3]*x14[k]
        + A15_18[0]*x15[k] + A15_18[1]*x16[k] + A15_18[2]*x17[k] + A15_18[3]*x18[k];
        rhs[k] = b9[k] - tmp;
        A0_3 += 4; A4_7 += 4; A11_14 += 4; A15_18 += 4;
    }
}

void inline SOA_line_backward_ALL_3d19_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[4], const float * b9, const float * x9, float * rhs)
{// (0,1,2,3) (4,5,6,7) || (11,12,13,14) (15,16,17,18)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A11_14 = Diags[2], * A15_18 = Diags[3];
    const float * x2 = x9 - vec_ki_size, * x6 = x9 - vec_k_size,
                * x16= x9 + vec_ki_size, * x12= x9 + vec_k_size;
    const float * x0 = x2 - vec_k_size, * x4 = x2 + vec_k_size,
                * x14= x16- vec_k_size, * x18= x16+ vec_k_size,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1,
                * x11= x12- 1, * x13= x12+ 1, * x15= x16- 1, * x17= x16+ 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        b9 -= NEON_LEN; tmp_1 = vld1q_f32(b9); b9 -= NEON_LEN; tmp_0 = vld1q_f32(b9); __builtin_prefetch(b9 - GROUP_LEN, 0);
        // A0~A3
        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0 -= NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - GROUP_LEN, 0);
        x1 -= NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - GROUP_LEN, 0);
        x2 -= NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 -= NEON_LEN; x2_32_0 = vld1q_f32(x2); __builtin_prefetch(x2 - GROUP_LEN, 0);
        x3 -= NEON_LEN; x3_32_1 = vld1q_f32(x3); x3 -= NEON_LEN; x3_32_0 = vld1q_f32(x3); __builtin_prefetch(x3 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); 
        // A4~A7
        A4_7 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A4_7); __builtin_prefetch(A4_7 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x4 -= NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 -= NEON_LEN; x0_32_0 = vld1q_f32(x4); __builtin_prefetch(x4 - GROUP_LEN, 0);
        x5 -= NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - GROUP_LEN, 0);
        x6 -= NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x2_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - GROUP_LEN, 0);
        x7 -= NEON_LEN; x3_32_1 = vld1q_f32(x7); x7 -= NEON_LEN; x3_32_0 = vld1q_f32(x7); __builtin_prefetch(x7 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A11~A14
        A11_14 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A11_14); __builtin_prefetch(A11_14 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x11 -= NEON_LEN; x0_32_1 = vld1q_f32(x11); x11 -= NEON_LEN; x0_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - GROUP_LEN, 0);
        x12 -= NEON_LEN; x1_32_1 = vld1q_f32(x12); x12 -= NEON_LEN; x1_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - GROUP_LEN, 0);
        x13 -= NEON_LEN; x2_32_1 = vld1q_f32(x13); x13 -= NEON_LEN; x2_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - GROUP_LEN, 0);
        x14 -= NEON_LEN; x3_32_1 = vld1q_f32(x14); x14 -= NEON_LEN; x3_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15~A18
        A15_18 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A15_18); __builtin_prefetch(A15_18 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x15 -= NEON_LEN; x0_32_1 = vld1q_f32(x15); x15 -= NEON_LEN; x0_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - GROUP_LEN, 0);
        x16 -= NEON_LEN; x1_32_1 = vld1q_f32(x16); x16 -= NEON_LEN; x1_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - GROUP_LEN, 0);
        x17 -= NEON_LEN; x2_32_1 = vld1q_f32(x17); x17 -= NEON_LEN; x2_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - GROUP_LEN, 0);
        x18 -= NEON_LEN; x3_32_1 = vld1q_f32(x18); x18 -= NEON_LEN; x3_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        rhs -= NEON_LEN; vst1q_f32(rhs, tmp_1); rhs -= NEON_LEN; vst1q_f32(rhs, tmp_0); __builtin_prefetch(rhs - GROUP_LEN, 1);
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        b9 -= NEON_LEN; tmp_0 = vld1q_f32(b9); __builtin_prefetch(b9 - NEON_LEN, 0);
        // A0~A3
        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - NEON_LEN, 0);
        x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - NEON_LEN, 0);
        x2 -= NEON_LEN; x2_32_0 = vld1q_f32(x2); __builtin_prefetch(x2 - NEON_LEN, 0);
        x3 -= NEON_LEN; x3_32_0 = vld1q_f32(x3); __builtin_prefetch(x3 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0, x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0, x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0, x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0, x3_32_0);
        // A4~A7
        A4_7 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A4_7); __builtin_prefetch(A4_7 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x4 -= NEON_LEN; x0_32_0 = vld1q_f32(x4); __builtin_prefetch(x4 - NEON_LEN, 0);
        x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - NEON_LEN, 0);
        x6 -= NEON_LEN; x2_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - NEON_LEN, 0);
        x7 -= NEON_LEN; x3_32_0 = vld1q_f32(x7); __builtin_prefetch(x7 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A11~A14
        A11_14 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A11_14); __builtin_prefetch(A11_14 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x11 -= NEON_LEN; x0_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - NEON_LEN, 0);
        x12 -= NEON_LEN; x1_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - NEON_LEN, 0);
        x13 -= NEON_LEN; x2_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - NEON_LEN, 0);
        x14 -= NEON_LEN; x3_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15~A18
        A15_18 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A15_18); __builtin_prefetch(A15_18 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x15 -= NEON_LEN; x0_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - NEON_LEN, 0);
        x16 -= NEON_LEN; x1_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - NEON_LEN, 0);
        x17 -= NEON_LEN; x2_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - NEON_LEN, 0);
        x18 -= NEON_LEN; x3_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        rhs -= NEON_LEN; vst1q_f32(rhs, tmp_0); __builtin_prefetch(rhs - NEON_LEN, 1);
    }
    A0_3 -= 4; A4_7 -= 4; A11_14 -= 4; A15_18 -= 4;
    x0 -= min_nk; x1 -= min_nk; x2 -= min_nk; x3 -= min_nk;
    x4 -= min_nk; x5 -= min_nk; x6 -= min_nk; x7 -= min_nk;
    x11-= min_nk; x12-= min_nk; x13-= min_nk; x14-= min_nk;
    x15-= min_nk; x16-= min_nk; x17-= min_gk; x18-= min_nk; b9 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float tmp = 
        + A0_3[0]*x0[k] + A0_3[1]*x1[k] + A0_3[2]*x2[k] + A0_3[3]*x3[k]
        + A4_7[0]*x4[k] + A4_7[1]*x5[k] + A4_7[2]*x6[k] + A4_7[3]*x7[k]
        
        + A11_14[0]*x11[k] + A11_14[1]*x12[k] + A11_14[2]*x13[k] + A11_14[3]*x14[k]
        + A15_18[0]*x15[k] + A15_18[1]*x16[k] + A15_18[2]*x17[k] + A15_18[3]*x18[k];
        rhs[k] = b9[k] - tmp;// b - L*x_{k+1}
        A0_3 -= 4; A4_7 -= 4; A11_14 -= 4; A15_18 -= 4;
    }
}

// =================================== BILU ==================================

void inline SOA_ilu_forward_zero_3d19_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[3], const float * b9, float * x9)
{// L(0,1,2) (3,4,5) (6,7,8)
    const __fp16* A0_2 = Diags[0], * A3_5 = Diags[1], * A6_8 = Diags[2];
    const float * x2 = x9 - dim_1 * dim_2, * x6 = x9 - dim_2;
    const float * x0 = x2 - dim_2, * x4 = x2 + dim_2,
                * x1 = x2 - 1, * x3 = x2 + 1, * x5 = x6 - 1, * x7 = x6 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = dim_2 & (~(GROUP_LEN - 1)), max_nk = dim_2 & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x3_t A0_2_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, A0_32_1, A1_32_1, A2_32_1;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; tmp_1   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9, 0);
        // A0 ~ A2
        A0_2_16 = vld3q_f16(A0_2); A0_2 += GROUP_LEN * 3; __builtin_prefetch(A0_2 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_2_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_2_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_2_16.val[2]);// 对应x2
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        // A3 ~ A5
        A0_2_16 = vld3q_f16(A3_5); A3_5 += GROUP_LEN * 3; __builtin_prefetch(A3_5 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_2_16.val[0]);// 对应x3
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_2_16.val[1]);// 对应x4
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_2_16.val[2]);// 对应x5
        x0_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x0_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 0);
        x1_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x1_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x2_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x2_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        // A6 ~ A8
        A0_2_16 = vld3q_f16(A6_8); A6_8 += GROUP_LEN * 3; __builtin_prefetch(A6_8 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_2_16.val[0]);// 对应x6
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_2_16.val[1]);// 对应x7
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_2_16.val[2]);// 对应本柱的 x8
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        x1_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x1_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        vst1q_f32(x9, tmp_0);
        x9[0] = x9[0] - vgetq_lane_f32(A2_32_0, 0) * x9[-1];
        x9[1] = x9[1] - vgetq_lane_f32(A2_32_0, 1) * x9[ 0];
        x9[2] = x9[2] - vgetq_lane_f32(A2_32_0, 2) * x9[ 1];
        x9[3] = x9[3] - vgetq_lane_f32(A2_32_0, 3) * x9[ 2];
        x9 += NEON_LEN;
        vst1q_f32(x9, tmp_1);
        x9[0] = x9[0] - vgetq_lane_f32(A2_32_1, 0) * x9[-1];
        x9[1] = x9[1] - vgetq_lane_f32(A2_32_1, 1) * x9[ 0];
        x9[2] = x9[2] - vgetq_lane_f32(A2_32_1, 2) * x9[ 1];
        x9[3] = x9[3] - vgetq_lane_f32(A2_32_1, 3) * x9[ 2];
        x9 += NEON_LEN; __builtin_prefetch(x9,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x3_t A0_2_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b9); b9 += NEON_LEN; __builtin_prefetch(b9, 0);
        // A0 ~ A2
        A0_2_16 = vld3_f16(A0_2); A0_2 += NEON_LEN * 3; __builtin_prefetch(A0_2 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_2_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(A0_2_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(A0_2_16.val[2]);// 对应x2
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        // A3 ~ A5
        A0_2_16 = vld3_f16(A3_5); A3_5 += NEON_LEN * 3; __builtin_prefetch(A3_5 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_2_16.val[0]);// 对应x3
        A1_32_0 = vcvt_f32_f16(A0_2_16.val[1]);// 对应x4
        A2_32_0 = vcvt_f32_f16(A0_2_16.val[2]);// 对应x5
        x0_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 0);
        x1_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x2_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        // A6 ~ A8
        A0_2_16 = vld3_f16(A6_8); A6_8 += NEON_LEN * 3; __builtin_prefetch(A6_8 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_2_16.val[0]);// 对应x6
        A1_32_0 = vcvt_f32_f16(A0_2_16.val[1]);// 对应x7
        A2_32_0 = vcvt_f32_f16(A0_2_16.val[2]);// 对应本柱的 x8
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        x1_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        vst1q_f32(x9, tmp_0);
        x9[0] = x9[0] - vgetq_lane_f32(A2_32_0, 0) * x9[-1];
        x9[1] = x9[1] - vgetq_lane_f32(A2_32_0, 1) * x9[ 0];
        x9[2] = x9[2] - vgetq_lane_f32(A2_32_0, 2) * x9[ 1];
        x9[3] = x9[3] - vgetq_lane_f32(A2_32_0, 3) * x9[ 2];
        x9 += NEON_LEN; __builtin_prefetch(x9,1);
    }
    for (k = 0; k < dim_2 - max_nk; k++) {// 做完剩下的元素
        float tmp = 
        + A0_2[0]*x0[k] + A0_2[1]*x1[k] + A0_2[2]*x2[k]
        + A3_5[0]*x3[k] + A3_5[1]*x4[k] + A3_5[2]*x5[k]
        + A6_8[0]*x6[k] + A6_8[1]*x7[k] + A6_8[2]*x9[k-1];
        x9[k] = b9[k] - tmp;// b - L*x_{k+1}
        A0_2 += 3; A3_5 += 3; A6_8 += 3;
    }
}

void inline SOA_ilu_backward_zero_3d19_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[3], const float * b9, float * x9)
{// U(0,1,2) (3,4,5) (6,7,8,9)
    const __fp16* A0_2 = Diags[0], * A3_5 = Diags[1], * A6_9 = Diags[2];
    const float * x16= x9 + dim_1 * dim_2, * x12 = x9 + dim_2;
    const float * x14= x16 - dim_2, * x18 = x16 + dim_2,
                * x11= x12 - 1, * x13= x12 + 1, * x15= x16 - 1, * x17= x16 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    float32x4_t vones = vdupq_n_f32(1.0);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = dim_2, min_gk = dim_2 & (GROUP_LEN - 1), min_nk = dim_2 & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        b9 -= NEON_LEN; tmp_1 = vld1q_f32(b9)  ; b9 -= NEON_LEN; tmp_0 = vld1q_f32(b9)  ; __builtin_prefetch(b9 - GROUP_LEN, 0);
        A6_9 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A6_9); __builtin_prefetch(A6_9 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 对应x15
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应x16
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应x17
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// 对应x18
        x15-= NEON_LEN; x0_32_1 = vld1q_f32(x15); x15 -= NEON_LEN; x0_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - GROUP_LEN, 0);
        x16-= NEON_LEN; x1_32_1 = vld1q_f32(x16); x16 -= NEON_LEN; x1_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - GROUP_LEN, 0);
        x17-= NEON_LEN; x2_32_1 = vld1q_f32(x17); x17 -= NEON_LEN; x2_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - GROUP_LEN, 0);
        x18-= NEON_LEN; x3_32_1 = vld1q_f32(x18); x18 -= NEON_LEN; x3_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        float16x8x3_t A3_5_16;
        A3_5 -= GROUP_LEN * 3;
        A3_5_16 = vld3q_f16(A3_5); __builtin_prefetch(A3_5 - GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A3_5_16.val[0]);// 对应x12
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A3_5_16.val[1]);// 对应x13
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A3_5_16.val[2]);// 对应x14
        x12-= NEON_LEN; x0_32_1 = vld1q_f32(x12); x12 -= NEON_LEN; x0_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - GROUP_LEN, 0);
        x13-= NEON_LEN; x1_32_1 = vld1q_f32(x13); x13 -= NEON_LEN; x1_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - GROUP_LEN, 0);
        x14-= NEON_LEN; x2_32_1 = vld1q_f32(x14); x14 -= NEON_LEN; x2_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        A0_2 -= GROUP_LEN * 3;
        A3_5_16 = vld3q_f16(A0_2); __builtin_prefetch(A0_2 - GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A3_5_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A3_5_16.val[1]);// 对应本柱的 x10
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A3_5_16.val[2]);// 对应x11
        A0_32_0 = vdivq_f32(vones, A0_32_0)                 ; A0_32_1 = vdivq_f32(vones, A0_32_1);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0)               ; A1_32_1 = vmulq_f32(A1_32_1, A0_32_1);// A10/A9
        float A10_buf[GROUP_LEN];// 暂存本柱的 A10/A9
        vst1q_f32(A10_buf, A1_32_0)                         ; vst1q_f32(A10_buf + NEON_LEN, A1_32_1);
        x11-= NEON_LEN; x2_32_1 = vld1q_f32(x11); x11 -= NEON_LEN; x2_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_1 = vmulq_f32(tmp_1, A0_32_1)                   ; tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x9 -= GROUP_LEN; __builtin_prefetch(x9 - GROUP_LEN, 1);
        x9[7] = vgetq_lane_f32(tmp_1, 3) - A10_buf[7] * x9[8];
        x9[6] = vgetq_lane_f32(tmp_1, 2) - A10_buf[6] * x9[7];
        x9[5] = vgetq_lane_f32(tmp_1, 1) - A10_buf[5] * x9[6];
        x9[4] = vgetq_lane_f32(tmp_1, 0) - A10_buf[4] * x9[5];
        x9[3] = vgetq_lane_f32(tmp_0, 3) - A10_buf[3] * x9[4];
        x9[2] = vgetq_lane_f32(tmp_0, 2) - A10_buf[2] * x9[3];
        x9[1] = vgetq_lane_f32(tmp_0, 1) - A10_buf[1] * x9[2];
        x9[0] = vgetq_lane_f32(tmp_0, 0) - A10_buf[0] * x9[1];
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        b9 -= NEON_LEN; tmp_0 = vld1q_f32(b9)  ; __builtin_prefetch(b9 - NEON_LEN, 0);
        A6_9 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A6_9); __builtin_prefetch(A6_9 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 对应x15
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应x16
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应x17
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// 对应x18
        x15 -= NEON_LEN; x0_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - NEON_LEN, 0);
        x16 -= NEON_LEN; x1_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - NEON_LEN, 0);
        x17 -= NEON_LEN; x2_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - NEON_LEN, 0);
        x18 -= NEON_LEN; x3_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        float16x4x3_t A3_5_16;
        A3_5 -= NEON_LEN * 3;
        A3_5_16 = vld3_f16(A3_5); __builtin_prefetch(A3_5 - NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A3_5_16.val[0]);// 对应x12
        A1_32_0 = vcvt_f32_f16(A3_5_16.val[1]);// 对应x13
        A2_32_0 = vcvt_f32_f16(A3_5_16.val[2]);// 对应x14
        x12 -= NEON_LEN; x0_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - NEON_LEN, 0);
        x13 -= NEON_LEN; x1_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - NEON_LEN, 0);
        x14 -= NEON_LEN; x2_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        A0_2 -= NEON_LEN * 3;
        A3_5_16 = vld3_f16(A0_2); __builtin_prefetch(A0_2 - NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A3_5_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(A3_5_16.val[1]);// 对应本柱的 x10
        A2_32_0 = vcvt_f32_f16(A3_5_16.val[2]);// 对应x11
        A0_32_0 = vdivq_f32(vones, A0_32_0);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0);// A10/A9
        float A10_buf[NEON_LEN];// 暂存本柱的 A10/A9
        vst1q_f32(A10_buf, A1_32_0);
        x11 -= NEON_LEN; x2_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x9 -= NEON_LEN; __builtin_prefetch(x9 - NEON_LEN, 1);
        x9[3] = vgetq_lane_f32(tmp_0, 3) - A10_buf[3] * x9[4];
        x9[2] = vgetq_lane_f32(tmp_0, 2) - A10_buf[2] * x9[3];
        x9[1] = vgetq_lane_f32(tmp_0, 1) - A10_buf[1] * x9[2];
        x9[0] = vgetq_lane_f32(tmp_0, 0) - A10_buf[0] * x9[1];
    }
    A0_2 -= 3; A3_5 -= 3; A6_9 -= 4;
    x9 -= min_nk; b9 -= min_nk;
    x11 -= min_nk; x12 -= min_nk; x13 -= min_nk; x14 -= min_nk;
    x15 -= min_nk; x16 -= min_nk; x17 -= min_nk; x18 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A0_2[0];
        float tmp = 
            +                  A0_2[1]*x9[k+1] + A0_2[2]*x11[k]
            + A3_5[0]*x12[k] + A3_5[1]*x13[k] + A3_5[2]*x14[k]
            + A6_9[0]*x15[k] + A6_9[1]*x16[k] + A6_9[2]*x17[k] + A6_9[3]*x18[k];
        x9[k] = (b9[k] - tmp) / diag_val;
        A0_2 -= 3; A3_5 -= 3; A6_9 -= 4;
    }
}

#undef NEON_LEN
#undef GROUP_LEN
#endif