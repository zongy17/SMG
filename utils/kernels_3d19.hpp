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

template<typename idx_t, typename data_t>
void inline AOS_spmv_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const data_t * x_jik, data_t * y_jik, const data_t * dummy)
{
    const data_t * x_jNi = x_jik - vec_ki_size, * x_jPi = x_jik + vec_ki_size;
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

template<typename idx_t, typename data_t>
void inline SOA_spmv_3d19(const idx_t num,
    const idx_t vec_k_size , const idx_t vec_ki_size,
    const data_t * A_jik[19], const data_t * x_jik, data_t * y_jik)
{
    // 要打桩查一下是否生成了向量化代码！！！！
    const data_t * x2 = x_jik - vec_ki_size, * x16 = x_jik + vec_ki_size,
                * x6 = x_jik - vec_k_size , * x12 = x_jik + vec_k_size,
                * x8 = x_jik - 1, * x10 = x_jik + 1;
    const data_t * x0 = x2 - vec_k_size, * x4 = x2 + vec_k_size,
                * x3 = x2 + 1, * x1 = x2 - 1,
                * x14= x16- vec_k_size, * x18= x16+ vec_k_size,
                * x17= x16+ 1, * x15= x16- 1,
                * x5 = x8 - vec_k_size, * x11 = x8 + vec_k_size,
                * x7 = x10- vec_k_size, * x13 = x10+ vec_k_size;
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3],
                * A4 = A_jik[4], * A5 = A_jik[5], * A6 = A_jik[6], * A7 = A_jik[7],
                * A8 = A_jik[8], * A9 = A_jik[9], * A10= A_jik[10],* A11= A_jik[11],
                * A12= A_jik[12],* A13= A_jik[13],* A14= A_jik[14],* A15= A_jik[15],
                * A16= A_jik[16],* A17= A_jik[17],* A18= A_jik[18]; 
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k]= A0[k] * x0[k] + A1[k] * x1[k] + A2[k] * x2[k] + A3[k] * x3[k]
                + A4[k] * x4[k] + A5[k] * x5[k] + A6[k] * x6[k] + A7[k] * x7[k]
                + A8[k] *  x8[k]  + A9[k] * x_jik[k]+ A10[k] * x10[k]
                + A11[k] * x11[k] + A12[k] * x12[k] + A13[k] * x13[k]
                + A14[k] * x14[k] + A15[k] * x15[k] + A16[k] * x16[k]
                + A17[k] * x17[k] + A18[k] * x18[k];
    }
}

// =================================== PGS =============================

// =================================== LGS =============================

template<typename idx_t, typename data_t>
void inline AOS_line_forward_zero_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * dummy, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t* x_jNiZ = x_jik  - vec_ki_size,
                * x_jZiN = x_jik  - vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp = 
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

template<typename idx_t, typename data_t>
void inline AOS_line_backward_zero_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * dummy, const data_t * U_jik, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t* x_jPiZ  = x_jik   + vec_ki_size,
                * x_jZiP  = x_jik   + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =
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

template<typename idx_t, typename data_t>
void inline AOS_line_ALL_3d19(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t* x_jNiZ  = x_jik - vec_ki_size, * x_jZiN = x_jik - vec_k_size,
                * x_jPiZ  = x_jik + vec_ki_size, * x_jZiP = x_jik + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =
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
template<typename idx_t, typename data_t>
void inline AOS_ilu_forward_zero_3d19(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t* x_jNiZ = x_jik  - dim_1 * dim_2,
                * x_jZiN = x_jik  - dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        data_t tmp = 
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

template<typename idx_t, typename data_t>
void inline AOS_ilu_backward_zero_3d19(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t* x_jPiZ  = x_jik   + dim_1 * dim_2,
                * x_jZiP  = x_jik   + dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        data_t para = U_jik[0];
        data_t tmp = 
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

#endif