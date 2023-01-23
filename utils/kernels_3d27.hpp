#ifndef SMG_KERNELS_3D27_HPP
#define SMG_KERNELS_3D27_HPP

// ===================================================================
// ========================  3d27  kernels  ==========================
// ===================================================================
/*  
                  /20--- 23------26
                 11|    14      17|
               2---|---5-------8  |
               |   |           |  |
               |   19----22----|-25
   z   y       | 10|    13     |16|
   ^  ^        1---|-- 4 ------7  |
   | /         |   |           |  |
   |/          |   18----21----|-24 
   O-------> x | 9      12     |15 
               0/------3-------6/

        */

// =================================== SPMV =============================

template<typename idx_t, typename data_t>
void inline AOS_spmv_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const data_t * x_jik, data_t * y_jik, const data_t * dummy)
{
    const data_t * x_jNi = x_jik - vec_ki_size, * x_jPi = x_jik + vec_ki_size;
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k] = 
            + A_jik[0] * x_jNi[- vec_k_size + k - 1]
            + A_jik[1] * x_jNi[- vec_k_size + k    ]
            + A_jik[2] * x_jNi[- vec_k_size + k + 1]
            + A_jik[3] * x_jNi[               k - 1]
            + A_jik[4] * x_jNi[               k    ]
            + A_jik[5] * x_jNi[               k + 1]
            + A_jik[6] * x_jNi[  vec_k_size + k - 1]
            + A_jik[7] * x_jNi[  vec_k_size + k    ]
            + A_jik[8] * x_jNi[  vec_k_size + k + 1]

            + A_jik[ 9] * x_jik[- vec_k_size + k - 1]
            + A_jik[10] * x_jik[- vec_k_size + k    ]
            + A_jik[11] * x_jik[- vec_k_size + k + 1]
            + A_jik[12] * x_jik[               k - 1]
            + A_jik[13] * x_jik[               k    ]
            + A_jik[14] * x_jik[               k + 1]
            + A_jik[15] * x_jik[  vec_k_size + k - 1]
            + A_jik[16] * x_jik[  vec_k_size + k    ]
            + A_jik[17] * x_jik[  vec_k_size + k + 1]

            + A_jik[18] * x_jPi[- vec_k_size + k - 1]
            + A_jik[19] * x_jPi[- vec_k_size + k    ]
            + A_jik[20] * x_jPi[- vec_k_size + k + 1]
            + A_jik[21] * x_jPi[               k - 1]
            + A_jik[22] * x_jPi[               k    ]
            + A_jik[23] * x_jPi[               k + 1]
            + A_jik[24] * x_jPi[  vec_k_size + k - 1]
            + A_jik[25] * x_jPi[  vec_k_size + k    ]
            + A_jik[26] * x_jPi[  vec_k_size + k + 1];

        A_jik += 27;// move the ptr
    }
}

template<typename idx_t, typename data_t>
void inline SOA_spmv_3d27(const idx_t num,
    const idx_t vec_k_size , const idx_t vec_ki_size,
    const data_t * A_jik[27], const data_t * x_jik, data_t * y_jik)
{
    // 要打桩查一下是否生成了向量化代码！！！！
    const data_t * x4 = x_jik - vec_ki_size, * x22 = x_jik + vec_ki_size,
                * x10= x_jik - vec_k_size , * x16 = x_jik + vec_k_size,
                * x12= x_jik - 1, * x14 = x_jik + 1;
    const data_t * x1 = x4 - vec_k_size, * x7 = x4 + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1,
                * x19= x22- vec_k_size, * x25= x22+ vec_k_size,
                * x21= x22- 1, * x23= x22+ 1;
    const data_t * x9 = x10- 1, * x11= x10+ 1,
                * x15= x16- 1, * x17= x16+ 1,
                * x0 = x1 - 1, * x2 = x1 + 1,
                * x6 = x7 - 1, * x8 = x7 + 1,
                * x18= x19- 1, * x20= x19+ 1,
                * x24= x25- 1, * x26= x25+ 1;
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3],
                * A4 = A_jik[4], * A5 = A_jik[5], * A6 = A_jik[6], * A7 = A_jik[7],
                * A8 = A_jik[8], * A9 = A_jik[9], * A10= A_jik[10],* A11= A_jik[11],
                * A12= A_jik[12],* A13= A_jik[13],* A14= A_jik[14],* A15= A_jik[15],
                * A16= A_jik[16],* A17= A_jik[17],* A18= A_jik[18],* A19= A_jik[19],
                * A20= A_jik[20],* A21= A_jik[21],* A22= A_jik[22],* A23= A_jik[23],
                * A24= A_jik[24],* A25= A_jik[25],* A26= A_jik[26];
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k]= A0[k] * x0[k] + A1[k] * x1[k] + A2[k] * x2[k] + A3[k] * x3[k] + A4[k] * x4[k]
                + A5[k] * x5[k] + A6[k] * x6[k] + A7[k] * x7[k] + A8[k] * x8[k] + A9[k] * x9[k]
                + A10[k] * x10[k] + A11[k] * x11[k] + A12[k] * x12[k]
                + A13[k] * x_jik[k]
                + A14[k] * x14[k] + A15[k] * x15[k] + A16[k] * x16[k]
                + A17[k] * x17[k] + A18[k] * x18[k] + A19[k] * x19[k]
                + A20[k] * x20[k] + A21[k] * x21[k] + A22[k] * x22[k]
                + A23[k] * x23[k] + A24[k] * x24[k] + A25[k] * x25[k]
                + A26[k] * x26[k];
    }
}

// =================================== PGS =============================

// =================================== LGS =============================

template<typename idx_t, typename data_t>
void inline AOS_line_forward_zero_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * dummy, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jNiN = x_jik - vec_ki_size - vec_k_size,
                * x_jNiP = x_jik - vec_ki_size + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp = 
            + L_jik[ 0] * x_jNiN[k-1]
            + L_jik[ 1] * x_jNiN[k  ]
            + L_jik[ 2] * x_jNiN[k+1]
            + L_jik[ 3] * x_jNiZ[k-1]
            + L_jik[ 4] * x_jNiZ[k  ]
            + L_jik[ 5] * x_jNiZ[k+1]
            + L_jik[ 6] * x_jNiP[k-1]
            + L_jik[ 7] * x_jNiP[k  ]
            + L_jik[ 8] * x_jNiP[k+1]
            + L_jik[ 9] * x_jZiN[k-1]
            + L_jik[10] * x_jZiN[k  ]
            + L_jik[11] * x_jZiN[k+1];// L * x_{k+1}
        rhs[k] = b_jik[k] - tmp;// b - L*x_{k+1}
        L_jik += 12;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t>
void inline AOS_line_backward_zero_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * dummy, const data_t * U_jik, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t* x_jPiZ = x_jik + vec_ki_size,
                * x_jZiP = x_jik               + vec_k_size,
                * x_jPiN = x_jik + vec_ki_size - vec_k_size,
                * x_jPiP = x_jik + vec_ki_size + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =
            + U_jik[ 0] * x_jZiP[k-1]
            + U_jik[ 1] * x_jZiP[k  ]
            + U_jik[ 2] * x_jZiP[k+1]
            + U_jik[ 3] * x_jPiN[k-1]
            + U_jik[ 4] * x_jPiN[k  ]
            + U_jik[ 5] * x_jPiN[k+1]
            + U_jik[ 6] * x_jPiZ[k-1]
            + U_jik[ 7] * x_jPiZ[k  ]
            + U_jik[ 8] * x_jPiZ[k+1]
            + U_jik[ 9] * x_jPiP[k-1]
            + U_jik[10] * x_jPiP[k  ]
            + U_jik[11] * x_jPiP[k+1];// U*x_{k+1}
        rhs[k] = b_jik[k] - tmp;
        U_jik += 12;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_line_ALL_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jPiZ = x_jik + vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jZiP = x_jik               + vec_k_size;
    const data_t* x_jNiN = x_jNiZ - vec_k_size, * x_jPiN = x_jPiZ - vec_k_size,
                * x_jNiP = x_jNiZ + vec_k_size, * x_jPiP = x_jPiZ + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =
           +  L_jik[ 0] * x_jNiN[k-1]
            + L_jik[ 1] * x_jNiN[k  ]
            + L_jik[ 2] * x_jNiN[k+1]
            + L_jik[ 3] * x_jNiZ[k-1]
            + L_jik[ 4] * x_jNiZ[k  ]
            + L_jik[ 5] * x_jNiZ[k+1]
            + L_jik[ 6] * x_jNiP[k-1]
            + L_jik[ 7] * x_jNiP[k  ]
            + L_jik[ 8] * x_jNiP[k+1]
            + L_jik[ 9] * x_jZiN[k-1]
            + L_jik[10] * x_jZiN[k  ]
            + L_jik[11] * x_jZiN[k+1]
            + U_jik[ 0] * x_jZiP[k-1]
            + U_jik[ 1] * x_jZiP[k  ]
            + U_jik[ 2] * x_jZiP[k+1]
            + U_jik[ 3] * x_jPiN[k-1]
            + U_jik[ 4] * x_jPiN[k  ]
            + U_jik[ 5] * x_jPiN[k+1]
            + U_jik[ 6] * x_jPiZ[k-1]
            + U_jik[ 7] * x_jPiZ[k  ]
            + U_jik[ 8] * x_jPiZ[k+1]
            + U_jik[ 9] * x_jPiP[k-1]
            + U_jik[10] * x_jPiP[k  ]
            + U_jik[11] * x_jPiP[k+1];
        rhs[k] = b_jik[k] - tmp;
        L_jik += 12;
        U_jik += 12;
    }
}

// =================================== BILU ==================================
template<typename idx_t, typename data_t>
void inline AOS_ilu_forward_zero_3d27(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t* x_jNiZ = x_jik -  dim_1      * dim_2,
                * x_jZiN = x_jik -               dim_2,
                * x_jNiN = x_jik - (dim_1 + 1) * dim_2,
                * x_jNiP = x_jik - (dim_1 - 1) * dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        data_t tmp = 
            + L_jik[ 0] * x_jNiN[k-1]
            + L_jik[ 1] * x_jNiN[k  ]
            + L_jik[ 2] * x_jNiN[k+1]
            + L_jik[ 3] * x_jNiZ[k-1]
            + L_jik[ 4] * x_jNiZ[k  ]
            + L_jik[ 5] * x_jNiZ[k+1]
            + L_jik[ 6] * x_jNiP[k-1]
            + L_jik[ 7] * x_jNiP[k  ]
            + L_jik[ 8] * x_jNiP[k+1]
            + L_jik[ 9] * x_jZiN[k-1]
            + L_jik[10] * x_jZiN[k  ]
            + L_jik[11] * x_jZiN[k+1]
            + L_jik[12] * x_jik [k-1];// L * x_{k+1}
        x_jik[k] = b_jik[k] - tmp;
        L_jik += 13;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_ilu_backward_zero_3d27(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t* x_jPiZ = x_jik + dim_1       * dim_2,
                * x_jZiP = x_jik +               dim_2,
                * x_jPiN = x_jik + (dim_1 - 1) * dim_2,
                * x_jPiP = x_jik + (dim_1 + 1) * dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        data_t para = U_jik[0];
        data_t tmp = 
            + U_jik[ 1] * x_jik [k+1]
            + U_jik[ 2] * x_jZiP[k-1]
            + U_jik[ 3] * x_jZiP[k  ]
            + U_jik[ 4] * x_jZiP[k+1]
            + U_jik[ 5] * x_jPiN[k-1]
            + U_jik[ 6] * x_jPiN[k  ]
            + U_jik[ 7] * x_jPiN[k+1]
            + U_jik[ 8] * x_jPiZ[k-1]
            + U_jik[ 9] * x_jPiZ[k  ]
            + U_jik[10] * x_jPiZ[k+1]
            + U_jik[11] * x_jPiP[k-1]
            + U_jik[12] * x_jPiP[k  ]
            + U_jik[13] * x_jPiP[k+1];
        x_jik[k] = (b_jik[k] - tmp) / para;
        U_jik -= 14;
    }
}

#endif