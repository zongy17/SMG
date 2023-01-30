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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_spmv_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const data_t * dummy)
{
    const calc_t * x_jNi = x_jik - vec_ki_size, * x_jPi = x_jik + vec_ki_size;
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

template<typename idx_t, typename data_t, typename calc_t>
void inline SOA_spmv_3d27(const idx_t num,
    const idx_t vec_k_size , const idx_t vec_ki_size,
    const data_t * A_jik[27], const calc_t * x_jik, calc_t * y_jik)
{
    // 要打桩查一下是否生成了向量化代码！！！！
    const calc_t * x4 = x_jik - vec_ki_size, * x22 = x_jik + vec_ki_size,
                * x10= x_jik - vec_k_size , * x16 = x_jik + vec_k_size,
                * x12= x_jik - 1, * x14 = x_jik + 1;
    const calc_t * x1 = x4 - vec_k_size, * x7 = x4 + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1,
                * x19= x22- vec_k_size, * x25= x22+ vec_k_size,
                * x21= x22- 1, * x23= x22+ 1;
    const calc_t * x9 = x10- 1, * x11= x10+ 1,
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_zero_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jNiN = x_jik - vec_ki_size - vec_k_size,
                * x_jNiP = x_jik - vec_ki_size + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[13];
        calc_t tmp = 
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
            + L_jik[12] * x_jik [k-1];
        tmp = b_jik[k] - tmp;// b - L*x_{k+1}
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 14;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_zero_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t* x_jPiZ = x_jik + vec_ki_size,
                * x_jZiP = x_jik               + vec_k_size,
                * x_jPiN = x_jik + vec_ki_size - vec_k_size,
                * x_jPiP = x_jik + vec_ki_size + vec_k_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[13];
        calc_t tmp = 
            + U_jik[ 0] * x_jik [k+1]
            + U_jik[ 1] * x_jZiP[k-1]
            + U_jik[ 2] * x_jZiP[k  ]
            + U_jik[ 3] * x_jZiP[k+1]
            + U_jik[ 4] * x_jPiN[k-1]
            + U_jik[ 5] * x_jPiN[k  ]
            + U_jik[ 6] * x_jPiN[k+1]
            + U_jik[ 7] * x_jPiZ[k-1]
            + U_jik[ 8] * x_jPiZ[k  ]
            + U_jik[ 9] * x_jPiZ[k+1]
            + U_jik[10] * x_jPiP[k-1]
            + U_jik[11] * x_jPiP[k  ]
            + U_jik[12] * x_jPiP[k+1];// U*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k+1}

        x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 14;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_ALL_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jPiZ = x_jik + vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jZiP = x_jik               + vec_k_size;
    const calc_t* x_jNiN = x_jNiZ - vec_k_size, * x_jPiN = x_jPiZ - vec_k_size,
                * x_jNiP = x_jNiZ + vec_k_size, * x_jPiP = x_jPiZ + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[13];
        calc_t tmp = 
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
            + L_jik[12] * x_jik [k-1]
        // 
            + U_jik[ 0] * x_jik [k+1]
            + U_jik[ 1] * x_jZiP[k-1]
            + U_jik[ 2] * x_jZiP[k  ]
            + U_jik[ 3] * x_jZiP[k+1]
            + U_jik[ 4] * x_jPiN[k-1]
            + U_jik[ 5] * x_jPiN[k  ]
            + U_jik[ 6] * x_jPiN[k+1]
            + U_jik[ 7] * x_jPiZ[k-1]
            + U_jik[ 8] * x_jPiZ[k  ]
            + U_jik[ 9] * x_jPiZ[k+1]
            + U_jik[10] * x_jPiP[k-1]
            + U_jik[11] * x_jPiP[k  ]
            + U_jik[12] * x_jPiP[k+1];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik += 14;
        U_jik += 14;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_ALL_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jPiZ = x_jik + vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jZiP = x_jik               + vec_k_size;
    const calc_t* x_jNiN = x_jNiZ - vec_k_size, * x_jPiN = x_jPiZ - vec_k_size,
                * x_jNiP = x_jNiZ + vec_k_size, * x_jPiP = x_jPiZ + vec_k_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[13];
        calc_t tmp = 
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
            + L_jik[12] * x_jik [k-1]
        // 
            + U_jik[ 0] * x_jik [k+1]
            + U_jik[ 1] * x_jZiP[k-1]
            + U_jik[ 2] * x_jZiP[k  ]
            + U_jik[ 3] * x_jZiP[k+1]
            + U_jik[ 4] * x_jPiN[k-1]
            + U_jik[ 5] * x_jPiN[k  ]
            + U_jik[ 6] * x_jPiN[k+1]
            + U_jik[ 7] * x_jPiZ[k-1]
            + U_jik[ 8] * x_jPiZ[k  ]
            + U_jik[ 9] * x_jPiZ[k+1]
            + U_jik[10] * x_jPiP[k-1]
            + U_jik[11] * x_jPiP[k  ]
            + U_jik[12] * x_jPiP[k+1];
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik -= 14;
        U_jik -= 14;
    }
}

// =================================== LGS =============================

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_forward_zero_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * dummy, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jNiN = x_jik - vec_ki_size - vec_k_size,
                * x_jNiP = x_jik - vec_ki_size + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp = 
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_backward_zero_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * dummy, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t* x_jPiZ = x_jik + vec_ki_size,
                * x_jZiP = x_jik               + vec_k_size,
                * x_jPiN = x_jik + vec_ki_size - vec_k_size,
                * x_jPiP = x_jik + vec_ki_size + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_ALL_3d27(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t* x_jNiZ = x_jik - vec_ki_size,
                * x_jPiZ = x_jik + vec_ki_size,
                * x_jZiN = x_jik               - vec_k_size,
                * x_jZiP = x_jik               + vec_k_size;
    const calc_t* x_jNiN = x_jNiZ - vec_k_size, * x_jPiN = x_jPiZ - vec_k_size,
                * x_jNiP = x_jNiZ + vec_k_size, * x_jPiP = x_jPiZ + vec_k_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
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
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_forward_zero_3d27(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t* x_jNiZ = x_jik -  dim_1      * dim_2,
                * x_jZiN = x_jik -               dim_2,
                * x_jNiN = x_jik - (dim_1 + 1) * dim_2,
                * x_jNiP = x_jik - (dim_1 - 1) * dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        calc_t tmp = 
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_backward_zero_3d27(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t* x_jPiZ = x_jik + dim_1       * dim_2,
                * x_jZiP = x_jik +               dim_2,
                * x_jPiN = x_jik + (dim_1 - 1) * dim_2,
                * x_jPiP = x_jik + (dim_1 + 1) * dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        calc_t para = U_jik[0];
        calc_t tmp = 
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

// =========================================================================
// =========================== Structure Of Array ==========================
// =========================================================================
#define GROUP_LEN 8
#define NEON_LEN 4
// ============================ SPMV =================================
void inline SOA_spmv_3d27_Cal32Stg16(const int num,
    const int vec_k_size , const int vec_ki_size,
    const __fp16 * Diags[7], const float * x13, float * y_jik)
{
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2], * A12_15 = Diags[3],
                * A16_19 = Diags[4], * A20_23 = Diags[5], * A24_26 = Diags[6];
    const float * x4 = x13 - vec_ki_size, * x22 = x13 + vec_ki_size,
                * x10= x13 - vec_k_size , * x16 = x13 + vec_k_size ,
                * x12= x13 - 1          , * x14 = x13 + 1          ;
    const float * x1 = x4 - vec_k_size , * x7  = x4 + vec_k_size ,
                * x3 = x4 - 1          , * x5  = x4 + 1          ,
                * x19= x22- vec_k_size , * x25 = x22+ vec_k_size ,
                * x21= x22- 1          , * x23 = x22+ 1          ,
                * x9 = x10- 1          , * x11 = x10+ 1          ,
                * x15= x16- 1          , * x17 = x16+ 1          ;
    const float * x0 = x1 - 1          , * x2  = x1 + 1,
                * x6 = x7 - 1          , * x8  = x7 + 1,
                * x18= x19- 1          , * x20 = x19+ 1,
                * x24= x23- 1          , * x26 = x25+ 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0,
                x0_32_1, x1_32_1, x2_32_1, x3_32_1;
    float32x4_t tmp_0, tmp_1;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float16x8x3_t A24_26_16;
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
        // A16~A19
        A0_3_16 = vld4q_f16(A16_19);  A16_19  += GROUP_LEN * 4; __builtin_prefetch(A16_19 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x16); x16 += NEON_LEN; x0_32_1 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16,0);
        x1_32_0 = vld1q_f32(x17); x17 += NEON_LEN; x1_32_1 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17,0);
        x2_32_0 = vld1q_f32(x18); x18 += NEON_LEN; x2_32_1 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18,0);
        x3_32_0 = vld1q_f32(x19); x19 += NEON_LEN; x3_32_1 = vld1q_f32(x19); x19 += NEON_LEN; __builtin_prefetch(x19,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlaq_f32(tmp_1, A3_32_1, x3_32_1);
        // A20~A23
        A0_3_16 = vld4q_f16(A20_23);  A20_23  += GROUP_LEN * 4; __builtin_prefetch(A20_23 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x20); x20 += NEON_LEN; x0_32_1 = vld1q_f32(x20); x20 += NEON_LEN; __builtin_prefetch(x20,0);
        x1_32_0 = vld1q_f32(x21); x21 += NEON_LEN; x1_32_1 = vld1q_f32(x21); x21 += NEON_LEN; __builtin_prefetch(x21,0);
        x2_32_0 = vld1q_f32(x22); x22 += NEON_LEN; x2_32_1 = vld1q_f32(x22); x22 += NEON_LEN; __builtin_prefetch(x22,0);
        x3_32_0 = vld1q_f32(x23); x23 += NEON_LEN; x3_32_1 = vld1q_f32(x23); x23 += NEON_LEN; __builtin_prefetch(x23,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlaq_f32(tmp_1, A3_32_1, x3_32_1);
        // A24~A26
        A24_26_16 = vld3q_f16(A24_26);  A24_26  += GROUP_LEN * 3; __builtin_prefetch(A24_26 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A24_26_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A24_26_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A24_26_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A24_26_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A24_26_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A24_26_16.val[2]);
        x0_32_0 = vld1q_f32(x24); x24 += NEON_LEN; x0_32_1 = vld1q_f32(x24); x24 += NEON_LEN; __builtin_prefetch(x24,0);
        x1_32_0 = vld1q_f32(x25); x25 += NEON_LEN; x1_32_1 = vld1q_f32(x25); x25 += NEON_LEN; __builtin_prefetch(x25,0);
        x2_32_0 = vld1q_f32(x26); x26 += NEON_LEN; x2_32_1 = vld1q_f32(x26); x26 += NEON_LEN; __builtin_prefetch(x26,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);

        vst1q_f32(y_jik           , tmp_0);
        vst1q_f32(y_jik + NEON_LEN, tmp_1);
        y_jik += GROUP_LEN; __builtin_prefetch(y_jik,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A24_26_16;
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
        A0_3_16 = vld4_f16(A8_11); A8_11  += NEON_LEN * 4; __builtin_prefetch(A8_11 + NEON_LEN * 16, 0, 0);
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
        // A16~A19
        A0_3_16 = vld4_f16(A16_19);  A16_19  += NEON_LEN * 4; __builtin_prefetch(A16_19 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x16); x16 += NEON_LEN; __builtin_prefetch(x16,0);
        x1_32_0 = vld1q_f32(x17); x17 += NEON_LEN; __builtin_prefetch(x17,0);
        x2_32_0 = vld1q_f32(x18); x18 += NEON_LEN; __builtin_prefetch(x18,0);
        x3_32_0 = vld1q_f32(x19); x19 += NEON_LEN; __builtin_prefetch(x19,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A20~A23
        A0_3_16 = vld4_f16(A20_23);  A20_23  += NEON_LEN * 4; __builtin_prefetch(A20_23 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x20); x20 += NEON_LEN; __builtin_prefetch(x20,0);
        x1_32_0 = vld1q_f32(x21); x21 += NEON_LEN; __builtin_prefetch(x21,0);
        x2_32_0 = vld1q_f32(x22); x22 += NEON_LEN; __builtin_prefetch(x22,0);
        x3_32_0 = vld1q_f32(x23); x23 += NEON_LEN; __builtin_prefetch(x23,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A24~A26
        A24_26_16 = vld3_f16(A24_26);  A24_26  += NEON_LEN * 3; __builtin_prefetch(A24_26 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A24_26_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A24_26_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A24_26_16.val[2]);
        x0_32_0 = vld1q_f32(x24); x24 += NEON_LEN; __builtin_prefetch(x24,0);
        x1_32_0 = vld1q_f32(x25); x25 += NEON_LEN; __builtin_prefetch(x25,0);
        x2_32_0 = vld1q_f32(x26); x26 += NEON_LEN; __builtin_prefetch(x26,0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);

        vst1q_f32(y_jik           , tmp_0);
        vst1q_f32(y_jik + NEON_LEN, tmp_1);
        y_jik += NEON_LEN; __builtin_prefetch(y_jik,1);
    }
    for (k = 0; k < num - max_nk; k++) {
        y_jik[k] = 
            A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        +   A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        +   A8_11[0]* x8[k] + A8_11[1]* x9[k] + A8_11[2]* x9[k] + A8_11[3]* x9[k]
        +   A12_15[0]*x12[k]+ A12_15[1]*x13[k]+ A12_15[2]*x14[k]+ A12_15[3]*x15[k]
        +   A16_19[0]*x16[k]+ A16_19[1]*x17[k]+ A16_19[2]*x18[k]+ A16_19[3]*x19[k]
        +   A20_23[0]*x20[k]+ A20_23[1]*x21[k]+ A20_23[2]*x22[k]+ A20_23[3]*x23[k]
        +   A24_26[0]*x24[k]+ A24_26[1]*x25[k]+ A24_26[2]*x26[k];
        A0_3 += 4; A4_7 += 4; A8_11+= 4;
        A12_15+= 4; A16_19+= 4; A20_23+= 4;
        A24_26+= 3;
    }
}

// ============================= PGS ==================================
void inline SOA_point_forward_zero_3d27_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[4], const float * b13, float * x13, const float * dummy)
{// 对角线分组: (0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2], * A12_14 = Diags[3];
    const float * x4 = x13 - vec_ki_size, * x10 = x13 - vec_k_size;
    const float * x1 = x4 - vec_k_size  , * x7  = x4  + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1, * x9 = x10 - 1, * x11 = x10 + 1;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x6 = x7  - 1, * x8  = x7  + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    const float32x4_t vwgts = vdupq_n_f32(weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1,
                    x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float16x8x4_t A0_3_16;
        float16x8x3_t A12_14_16;
        float32x4_t tmp_0, tmp_1;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; tmp_1   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + GROUP_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4q_f16(A8_11); A8_11 += GROUP_LEN * 4; __builtin_prefetch(A8_11 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; x0_32_1 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; x1_32_1 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A12 ~ A14
        A12_14_16 = vld3q_f16(A12_14); A12_14 += GROUP_LEN * 3; __builtin_prefetch(A12_14 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A12_14_16.val[0]);// A12
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A12_14_16.val[1]);// A13
        A1_32_0 = vdivq_f32(vwgts, A1_32_0); A1_32_1 = vdivq_f32(vwgts, A1_32_1);
        vst1q_f32(x13, tmp_0);
        x13[0] = (x13[0] - vgetq_lane_f32(A0_32_0, 0) * x13[-1]) * vgetq_lane_f32(A1_32_0, 0);
        x13[1] = (x13[1] - vgetq_lane_f32(A0_32_0, 1) * x13[ 0]) * vgetq_lane_f32(A1_32_0, 1);
        x13[2] = (x13[2] - vgetq_lane_f32(A0_32_0, 2) * x13[ 1]) * vgetq_lane_f32(A1_32_0, 2);
        x13[3] = (x13[3] - vgetq_lane_f32(A0_32_0, 3) * x13[ 2]) * vgetq_lane_f32(A1_32_0, 3);
        x13 += NEON_LEN;
        vst1q_f32(x13, tmp_1);
        x13[0] = (x13[0] - vgetq_lane_f32(A0_32_1, 0) * x13[-1]) * vgetq_lane_f32(A1_32_1, 0);
        x13[1] = (x13[1] - vgetq_lane_f32(A0_32_1, 1) * x13[ 0]) * vgetq_lane_f32(A1_32_1, 1);
        x13[2] = (x13[2] - vgetq_lane_f32(A0_32_1, 2) * x13[ 1]) * vgetq_lane_f32(A1_32_1, 2);
        x13[3] = (x13[3] - vgetq_lane_f32(A0_32_1, 3) * x13[ 2]) * vgetq_lane_f32(A1_32_1, 3);
        x13 += NEON_LEN; __builtin_prefetch(x13 + GROUP_LEN,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A12_14_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + NEON_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4_f16(A8_11); A8_11 += NEON_LEN * 4; __builtin_prefetch(A8_11 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A12 ~ A14
        A12_14_16 = vld3_f16(A12_14); A12_14 += NEON_LEN * 3; __builtin_prefetch(A12_14 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A12_14_16.val[0]);// A12
        A1_32_0 = vcvt_f32_f16(A12_14_16.val[1]);// A13
        A1_32_0 = vdivq_f32(vwgts, A1_32_0);
        vst1q_f32(x13, tmp_0);
        x13[0] = (x13[0] - vgetq_lane_f32(A0_32_0, 0) * x13[-1]) * vgetq_lane_f32(A1_32_0, 0);
        x13[1] = (x13[1] - vgetq_lane_f32(A0_32_0, 1) * x13[ 0]) * vgetq_lane_f32(A1_32_0, 1);
        x13[2] = (x13[2] - vgetq_lane_f32(A0_32_0, 2) * x13[ 1]) * vgetq_lane_f32(A1_32_0, 2);
        x13[3] = (x13[3] - vgetq_lane_f32(A0_32_0, 3) * x13[ 2]) * vgetq_lane_f32(A1_32_0, 3);
        x13 += NEON_LEN; __builtin_prefetch(x13 + NEON_LEN,1);
    }
    for (k = 0; k < num - max_nk; k++) {
        float diag_val = A12_14[1];
        float tmp = 
        + A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        + A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        + A8_11[0]* x8[k] + A8_11[1]* x9[k] + A8_11[2]* x10[k]+ A8_11[3]* x11[k]
        + A12_14[0]* x13[k-1];
        tmp = b13[k] - tmp;// b - L*x_{k+1}
        x13[k] = weight * tmp / diag_val;
        A0_3 += 4; A4_7 += 4; A8_11 += 4; A12_14 += 3;
    }
}

void inline SOA_point_forward_ALL_3d27_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[7], const float * b13, float * x13, const float * dummy)
{// 对角线分组: (0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14),  (15,16,17,18), (19,20,21,22), (23,24,25,26)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2], * A12_14 = Diags[3],
                * A15_18= Diags[4], * A19_22 = Diags[5], * A23_26 = Diags[6];
    const float * x4 = x13 - vec_ki_size, * x10 = x13 - vec_k_size,
                * x22= x13 + vec_ki_size, * x16 = x13 + vec_k_size;
    const float * x1 = x4 - vec_k_size  , * x7  = x4  + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1, * x9 = x10 - 1, * x11 = x10 + 1,
                * x19= x22- vec_k_size  , * x25 = x22 + vec_k_size,
                * x21= x22- 1, * x23= x22+ 1, * x15= x16 - 1, * x17 = x16 + 1;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x6 = x7  - 1, * x8  = x7  + 1,
                * x18= x19- 1, * x20= x19+ 1, * x24= x25 - 1, * x26 = x25 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1,
                    x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float16x8x4_t A0_3_16;
        float16x8x3_t A12_14_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; tmp_1   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + GROUP_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4q_f16(A8_11); A8_11 += GROUP_LEN * 4; __builtin_prefetch(A8_11 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; x0_32_1 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; x1_32_1 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A15 ~ A18
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
        // A19 ~ A22
        A0_3_16 = vld4q_f16(A19_22); A19_22 += GROUP_LEN * 4; __builtin_prefetch(A19_22 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x19); x19 += NEON_LEN; x0_32_1 = vld1q_f32(x19); x19 += NEON_LEN; __builtin_prefetch(x19 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x20); x20 += NEON_LEN; x1_32_1 = vld1q_f32(x20); x20 += NEON_LEN; __builtin_prefetch(x20 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x21); x21 += NEON_LEN; x2_32_1 = vld1q_f32(x21); x21 += NEON_LEN; __builtin_prefetch(x21 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x22); x22 += NEON_LEN; x3_32_1 = vld1q_f32(x22); x22 += NEON_LEN; __builtin_prefetch(x22 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A23 ~ A26
        A0_3_16 = vld4q_f16(A23_26); A23_26 += GROUP_LEN * 4; __builtin_prefetch(A23_26 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x23); x23 += NEON_LEN; x0_32_1 = vld1q_f32(x23); x23 += NEON_LEN; __builtin_prefetch(x23 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x24); x24 += NEON_LEN; x1_32_1 = vld1q_f32(x24); x24 += NEON_LEN; __builtin_prefetch(x24 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x25); x25 += NEON_LEN; x2_32_1 = vld1q_f32(x25); x25 += NEON_LEN; __builtin_prefetch(x25 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x26); x26 += NEON_LEN; x3_32_1 = vld1q_f32(x26); x26 += NEON_LEN; __builtin_prefetch(x26 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // 本柱的 A12 ~ A14
        A12_14_16 = vld3q_f16(A12_14); A12_14 += GROUP_LEN * 3; __builtin_prefetch(A12_14 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A12_14_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A12_14_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A12_14_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0); A1_32_1 = vdivq_f32(vwgts, A1_32_1);// 此时A1_32存着wgt/A9
        float A12_buf[GROUP_LEN], A14_buf[GROUP_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0); A0_32_1 = vmulq_f32(A0_32_1, A1_32_1);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0); A2_32_1 = vmulq_f32(A2_32_1, A1_32_1);
        vst1q_f32(A12_buf, A0_32_0); vst1q_f32(A12_buf + NEON_LEN, A0_32_1);
        vst1q_f32(A14_buf, A2_32_0); vst1q_f32(A14_buf + NEON_LEN, A2_32_1);
        float * x_jik = x13;
        res_0 = vld1q_f32(x13); x13 += NEON_LEN     ; res_1 = vld1q_f32(x13); x13 += NEON_LEN; __builtin_prefetch(x13 + GROUP_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; res_1 = vmulq_f32(res_1, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0); res_1 = vmlaq_f32(res_1, tmp_1, A1_32_1);// 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A12_buf[0] * x_jik[-1] + A14_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A12_buf[1] * x_jik[ 0] + A14_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A12_buf[2] * x_jik[ 1] + A14_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A12_buf[3] * x_jik[ 2] + A14_buf[3] * x_jik[4]);
        x_jik[4] = vgetq_lane_f32(res_1, 0) - (A12_buf[4] * x_jik[ 3] + A14_buf[4] * x_jik[5]);
        x_jik[5] = vgetq_lane_f32(res_1, 1) - (A12_buf[5] * x_jik[ 4] + A14_buf[5] * x_jik[6]);
        x_jik[6] = vgetq_lane_f32(res_1, 2) - (A12_buf[6] * x_jik[ 5] + A14_buf[6] * x_jik[7]);
        x_jik[7] = vgetq_lane_f32(res_1, 3) - (A12_buf[7] * x_jik[ 6] + A14_buf[7] * x_jik[8]);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A12_14_16;
        float32x4_t tmp_0, res_0;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + NEON_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4_f16(A8_11); A8_11 += NEON_LEN * 4; __builtin_prefetch(A8_11 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + NEON_LEN, 0);
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
        // A19 ~ A22
        A0_3_16 = vld4_f16(A19_22); A19_22 += NEON_LEN * 4; __builtin_prefetch(A19_22 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x19); x19 += NEON_LEN; __builtin_prefetch(x19 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x20); x20 += NEON_LEN; __builtin_prefetch(x20 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x21); x21 += NEON_LEN; __builtin_prefetch(x21 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x22); x22 += NEON_LEN; __builtin_prefetch(x22 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A23 ~ A26
        A0_3_16 = vld4_f16(A23_26); A23_26 += NEON_LEN * 4; __builtin_prefetch(A23_26 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x23); x23 += NEON_LEN; __builtin_prefetch(x23 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x24); x24 += NEON_LEN; __builtin_prefetch(x24 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x25); x25 += NEON_LEN; __builtin_prefetch(x25 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x26); x26 += NEON_LEN; __builtin_prefetch(x26 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 本柱的 A12 ~ A14
        A12_14_16 = vld3_f16(A12_14); A12_14 += NEON_LEN * 3; __builtin_prefetch(A12_14 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A12_14_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(A12_14_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(A12_14_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0);// 此时A1_32存着wgt/A9
        float A12_buf[NEON_LEN], A14_buf[NEON_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0);
        vst1q_f32(A12_buf, A0_32_0);
        vst1q_f32(A14_buf, A2_32_0);
        float * x_jik = x13;
        res_0 = vld1q_f32(x13); x13 += NEON_LEN     ; __builtin_prefetch(x13 + NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; 
        res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0);// 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A12_buf[0] * x_jik[-1] + A14_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A12_buf[1] * x_jik[ 0] + A14_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A12_buf[2] * x_jik[ 1] + A14_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A12_buf[3] * x_jik[ 2] + A14_buf[3] * x_jik[4]);
    }
    for (k = 0; k < num - max_nk; k++) {
        float diag_val = A12_14[1];
        float tmp = 
        + A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        + A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        + A8_11[0]* x8[k] + A8_11[1]* x9[k] + A8_11[2]* x10[k]+ A8_11[3]* x11[k]
        + A12_14[0] * x13[k-1] + A12_14[2] * x13[k+1]
        + A15_18[0] * x15[k] + A15_18[1] * x16[k] + A15_18[2] * x17[k] + A15_18[3] * x18[k]
        + A19_22[0] * x19[k] + A19_22[1] * x20[k] + A19_22[2] * x21[k] + A19_22[3] * x22[k]
        + A23_26[0] * x23[k] + A23_26[1] * x24[k] + A23_26[2] * x25[k] + A23_26[3] * x26[k];
        tmp = b13[k] - tmp;// b - L*x_{k+1}
        x13[k] *= (1.0 - weight);
        x13[k] += weight * tmp / diag_val;
        A0_3 += 4; A4_7 += 4; A8_11 += 4; A12_14 += 3; A15_18 += 4; A19_22 += 4; A23_26 += 4;
    }
}

void inline SOA_point_backward_ALL_3d27_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[7], const float * b13, float * x13, const float * dummy)
{// 对角线分组: (0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14),  (15,16,17,18), (19,20,21,22), (23,24,25,26)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2], * A12_14 = Diags[3],
                * A15_18= Diags[4], * A19_22 = Diags[5], * A23_26 = Diags[6];
    const float * x4 = x13 - vec_ki_size, * x10 = x13 - vec_k_size,
                * x22= x13 + vec_ki_size, * x16 = x13 + vec_k_size;
    const float * x1 = x4 - vec_k_size  , * x7  = x4  + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1, * x9 = x10 - 1, * x11 = x10 + 1,
                * x19= x22- vec_k_size  , * x25 = x22 + vec_k_size,
                * x21= x22- 1, * x23= x22+ 1, * x15= x16 - 1, * x17 = x16 + 1;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x6 = x7  - 1, * x8  = x7  + 1,
                * x18= x19- 1, * x20= x19+ 1, * x24= x25 - 1, * x26 = x25 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1,
                    x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float16x8x4_t A0_3_16;
        float16x8x3_t A12_14_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1;
        b13 -= NEON_LEN; tmp_1 = vld1q_f32(b13); b13 -= NEON_LEN; tmp_0 = vld1q_f32(b13); __builtin_prefetch(b13 - GROUP_LEN, 0);
        // A0 ~ A3
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
        // A4 ~ A7
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
        // A8 ~ A11
        A8_11 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A8_11); __builtin_prefetch(A8_11 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x8  -= NEON_LEN; x0_32_1 = vld1q_f32(x8 ); x8  -= NEON_LEN; x0_32_0 = vld1q_f32(x8 ); __builtin_prefetch(x8  - GROUP_LEN, 0);
        x9  -= NEON_LEN; x1_32_1 = vld1q_f32(x9 ); x9  -= NEON_LEN; x1_32_0 = vld1q_f32(x9 ); __builtin_prefetch(x9  - GROUP_LEN, 0);
        x10 -= NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 -= NEON_LEN; x2_32_0 = vld1q_f32(x10); __builtin_prefetch(x10 - GROUP_LEN, 0);
        x11 -= NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 -= NEON_LEN; x3_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15 ~ A18
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
        // A19 ~ A22
        A19_22 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A19_22); __builtin_prefetch(A19_22 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x19 -= NEON_LEN; x0_32_1 = vld1q_f32(x19); x19 -= NEON_LEN; x0_32_0 = vld1q_f32(x19); __builtin_prefetch(x19 - GROUP_LEN, 0);
        x20 -= NEON_LEN; x1_32_1 = vld1q_f32(x20); x20 -= NEON_LEN; x1_32_0 = vld1q_f32(x20); __builtin_prefetch(x20 - GROUP_LEN, 0);
        x21 -= NEON_LEN; x2_32_1 = vld1q_f32(x21); x21 -= NEON_LEN; x2_32_0 = vld1q_f32(x21); __builtin_prefetch(x21 - GROUP_LEN, 0);
        x22 -= NEON_LEN; x3_32_1 = vld1q_f32(x22); x22 -= NEON_LEN; x3_32_0 = vld1q_f32(x22); __builtin_prefetch(x22 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A23 ~ A26
        A23_26 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A23_26); __builtin_prefetch(A23_26 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x23 -= NEON_LEN; x0_32_1 = vld1q_f32(x23); x23 -= NEON_LEN; x0_32_0 = vld1q_f32(x23); __builtin_prefetch(x23 - GROUP_LEN, 0);
        x24 -= NEON_LEN; x1_32_1 = vld1q_f32(x24); x24 -= NEON_LEN; x1_32_0 = vld1q_f32(x24); __builtin_prefetch(x24 - GROUP_LEN, 0);
        x25 -= NEON_LEN; x2_32_1 = vld1q_f32(x25); x25 -= NEON_LEN; x2_32_0 = vld1q_f32(x25); __builtin_prefetch(x25 - GROUP_LEN, 0);
        x26 -= NEON_LEN; x3_32_1 = vld1q_f32(x26); x26 -= NEON_LEN; x3_32_0 = vld1q_f32(x26); __builtin_prefetch(x26 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 本柱的 A12 ~ A14
        A12_14 -= GROUP_LEN * 3;
        A12_14_16 = vld3q_f16(A12_14); __builtin_prefetch(A12_14 - GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A12_14_16.val[0]);// A8
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A12_14_16.val[1]);// A9
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A12_14_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A12_14_16.val[2]);// A10
        A1_32_0 = vdivq_f32(vwgts, A1_32_0); A1_32_1 = vdivq_f32(vwgts, A1_32_1);// 此时A1_32存着wgt/A9
        float A12_buf[GROUP_LEN], A14_buf[GROUP_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0); A0_32_1 = vmulq_f32(A0_32_1, A1_32_1);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0); A2_32_1 = vmulq_f32(A2_32_1, A1_32_1);
        vst1q_f32(A12_buf, A0_32_0); vst1q_f32(A12_buf + NEON_LEN, A0_32_1);
        vst1q_f32(A14_buf, A2_32_0); vst1q_f32(A14_buf + NEON_LEN, A2_32_1);

        x13 -= NEON_LEN; res_1 = vld1q_f32(x13); x13 -= NEON_LEN; res_0 = vld1q_f32(x13); __builtin_prefetch(x13 - GROUP_LEN, 1);
        res_1 = vmulq_f32(res_1, vone_minus_wgts); res_0 = vmulq_f32(res_0, vone_minus_wgts) ; 
        res_1 = vmlaq_f32(res_1, tmp_1, A1_32_1);  res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0); // 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x13[7] = vgetq_lane_f32(res_1, 3) - (A12_buf[7] * x13[ 6] + A14_buf[7] * x13[8]);
        x13[6] = vgetq_lane_f32(res_1, 2) - (A12_buf[6] * x13[ 5] + A14_buf[6] * x13[7]);
        x13[5] = vgetq_lane_f32(res_1, 1) - (A12_buf[5] * x13[ 4] + A14_buf[5] * x13[6]);
        x13[4] = vgetq_lane_f32(res_1, 0) - (A12_buf[4] * x13[ 3] + A14_buf[4] * x13[5]);
        x13[3] = vgetq_lane_f32(res_0, 3) - (A12_buf[3] * x13[ 2] + A14_buf[3] * x13[4]);
        x13[2] = vgetq_lane_f32(res_0, 2) - (A12_buf[2] * x13[ 1] + A14_buf[2] * x13[3]);
        x13[1] = vgetq_lane_f32(res_0, 1) - (A12_buf[1] * x13[ 0] + A14_buf[1] * x13[2]);
        x13[0] = vgetq_lane_f32(res_0, 0) - (A12_buf[0] * x13[-1] + A14_buf[0] * x13[1]);
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A12_14_16;
        float32x4_t tmp_0, res_0;
        b13 -= NEON_LEN; tmp_0 = vld1q_f32(b13); __builtin_prefetch(b13 - NEON_LEN, 0);
        // A0 ~ A3
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
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A4 ~ A7
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
        // A8 ~ A11
        A8_11 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A8_11); __builtin_prefetch(A8_11 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x8  -= NEON_LEN; x0_32_0 = vld1q_f32(x8 ); __builtin_prefetch(x8  - NEON_LEN, 0);
        x9  -= NEON_LEN; x1_32_0 = vld1q_f32(x9 ); __builtin_prefetch(x9  - NEON_LEN, 0);
        x10 -= NEON_LEN; x2_32_0 = vld1q_f32(x10); __builtin_prefetch(x10 - NEON_LEN, 0);
        x11 -= NEON_LEN; x3_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15 ~ A18
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
        // A19 ~ A22
        A19_22 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A19_22); __builtin_prefetch(A19_22 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x19 -= NEON_LEN; x0_32_0 = vld1q_f32(x19); __builtin_prefetch(x19 - NEON_LEN, 0);
        x20 -= NEON_LEN; x1_32_0 = vld1q_f32(x20); __builtin_prefetch(x20 - NEON_LEN, 0);
        x21 -= NEON_LEN; x2_32_0 = vld1q_f32(x21); __builtin_prefetch(x21 - NEON_LEN, 0);
        x22 -= NEON_LEN; x3_32_0 = vld1q_f32(x22); __builtin_prefetch(x22 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A23 ~ A26
        A23_26 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A23_26); __builtin_prefetch(A23_26 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x23 -= NEON_LEN; x0_32_0 = vld1q_f32(x23); __builtin_prefetch(x23 - NEON_LEN, 0);
        x24 -= NEON_LEN; x1_32_0 = vld1q_f32(x24); __builtin_prefetch(x24 - NEON_LEN, 0);
        x25 -= NEON_LEN; x2_32_0 = vld1q_f32(x25); __builtin_prefetch(x25 - NEON_LEN, 0);
        x26 -= NEON_LEN; x3_32_0 = vld1q_f32(x26); __builtin_prefetch(x26 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 本柱的 A12 ~ A14
        A12_14 -= NEON_LEN * 3;
        A12_14_16 = vld3_f16(A12_14); __builtin_prefetch(A12_14 - NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A12_14_16.val[0]);// A12
        A1_32_0 = vcvt_f32_f16(A12_14_16.val[1]);// A13
        A2_32_0 = vcvt_f32_f16(A12_14_16.val[2]);// A14
        A1_32_0 = vdivq_f32(vwgts, A1_32_0);// 此时A1_32存着wgt/A9
        float A12_buf[NEON_LEN], A14_buf[NEON_LEN];// 暂存 wgt*A8/A9 和 wgt*A10/A9
        A0_32_0 = vmulq_f32(A0_32_0, A1_32_0);
        A2_32_0 = vmulq_f32(A2_32_0, A1_32_0);
        vst1q_f32(A12_buf, A0_32_0);
        vst1q_f32(A14_buf, A2_32_0);

        x13 -= NEON_LEN; res_0 = vld1q_f32(x13); __builtin_prefetch(x13 - NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; 
        res_0 = vmlaq_f32(res_0, tmp_0, A1_32_0); // 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x13[3] = vgetq_lane_f32(res_0, 3) - (A12_buf[3] * x13[ 2] + A14_buf[3] * x13[4]);
        x13[2] = vgetq_lane_f32(res_0, 2) - (A12_buf[2] * x13[ 1] + A14_buf[2] * x13[3]);
        x13[1] = vgetq_lane_f32(res_0, 1) - (A12_buf[1] * x13[ 0] + A14_buf[1] * x13[2]);
        x13[0] = vgetq_lane_f32(res_0, 0) - (A12_buf[0] * x13[-1] + A14_buf[0] * x13[1]);
    }
    A0_3 -= 4; A4_7 -= 4; A8_11 -= 4; A12_14 -= 3; A15_18 -= 4; A19_22 -= 4; A23_26 -= 4;
    x0 -= min_nk; x1 -= min_nk; x2 -= min_nk; x3 -= min_nk;
    x4 -= min_nk; x5 -= min_nk; x6 -= min_nk; x7 -= min_nk;
    x8 -= min_nk; x9 -= min_nk; x10-= min_nk; x11-= min_nk;
    x13-= min_nk;
    x15-= min_nk; x16-= min_nk; x17-= min_gk; x18-= min_nk;
    x19-= min_nk; x20-= min_nk; x21-= min_nk; x22-= min_nk;
    x23-= min_nk; x24-= min_nk; x25-= min_nk; x26-= min_nk; b13 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A12_14[1];
        float tmp = 
        + A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        + A4_7[0] * x4[k] + A4_7[1] * x5[k] + A4_7[2] * x6[k] + A4_7[3] * x7[k]
        + A8_11[0]* x8[k] + A8_11[1]* x9[k] + A8_11[2]* x10[k]+ A8_11[3]* x11[k]
        + A12_14[0] * x13[k-1] + A12_14[2] * x13[k+1]
        + A15_18[0] * x15[k] + A15_18[1] * x16[k] + A15_18[2] * x17[k] + A15_18[3] * x18[k]
        + A19_22[0] * x19[k] + A19_22[1] * x20[k] + A19_22[2] * x21[k] + A19_22[3] * x22[k]
        + A23_26[0] * x23[k] + A23_26[1] * x24[k] + A23_26[2] * x25[k] + A23_26[3] * x26[k];
        tmp = b13[k] - tmp;// b - L*x_{k+1}
        x13[k] *= (1.0 - weight);
        x13[k] += weight * tmp / diag_val;
        A0_3 -= 4; A4_7 -= 4; A8_11 -= 4; A12_14 -= 3; A15_18 -= 4; A19_22 -= 4; A23_26 -= 4;
    }
}

// =================================== LGS =============================

void inline SOA_line_forward_zero_3d27_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[3], const float * b13, const float * x13, float * rhs)
{// 对角线分组: (0,1,2,3), (4,5,6,7), (8,9,10,11)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2];
    const float * x4 = x13 - vec_ki_size, * x10 = x13 - vec_k_size;
    const float * x1 = x4 - vec_k_size  , * x7  = x4  + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1, * x9 = x10 - 1, * x11 = x10 + 1;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x6 = x7  - 1, * x8  = x7  + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; tmp_1   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + GROUP_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4q_f16(A8_11); A8_11 += GROUP_LEN * 4; __builtin_prefetch(A8_11 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; x0_32_1 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; x1_32_1 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; vst1q_f32(rhs, tmp_1); rhs += NEON_LEN; __builtin_prefetch(rhs + GROUP_LEN, 1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + NEON_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4_f16(A8_11); A8_11 += NEON_LEN * 4; __builtin_prefetch(A8_11 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + NEON_LEN, 0);
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
        + A8_11[0]*x8[k] + A8_11[1]*x9[k] + A8_11[2]*x10[k]+ A8_11[3]*x11[k];
        rhs[k] = b13[k] - tmp;// b - L*x_{k+1}
        A0_3 += 4; A4_7 += 4; A8_11 += 4;
    }
}

void inline SOA_line_forward_ALL_3d27_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[6], const float * b13, const float * x13, float * rhs)
{// 对角线分组: (0,1,2,3), (4,5,6,7), (8,9,10,11) || (15,16,17,18) (19,20,21,22) (23,24,25,26)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2],
                * A15_18= Diags[3], * A19_22 = Diags[4], * A23_26 = Diags[5];
    const float * x4 = x13 - vec_ki_size, * x10 = x13 - vec_k_size,
                * x22= x13 + vec_ki_size, * x16 = x13 + vec_k_size;
    const float * x1 = x4 - vec_k_size  , * x7  = x4  + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1, * x9 = x10 - 1, * x11 = x10 + 1,
                * x19= x22- vec_k_size  , * x25 = x22 + vec_k_size,
                * x21= x22- 1, * x23= x22+ 1, * x15= x16 - 1, * x17 = x16 + 1;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x6 = x7  - 1, * x8  = x7  + 1,
                * x18= x19- 1, * x20= x19+ 1, * x24= x25 - 1, * x26 = x25 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1,
                    x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; tmp_1   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + GROUP_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4q_f16(A8_11); A8_11 += GROUP_LEN * 4; __builtin_prefetch(A8_11 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; x0_32_1 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; x1_32_1 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A15 ~ A18
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
        // A19 ~ A22
        A0_3_16 = vld4q_f16(A19_22); A19_22 += GROUP_LEN * 4; __builtin_prefetch(A19_22 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x19); x19 += NEON_LEN; x0_32_1 = vld1q_f32(x19); x19 += NEON_LEN; __builtin_prefetch(x19 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x20); x20 += NEON_LEN; x1_32_1 = vld1q_f32(x20); x20 += NEON_LEN; __builtin_prefetch(x20 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x21); x21 += NEON_LEN; x2_32_1 = vld1q_f32(x21); x21 += NEON_LEN; __builtin_prefetch(x21 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x22); x22 += NEON_LEN; x3_32_1 = vld1q_f32(x22); x22 += NEON_LEN; __builtin_prefetch(x22 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        // A23 ~ A26
        A0_3_16 = vld4q_f16(A23_26); A23_26 += GROUP_LEN * 4; __builtin_prefetch(A23_26 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x23); x23 += NEON_LEN; x0_32_1 = vld1q_f32(x23); x23 += NEON_LEN; __builtin_prefetch(x23 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x24); x24 += NEON_LEN; x1_32_1 = vld1q_f32(x24); x24 += NEON_LEN; __builtin_prefetch(x24 + GROUP_LEN, 0);
        x2_32_0 = vld1q_f32(x25); x25 += NEON_LEN; x2_32_1 = vld1q_f32(x25); x25 += NEON_LEN; __builtin_prefetch(x25 + GROUP_LEN, 0);
        x3_32_0 = vld1q_f32(x26); x26 += NEON_LEN; x3_32_1 = vld1q_f32(x26); x26 += NEON_LEN; __builtin_prefetch(x26 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0); tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; vst1q_f32(rhs, tmp_1); rhs += NEON_LEN; __builtin_prefetch(rhs + GROUP_LEN, 1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13 + NEON_LEN, 0);
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
        // A8 ~ A11
        A0_3_16 = vld4_f16(A8_11); A8_11 += NEON_LEN * 4; __builtin_prefetch(A8_11 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x8 ); x8  += NEON_LEN; __builtin_prefetch(x8  + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9  + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11 + NEON_LEN, 0);
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
        // A19 ~ A22
        A0_3_16 = vld4_f16(A19_22); A19_22 += NEON_LEN * 4; __builtin_prefetch(A19_22 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x19); x19 += NEON_LEN; __builtin_prefetch(x19 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x20); x20 += NEON_LEN; __builtin_prefetch(x20 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x21); x21 += NEON_LEN; __builtin_prefetch(x21 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x22); x22 += NEON_LEN; __builtin_prefetch(x22 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A23 ~ A26
        A0_3_16 = vld4_f16(A23_26); A23_26 += NEON_LEN * 4; __builtin_prefetch(A23_26 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x0_32_0 = vld1q_f32(x23); x23 += NEON_LEN; __builtin_prefetch(x23 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x24); x24 += NEON_LEN; __builtin_prefetch(x24 + NEON_LEN, 0);
        x2_32_0 = vld1q_f32(x25); x25 += NEON_LEN; __builtin_prefetch(x25 + NEON_LEN, 0);
        x3_32_0 = vld1q_f32(x26); x26 += NEON_LEN; __builtin_prefetch(x26 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; __builtin_prefetch(rhs + NEON_LEN, 1);
    }
    for (k = 0; k < num - max_nk; k++) {
        float tmp = 
        + A0_3[0]*x0[k] + A0_3[1]*x1[k] + A0_3[2]*x2[k]  + A0_3[3]*x3[k]
        + A4_7[0]*x4[k] + A4_7[1]*x5[k] + A4_7[2]*x6[k]  + A4_7[3]*x7[k]
        + A8_11[0]*x8[k]+ A8_11[1]*x9[k]+ A8_11[2]*x10[k]+ A8_11[3]*x11[k]
        + A15_18[0]*x15[k] + A15_18[1]*x16[k] + A15_18[2]*x17[k] + A15_18[3]*x18[k]
        + A19_22[0]*x19[k] + A19_22[1]*x20[k] + A19_22[2]*x21[k] + A19_22[3]*x22[k]
        + A23_26[0]*x23[k] + A23_26[1]*x24[k] + A23_26[2]*x25[k] + A23_26[3]*x26[k];
        rhs[k] = b13[k] - tmp;// b - L*x_{k+1}
        A0_3 += 4; A4_7 += 4; A8_11 += 4; A15_18 += 4; A19_22 += 4; A23_26 += 4;
    }
}

void inline SOA_line_backward_ALL_3d27_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[6], const float * b13, const float * x13, float * rhs)
{// 对角线分组: (0,1,2,3), (4,5,6,7), (8,9,10,11) || (15,16,17,18) (19,20,21,22) (23,24,25,26)
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1], * A8_11 = Diags[2],
                * A15_18= Diags[3], * A19_22 = Diags[4], * A23_26 = Diags[5];
    const float * x4 = x13 - vec_ki_size, * x10 = x13 - vec_k_size,
                * x22= x13 + vec_ki_size, * x16 = x13 + vec_k_size;
    const float * x1 = x4 - vec_k_size  , * x7  = x4  + vec_k_size,
                * x3 = x4 - 1, * x5 = x4 + 1, * x9 = x10 - 1, * x11 = x10 + 1,
                * x19= x22- vec_k_size  , * x25 = x22 + vec_k_size,
                * x21= x22- 1, * x23= x22+ 1, * x15= x16 - 1, * x17 = x16 + 1;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x6 = x7  - 1, * x8  = x7  + 1,
                * x18= x19- 1, * x20= x19+ 1, * x24= x25 - 1, * x26 = x25 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float32x4_t A0_32_1, A1_32_1, A2_32_1, A3_32_1,
                    x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1;
        b13 -= NEON_LEN; tmp_1 = vld1q_f32(b13); b13 -= NEON_LEN; tmp_0 = vld1q_f32(b13); __builtin_prefetch(b13 - GROUP_LEN, 0);
        // A0 ~ A3
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
        // A4 ~ A7
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
        // A8 ~ A11
        A8_11 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A8_11); __builtin_prefetch(A8_11 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x8  -= NEON_LEN; x0_32_1 = vld1q_f32(x8 ); x8  -= NEON_LEN; x0_32_0 = vld1q_f32(x8 ); __builtin_prefetch(x8  - GROUP_LEN, 0);
        x9  -= NEON_LEN; x1_32_1 = vld1q_f32(x9 ); x9  -= NEON_LEN; x1_32_0 = vld1q_f32(x9 ); __builtin_prefetch(x9  - GROUP_LEN, 0);
        x10 -= NEON_LEN; x2_32_1 = vld1q_f32(x10); x10 -= NEON_LEN; x2_32_0 = vld1q_f32(x10); __builtin_prefetch(x10 - GROUP_LEN, 0);
        x11 -= NEON_LEN; x3_32_1 = vld1q_f32(x11); x11 -= NEON_LEN; x3_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15 ~ A18
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
        // A19 ~ A22
        A19_22 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A19_22); __builtin_prefetch(A19_22 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x19 -= NEON_LEN; x0_32_1 = vld1q_f32(x19); x19 -= NEON_LEN; x0_32_0 = vld1q_f32(x19); __builtin_prefetch(x19 - GROUP_LEN, 0);
        x20 -= NEON_LEN; x1_32_1 = vld1q_f32(x20); x20 -= NEON_LEN; x1_32_0 = vld1q_f32(x20); __builtin_prefetch(x20 - GROUP_LEN, 0);
        x21 -= NEON_LEN; x2_32_1 = vld1q_f32(x21); x21 -= NEON_LEN; x2_32_0 = vld1q_f32(x21); __builtin_prefetch(x21 - GROUP_LEN, 0);
        x22 -= NEON_LEN; x3_32_1 = vld1q_f32(x22); x22 -= NEON_LEN; x3_32_0 = vld1q_f32(x22); __builtin_prefetch(x22 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A23 ~ A26
        A23_26 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A23_26); __builtin_prefetch(A23_26 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        x23 -= NEON_LEN; x0_32_1 = vld1q_f32(x23); x23 -= NEON_LEN; x0_32_0 = vld1q_f32(x23); __builtin_prefetch(x23 - GROUP_LEN, 0);
        x24 -= NEON_LEN; x1_32_1 = vld1q_f32(x24); x24 -= NEON_LEN; x1_32_0 = vld1q_f32(x24); __builtin_prefetch(x24 - GROUP_LEN, 0);
        x25 -= NEON_LEN; x2_32_1 = vld1q_f32(x25); x25 -= NEON_LEN; x2_32_0 = vld1q_f32(x25); __builtin_prefetch(x25 - GROUP_LEN, 0);
        x26 -= NEON_LEN; x3_32_1 = vld1q_f32(x26); x26 -= NEON_LEN; x3_32_0 = vld1q_f32(x26); __builtin_prefetch(x26 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);

        rhs -= NEON_LEN; vst1q_f32(rhs, tmp_1); rhs -= NEON_LEN; vst1q_f32(rhs, tmp_0); __builtin_prefetch(rhs - GROUP_LEN, 1);
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        b13 -= NEON_LEN; tmp_0 = vld1q_f32(b13); __builtin_prefetch(b13 - NEON_LEN, 0);
        // A0 ~ A3
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
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A4 ~ A7
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
        // A8 ~ A11
        A8_11 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A8_11); __builtin_prefetch(A8_11 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x8  -= NEON_LEN; x0_32_0 = vld1q_f32(x8 ); __builtin_prefetch(x8  - NEON_LEN, 0);
        x9  -= NEON_LEN; x1_32_0 = vld1q_f32(x9 ); __builtin_prefetch(x9  - NEON_LEN, 0);
        x10 -= NEON_LEN; x2_32_0 = vld1q_f32(x10); __builtin_prefetch(x10 - NEON_LEN, 0);
        x11 -= NEON_LEN; x3_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A15 ~ A18
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
        // A19 ~ A22
        A19_22 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A19_22); __builtin_prefetch(A19_22 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x19 -= NEON_LEN; x0_32_0 = vld1q_f32(x19); __builtin_prefetch(x19 - NEON_LEN, 0);
        x20 -= NEON_LEN; x1_32_0 = vld1q_f32(x20); __builtin_prefetch(x20 - NEON_LEN, 0);
        x21 -= NEON_LEN; x2_32_0 = vld1q_f32(x21); __builtin_prefetch(x21 - NEON_LEN, 0);
        x22 -= NEON_LEN; x3_32_0 = vld1q_f32(x22); __builtin_prefetch(x22 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // A23 ~ A26
        A23_26 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A23_26); __builtin_prefetch(A23_26 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        x23 -= NEON_LEN; x0_32_0 = vld1q_f32(x23); __builtin_prefetch(x23 - NEON_LEN, 0);
        x24 -= NEON_LEN; x1_32_0 = vld1q_f32(x24); __builtin_prefetch(x24 - NEON_LEN, 0);
        x25 -= NEON_LEN; x2_32_0 = vld1q_f32(x25); __builtin_prefetch(x25 - NEON_LEN, 0);
        x26 -= NEON_LEN; x3_32_0 = vld1q_f32(x26); __builtin_prefetch(x26 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        
        rhs -= NEON_LEN; vst1q_f32(rhs, tmp_0); __builtin_prefetch(rhs - NEON_LEN, 1);
    }
    A0_3 -= 4; A4_7 -= 4; A8_11 -= 4; A15_18 -= 4; A19_22 -= 4; A23_26 -= 4;
    x0 -= min_nk; x1 -= min_nk; x2 -= min_nk; x3 -= min_nk;
    x4 -= min_nk; x5 -= min_nk; x6 -= min_nk; x7 -= min_nk;
    x8 -= min_nk; x9 -= min_nk; x10-= min_nk; x11-= min_nk;
    x15-= min_nk; x16-= min_nk; x17-= min_gk; x18-= min_nk;
    x19-= min_nk; x20-= min_nk; x21-= min_nk; x22-= min_nk;
    x23-= min_nk; x24-= min_nk; x25-= min_nk; x26-= min_nk; b13 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float tmp = 
        + A0_3[0]*x0[k] + A0_3[1]*x1[k] + A0_3[2]*x2[k] + A0_3[3]*x3[k]
        + A4_7[0]*x4[k] + A4_7[1]*x5[k] + A4_7[2]*x6[k] + A4_7[3]*x7[k]
        + A8_11[0]*x8[k]+ A8_11[1]*x9[k]+ A8_11[2]*x10[k]+ A8_11[3]*x11[k]
        + A15_18[0]*x15[k] + A15_18[1]*x16[k] + A15_18[2]*x17[k] + A15_18[3]*x18[k]
        + A19_22[0]*x19[k] + A19_22[1]*x20[k] + A19_22[2]*x21[k] + A19_22[3]*x22[k]
        + A23_26[0]*x23[k] + A23_26[1]*x24[k] + A23_26[2]*x25[k] + A23_26[3]*x26[k];
        rhs[k] = b13[k] - tmp;
        A0_3 -= 4; A4_7 -= 4; A8_11 -= 4; A15_18 -= 4; A19_22 -= 4; A23_26 -= 4;
    }
}

// ============================= BILU =================================

void inline SOA_ilu_forward_zero_3d27_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[3], const float * b13, float * x13)
{// L(0,1,2) (3,4,5) (6,7,8) (9,10,11,12)
    const __fp16* A0_2 = Diags[0], * A3_5 = Diags[1], * A6_8 = Diags[2], * A9_12 = Diags[3];
    const float * x4 = x13 - dim_1 * dim_2, * x10 = x13 - dim_2;
    const float * x1 = x4  - dim_2, * x7 = x4 + dim_2, * x3 = x4 - 1, * x5 = x4 + 1,
                * x9 = x10 - 1    , * x11= x10+ 1;
    const float * x0 = x1  - 1    , * x2 = x1 + 1, * x6 = x7 - 1, * x8 = x7 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = dim_2 & (~(GROUP_LEN - 1)), max_nk = dim_2 & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x3_t A0_2_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; tmp_1   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13, 0);
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
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_2_16.val[2]);// 对应x8
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        x1_32_0 = vld1q_f32(x7); x7 += NEON_LEN; x1_32_1 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7, 0);
        x2_32_0 = vld1q_f32(x8); x8 += NEON_LEN; x2_32_1 = vld1q_f32(x8); x8 += NEON_LEN; __builtin_prefetch(x8, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        // A9 ~ A12
        float16x8x4_t A9_12_16;
        A9_12_16 = vld4q_f16(A9_12); A9_12 += GROUP_LEN * 4; __builtin_prefetch(A9_12 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A9_12_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A9_12_16.val[0]);// 对应x9
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A9_12_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A9_12_16.val[1]);// 对应x10
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A9_12_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A9_12_16.val[2]);// 对应x11
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A9_12_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A9_12_16.val[3]);// 对应本柱的 x12
        x0_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; x0_32_1 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9 , 0);
        x1_32_0 = vld1q_f32(x10); x10 += NEON_LEN; x1_32_1 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10, 0);
        x2_32_0 = vld1q_f32(x11); x11 += NEON_LEN; x2_32_1 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        vst1q_f32(x13, tmp_0);
        x13[0] = x13[0] - vgetq_lane_f32(A3_32_0, 0) * x13[-1];
        x13[1] = x13[1] - vgetq_lane_f32(A3_32_0, 1) * x13[ 0];
        x13[2] = x13[2] - vgetq_lane_f32(A3_32_0, 2) * x13[ 1];
        x13[3] = x13[3] - vgetq_lane_f32(A3_32_0, 3) * x13[ 2];
        x13 += NEON_LEN;
        vst1q_f32(x13, tmp_1);
        x13[0] = x13[0] - vgetq_lane_f32(A3_32_1, 0) * x13[-1];
        x13[1] = x13[1] - vgetq_lane_f32(A3_32_1, 1) * x13[ 0];
        x13[2] = x13[2] - vgetq_lane_f32(A3_32_1, 2) * x13[ 1];
        x13[3] = x13[3] - vgetq_lane_f32(A3_32_1, 3) * x13[ 2];
        x13 += NEON_LEN; __builtin_prefetch(x13,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x3_t A0_2_16;
        float32x4_t tmp_0;
        tmp_0   = vld1q_f32(b13); b13 += NEON_LEN; __builtin_prefetch(b13, 0);
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
        A2_32_0 = vcvt_f32_f16(A0_2_16.val[2]);// 对应x8
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        x1_32_0 = vld1q_f32(x7); x7 += NEON_LEN; __builtin_prefetch(x7, 0);
        x2_32_0 = vld1q_f32(x8); x8 += NEON_LEN; __builtin_prefetch(x8, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        // A9 ~ A12
        float16x4x4_t A9_12_16;
        A9_12_16 = vld4_f16(A9_12); A9_12 += NEON_LEN * 4; __builtin_prefetch(A9_12 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A9_12_16.val[0]);// 对应x9
        A1_32_0 = vcvt_f32_f16(A9_12_16.val[1]);// 对应x10
        A2_32_0 = vcvt_f32_f16(A9_12_16.val[2]);// 对应x11
        A3_32_0 = vcvt_f32_f16(A9_12_16.val[3]);// 对应本柱的 x12
        x0_32_0 = vld1q_f32(x9 ); x9  += NEON_LEN; __builtin_prefetch(x9 , 0);
        x1_32_0 = vld1q_f32(x10); x10 += NEON_LEN; __builtin_prefetch(x10, 0);
        x2_32_0 = vld1q_f32(x11); x11 += NEON_LEN; __builtin_prefetch(x11, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        vst1q_f32(x13, tmp_0);
        x13[0] = x13[0] - vgetq_lane_f32(A3_32_0, 0) * x13[-1];
        x13[1] = x13[1] - vgetq_lane_f32(A3_32_0, 1) * x13[ 0];
        x13[2] = x13[2] - vgetq_lane_f32(A3_32_0, 2) * x13[ 1];
        x13[3] = x13[3] - vgetq_lane_f32(A3_32_0, 3) * x13[ 2];
        x13 += NEON_LEN; __builtin_prefetch(x13,1);
    }
    for (k = 0; k < dim_2 - max_nk; k++) {
        float tmp = 
        + A0_2[0]*x0[k] + A0_2[1]*x1[k] + A0_2[2]*x2[k]
        + A3_5[0]*x3[k] + A3_5[1]*x4[k] + A3_5[2]*x5[k]
        + A6_8[0]*x6[k] + A6_8[1]*x7[k] + A6_8[2]*x8[k]
        + A9_12[0]*x9[k] + A9_12[1]*x10[k] + A9_12[2]*x11[k] + A9_12[3]*x13[k-1];
        x13[k] = b13[k] - tmp;// b - L*x_{k+1}
        A0_2 += 3; A3_5 += 3; A6_8 += 3; A9_12 += 4;
    }
}

void inline SOA_ilu_backward_zero_3d27_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[3], const float * b13, float * x13)
{// U:(0,1,2) (3,4,5) (6,7,8,9) (10,11,12,13) 其中U(0)是主对角元
    const __fp16* A0_2 = Diags[0], * A3_5 = Diags[1], * A6_9 = Diags[2], * A10_13 = Diags[3];
    const float * x22 = x13 + dim_1 * dim_2, * x16 = x13 + dim_2;
    const float * x19 = x22 - dim_2, * x25 = x22 + dim_2, * x21 = x22 - 1, * x23 = x22 + 1,
                * x15 = x16 - 1    , * x17 = x16 + 1;
    const float * x18 = x19 - 1    , * x20 = x19 + 1, * x24 = x25 - 1, * x26 = x25 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    float32x4_t vones = vdupq_n_f32(1.0);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = dim_2, min_gk = dim_2 & (GROUP_LEN - 1), min_nk = dim_2 & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        b13 -= NEON_LEN; tmp_1 = vld1q_f32(b13)  ; b13 -= NEON_LEN; tmp_0 = vld1q_f32(b13)  ; __builtin_prefetch(b13 - GROUP_LEN, 0);
        //
        A10_13 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A10_13); __builtin_prefetch(A10_13 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 对应x23
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应x24
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应x25
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// 对应x26
        x23-= NEON_LEN; x0_32_1 = vld1q_f32(x23); x23 -= NEON_LEN; x0_32_0 = vld1q_f32(x23); __builtin_prefetch(x23 - GROUP_LEN, 0);
        x24-= NEON_LEN; x1_32_1 = vld1q_f32(x24); x24 -= NEON_LEN; x1_32_0 = vld1q_f32(x24); __builtin_prefetch(x24 - GROUP_LEN, 0);
        x25-= NEON_LEN; x2_32_1 = vld1q_f32(x25); x25 -= NEON_LEN; x2_32_0 = vld1q_f32(x25); __builtin_prefetch(x25 - GROUP_LEN, 0);
        x26-= NEON_LEN; x3_32_1 = vld1q_f32(x26); x26 -= NEON_LEN; x3_32_0 = vld1q_f32(x26); __builtin_prefetch(x26 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 
        A6_9 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A6_9); __builtin_prefetch(A6_9 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 对应x19
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应x20
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应x21
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// 对应x22
        x19-= NEON_LEN; x0_32_1 = vld1q_f32(x19); x19 -= NEON_LEN; x0_32_0 = vld1q_f32(x19); __builtin_prefetch(x19 - GROUP_LEN, 0);
        x20-= NEON_LEN; x1_32_1 = vld1q_f32(x20); x20 -= NEON_LEN; x1_32_0 = vld1q_f32(x20); __builtin_prefetch(x20 - GROUP_LEN, 0);
        x21-= NEON_LEN; x2_32_1 = vld1q_f32(x21); x21 -= NEON_LEN; x2_32_0 = vld1q_f32(x21); __builtin_prefetch(x21 - GROUP_LEN, 0);
        x22-= NEON_LEN; x3_32_1 = vld1q_f32(x22); x22 -= NEON_LEN; x3_32_0 = vld1q_f32(x22); __builtin_prefetch(x22 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        //
        float16x8x3_t A3_5_16;
        A3_5 -= GROUP_LEN * 3;
        A3_5_16 = vld3q_f16(A3_5); __builtin_prefetch(A3_5 - GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A3_5_16.val[0]);// 对应x16
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A3_5_16.val[1]);// 对应x17
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A3_5_16.val[2]);// 对应x18
        x16-= NEON_LEN; x0_32_1 = vld1q_f32(x16); x16 -= NEON_LEN; x0_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - GROUP_LEN, 0);
        x17-= NEON_LEN; x1_32_1 = vld1q_f32(x17); x17 -= NEON_LEN; x1_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - GROUP_LEN, 0);
        x18-= NEON_LEN; x2_32_1 = vld1q_f32(x18); x18 -= NEON_LEN; x2_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        //
        A0_2 -= GROUP_LEN * 3;
        A3_5_16 = vld3q_f16(A0_2); __builtin_prefetch(A0_2 - GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A3_5_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A3_5_16.val[1]);// 对应本柱的 x14
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A3_5_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A3_5_16.val[2]);// 对应x15
        A0_32_0 = vdivq_f32(vones, A0_32_0)                 ; A0_32_1 = vdivq_f32(vones, A0_32_1);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0)               ; A1_32_1 = vmulq_f32(A1_32_1, A0_32_1);// A14/A13
        float A14_buf[GROUP_LEN];// 暂存本柱的 A14/A13
        vst1q_f32(A14_buf, A1_32_0)                         ; vst1q_f32(A14_buf + NEON_LEN, A1_32_1);
        x15-= NEON_LEN; x2_32_1 = vld1q_f32(x15); x15 -= NEON_LEN; x2_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_1 = vmulq_f32(tmp_1, A0_32_1)                   ; tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x13 -= GROUP_LEN; __builtin_prefetch(x13 - GROUP_LEN, 1);
        x13[7] = vgetq_lane_f32(tmp_1, 3) - A14_buf[7] * x13[8];
        x13[6] = vgetq_lane_f32(tmp_1, 2) - A14_buf[6] * x13[7];
        x13[5] = vgetq_lane_f32(tmp_1, 1) - A14_buf[5] * x13[6];
        x13[4] = vgetq_lane_f32(tmp_1, 0) - A14_buf[4] * x13[5];
        x13[3] = vgetq_lane_f32(tmp_0, 3) - A14_buf[3] * x13[4];
        x13[2] = vgetq_lane_f32(tmp_0, 2) - A14_buf[2] * x13[3];
        x13[1] = vgetq_lane_f32(tmp_0, 1) - A14_buf[1] * x13[2];
        x13[0] = vgetq_lane_f32(tmp_0, 0) - A14_buf[0] * x13[1];
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        b13 -= NEON_LEN; tmp_0 = vld1q_f32(b13)  ; __builtin_prefetch(b13 - NEON_LEN, 0);
        //
        A10_13 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A10_13); __builtin_prefetch(A10_13 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 对应x23
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应x24
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应x25
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// 对应x26
        x23 -= NEON_LEN; x0_32_0 = vld1q_f32(x23); __builtin_prefetch(x23 - NEON_LEN, 0);
        x24 -= NEON_LEN; x1_32_0 = vld1q_f32(x24); __builtin_prefetch(x24 - NEON_LEN, 0);
        x25 -= NEON_LEN; x2_32_0 = vld1q_f32(x25); __builtin_prefetch(x25 - NEON_LEN, 0);
        x26 -= NEON_LEN; x3_32_0 = vld1q_f32(x26); __builtin_prefetch(x26 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        // 
        A6_9 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A6_9); __builtin_prefetch(A6_9 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 对应x19
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应x20
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应x21
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// 对应x22
        x19 -= NEON_LEN; x0_32_0 = vld1q_f32(x19); __builtin_prefetch(x19 - NEON_LEN, 0);
        x20 -= NEON_LEN; x1_32_0 = vld1q_f32(x20); __builtin_prefetch(x20 - NEON_LEN, 0);
        x21 -= NEON_LEN; x2_32_0 = vld1q_f32(x21); __builtin_prefetch(x21 - NEON_LEN, 0);
        x22 -= NEON_LEN; x3_32_0 = vld1q_f32(x22); __builtin_prefetch(x22 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);
        //
        float16x4x3_t A3_5_16;
        A3_5 -= NEON_LEN * 3;
        A3_5_16 = vld3_f16(A3_5); __builtin_prefetch(A3_5 - NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A3_5_16.val[0]);// 对应x16
        A1_32_0 = vcvt_f32_f16(A3_5_16.val[1]);// 对应x17
        A2_32_0 = vcvt_f32_f16(A3_5_16.val[2]);// 对应x18
        x16 -= NEON_LEN; x0_32_0 = vld1q_f32(x16); __builtin_prefetch(x16 - NEON_LEN, 0);
        x17 -= NEON_LEN; x1_32_0 = vld1q_f32(x17); __builtin_prefetch(x17 - NEON_LEN, 0);
        x18 -= NEON_LEN; x2_32_0 = vld1q_f32(x18); __builtin_prefetch(x18 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        //
        A0_2 -= NEON_LEN * 3;
        A3_5_16 = vld3_f16(A0_2); __builtin_prefetch(A0_2 - NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A3_5_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(A3_5_16.val[1]);// 对应本柱的 x14
        A2_32_0 = vcvt_f32_f16(A3_5_16.val[2]);// 对应x15
        A0_32_0 = vdivq_f32(vones, A0_32_0);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0);// A14/A13
        float A14_buf[NEON_LEN];// 暂存本柱的 A14/A13
        vst1q_f32(A14_buf, A1_32_0);
        x15 -= NEON_LEN; x2_32_0 = vld1q_f32(x15); __builtin_prefetch(x15 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x13 -= NEON_LEN; __builtin_prefetch(x13 - NEON_LEN, 1);
        x13[3] = vgetq_lane_f32(tmp_0, 3) - A14_buf[3] * x13[4];
        x13[2] = vgetq_lane_f32(tmp_0, 2) - A14_buf[2] * x13[3];
        x13[1] = vgetq_lane_f32(tmp_0, 1) - A14_buf[1] * x13[2];
        x13[0] = vgetq_lane_f32(tmp_0, 0) - A14_buf[0] * x13[1];
    }
    A0_2 -= 3; A3_5 -= 3; A6_9 -= 4; A10_13 -= 4;
    x13 -= min_nk; b13 -= min_nk;
    x15 -= min_nk; x16 -= min_nk; x17 -= min_nk; x18 -= min_nk;
    x19 -= min_nk; x20 -= min_nk; x21 -= min_nk; x22 -= min_nk;
    x23 -= min_nk; x24 -= min_nk; x25 -= min_nk; x26 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A0_2[0];
        float tmp = 
            +                  A0_2[1]*x13[k+1] + A0_2[2]*x15[k]
            + A3_5[0]*x16[k] + A3_5[1]*x17[k] + A3_5[2]*x18[k]
            + A6_9[0]*x19[k] + A6_9[1]*x20[k] + A6_9[2]*x21[k] + A6_9[3]*x22[k]
            + A10_13[0]*x23[k] + A10_13[1]*x24[k] + A10_13[2]*x25[k] + A10_13[3]*x26[k];
        x13[k] = (b13[k] - tmp) / diag_val;
        A0_2 -= 3; A3_5 -= 3; A6_9 -= 4; A10_13 -= 4;
    }
}

#undef NEON_LEN
#undef GROUP_LEN

#endif