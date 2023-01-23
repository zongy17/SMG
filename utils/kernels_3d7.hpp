#ifndef SMG_KERNELS_3D7_HPP
#define SMG_KERNELS_3D7_HPP

// ===================================================================
// ========================  3d7  kernels  ===========================
// ===================================================================
        /*  
                  /--------------/                    
                 / |    4       / |                   
                /--|-----------/  |
               |   |           |  |
               |   |------6----|--|
   z   y       | 1 |    3      | 5|
   ^  ^        |---|-- 0 ------|  |
   | /         |   |           |  |
   |/          |   /-----------|--/ 
   O-------> x | /      2      | / 
               |/--------------|/

        */
// =================================== SPMV =============================

template<typename idx_t, typename data_t>
void inline AOS_spmv_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const data_t * x_jik, data_t * y_jik, const data_t * dummy)
{
    const data_t * x_jNi = x_jik - vec_ki_size, * x_jPi = x_jik + vec_ki_size;
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k] = 
                A_jik[0] * x_jNi[               k    ]
            +   A_jik[1] * x_jik[- vec_k_size + k    ]
            +   A_jik[2] * x_jik[               k - 1]
            +   A_jik[3] * x_jik[               k    ]
            +   A_jik[4] * x_jik[               k + 1]
            +   A_jik[5] * x_jik[  vec_k_size + k    ]
            +   A_jik[6] * x_jPi[               k    ];
        A_jik += 7;// move the ptr
    }
}

template<typename idx_t, typename data_t>
void inline AOS_spmv_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const data_t * x_jik, data_t * y_jik, const data_t * sqD_jik)
{
    const data_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const data_t * sqD_jNi = sqD_jik - vec_ki_size, * sqD_jPi = sqD_jik + vec_ki_size;
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        data_t tmp = 
            + A_jik[0] * x_jNi[               k    ] * sqD_jNi[               k    ]
            + A_jik[1] * x_jik[- vec_k_size + k    ] * sqD_jik[- vec_k_size + k    ]
            + A_jik[2] * x_jik[               k - 1] * sqD_jik[               k - 1]
            + A_jik[3] * x_jik[               k    ] * sqD_jik[               k    ]
            + A_jik[4] * x_jik[               k + 1] * sqD_jik[               k + 1]
            + A_jik[5] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
            + A_jik[6] * x_jPi[               k    ] * sqD_jPi[               k    ];
        
        y_jik[k] = tmp * sqD_jik[k];
        A_jik += 7;// move the ptr
    }
}

// ================================= PGS ==================================
template<typename idx_t, typename data_t>
void inline AOS_point_forward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * b_jik, data_t * x_jik, const data_t * dummy)
{
    const data_t * x_jNi   = x_jik   - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t diag_val = L_jik[3];
        data_t tmp = 
            + L_jik[0] * x_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ]
            + L_jik[2] * x_jik[               k-1];// L * x_{k+1}
        tmp = b_jik[k] - tmp;// b - L*x_{k+1}
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 4;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_forward_zero_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * b_jik, data_t * x_jik, const data_t * sqD_jik)
{
    const data_t * x_jNi   = x_jik   - vec_ki_size;
    const data_t * sqD_jNi = sqD_jik - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t diag_val = L_jik[3] * sqD_jik[k] * sqD_jik[k];
        data_t tmp = 
            + L_jik[0] * x_jNi[               k  ] * sqD_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ] * sqD_jik[- vec_k_size + k  ]
            + L_jik[2] * x_jik[               k-1] * sqD_jik[               k-1];// L * x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - L*x_{k+1}
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 4;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_forward_ALL_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik, const data_t * b_jik, data_t * x_jik, const data_t * dummy)
{
    const data_t one_minus_weight = 1.0 - wgt;
    const data_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t diag_val = A_jik[3];
        data_t tmp = 
        + A_jik[0] * x_jNi[               k    ]
        + A_jik[1] * x_jik[- vec_k_size + k    ]
        + A_jik[2] * x_jik[               k - 1]
        // + A_jik[3] * x_jik[               k    ]
        + A_jik[4] * x_jik[               k + 1]
        + A_jik[5] * x_jik[  vec_k_size + k    ]
        + A_jik[6] * x_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        A_jik += 7;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_forward_ALL_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik, const data_t * b_jik, data_t * x_jik, const data_t * sqD_jik)
{
    const data_t one_minus_weight = 1.0 - wgt;
    const data_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const data_t * sqD_jNi = sqD_jik - vec_ki_size, * sqD_jPi = sqD_jik + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t diag_val = A_jik[3] * sqD_jik[k] * sqD_jik[k];
        data_t tmp = 
        + A_jik[0] * x_jNi[               k    ] * sqD_jNi[               k    ]
        + A_jik[1] * x_jik[- vec_k_size + k    ] * sqD_jik[- vec_k_size + k    ]
        + A_jik[2] * x_jik[               k - 1] * sqD_jik[               k - 1]
        // + A_jik[3] * x_jik[               k    ]
        + A_jik[4] * x_jik[               k + 1] * sqD_jik[               k + 1]
        + A_jik[5] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
        + A_jik[6] * x_jPi[               k    ] * sqD_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        A_jik += 7;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_backward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * U_jik, const data_t * b_jik, data_t * x_jik, const data_t * dummy)
{
    const data_t * x_jPi   = x_jik   + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        data_t diag_val = U_jik[3];
        data_t tmp = 
            + U_jik[0] * x_jik[               k + 1]
            + U_jik[1] * x_jik[  vec_k_size + k    ]
            + U_jik[2] * x_jPi[               k    ];// U*x_{k+1}
            tmp = b_jik[k] - tmp;// b - U*x_{k+1}

            x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 4;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_backward_zero_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * U_jik, const data_t * b_jik, data_t * x_jik, const data_t * sqD_jik)
{
    const data_t * x_jPi   = x_jik   + vec_ki_size;
    const data_t * sqD_jPi = sqD_jik + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        data_t diag_val = U_jik[3] * sqD_jik[k] * sqD_jik[k];
        data_t tmp = 
        + U_jik[0] * x_jik[               k + 1] * sqD_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ] * sqD_jPi[               k    ];// U*x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - U*x_{k+1}

        x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 4;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_backward_ALL_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik, const data_t * b_jik, data_t * x_jik, const data_t * dummy)
{
    const data_t one_minus_weight = 1.0 - wgt;
    const data_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        data_t diag_val = A_jik[3];
        data_t tmp = 
        + A_jik[0] * x_jNi[               k    ]
        + A_jik[1] * x_jik[- vec_k_size + k    ]
        + A_jik[2] * x_jik[               k - 1]
        // + A_jik[3] * x_jik[               k    ]
        + A_jik[4] * x_jik[               k + 1]
        + A_jik[5] * x_jik[  vec_k_size + k    ]
        + A_jik[6] * x_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        A_jik -= 7;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_point_backward_ALL_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik, const data_t * b_jik, data_t * x_jik, const data_t * sqD_jik)
{
    const data_t one_minus_weight = 1.0 - wgt;
    const data_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const data_t * sqD_jNi = sqD_jik - vec_ki_size, * sqD_jPi = sqD_jik + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        data_t diag_val = A_jik[3] * sqD_jik[k] * sqD_jik[k];
        data_t tmp = 
        + A_jik[0] * x_jNi[               k    ] * sqD_jNi[               k    ]
        + A_jik[1] * x_jik[- vec_k_size + k    ] * sqD_jik[- vec_k_size + k    ]
        + A_jik[2] * x_jik[               k - 1] * sqD_jik[               k - 1]
        // + A_jik[3] * x_jik[               k    ]
        + A_jik[4] * x_jik[               k + 1] * sqD_jik[               k + 1]
        + A_jik[5] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
        + A_jik[6] * x_jPi[               k    ] * sqD_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        A_jik -= 7;
    }
}

// ================================= LGS =======================================
template<typename idx_t, typename data_t>
void inline AOS_line_forward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * dummy, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t * x_jNi   = x_jik   - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp = 
            + L_jik[0] * x_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ];// L * x_{k+1}
        rhs[k] = b_jik[k] - tmp;// b - L*x_{k+1}
        L_jik += 2;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t>
void inline AOS_line_backward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * dummy, const data_t * U_jik, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =
            + U_jik[0] * x_jik[ vec_k_size + k  ]
            + U_jik[1] * x_jPi[              k  ];
        rhs[k] = b_jik[k] - tmp;
        U_jik += 2;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_line_ALL_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const data_t * b_jik, const data_t * x_jik, data_t * rhs)
{
    const data_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =
            + L_jik[0] * x_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ]
            + U_jik[0] * x_jik[  vec_k_size + k  ]
            + U_jik[1] * x_jPi[               k  ];
        rhs[k] = b_jik[k] - tmp;
        L_jik += 2;
        U_jik += 2;
    }
}

// =============================== BILU =======================================
template<typename idx_t, typename data_t>
void inline AOS_ilu_forward_zero_3d7(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t * x_jNi   = x_jik   - dim_1 * dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        data_t tmp = 
            + L_jik[0] * x_jNi[          k  ]
            + L_jik[1] * x_jik[- dim_2 + k  ]
            + L_jik[2] * x_jik[          k-1];// L * x_{k+1}
        x_jik[k] = b_jik[k] - tmp;
        // printf("k %d b %.5e (%.5e, %.5e) (%.5e, %.5e) (%.5e, %.5e) res %.5e\n", k, b_jik[k],
        //     L_jik[0], x_jNi[          k  ], L_jik[1], x_jik[- dim_2 + k  ],
        //     L_jik[2], x_jik[          k-1], x_jik[k]);

        L_jik += 3;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_ilu_backward_zero_3d7(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t * x_jPi   = x_jik   + dim_1 * dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        data_t para = U_jik[0];
        data_t tmp = 
            + U_jik[1] * x_jik[         k + 1]
            + U_jik[2] * x_jik[ dim_2 + k    ]
            + U_jik[3] * x_jPi[         k    ];
        x_jik[k] = (b_jik[k] - tmp) / para;
        U_jik -= 4;
    }
}

/* 
---------------------------------------------------

    Structure of Array: diagonals separated !!!

-------------------------------------------------------
*/

template<typename idx_t, typename data_t>
void inline SOA_spmv_3d7(const idx_t num,
    const idx_t vec_k_size , const idx_t vec_ki_size,
    const data_t * A_jik[7], const data_t * x_jik, data_t * y_jik)
{
    // 要打桩查一下是否生成了向量化代码！！！！
    const data_t* x0 = x_jik - vec_ki_size, * x6 = x_jik + vec_ki_size,
                * x1 = x_jik - vec_k_size,  * x5 = x_jik + vec_k_size,
                * x2 = x_jik - 1, * x4 = x_jik + 1;
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3],
                * A4 = A_jik[4], * A5 = A_jik[5], * A6 = A_jik[6];
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        y_jik[k]= A0[k] * x0[k]
                + A1[k] * x1[k]
                + A2[k] * x2[k]
                + A3[k] * x_jik[k]
                + A4[k] * x4[k]
                + A5[k] * x5[k]
                + A6[k] * x6[k];
    }
}

template<typename idx_t, typename data_t>
void inline SOA_forward_zero_3d7(const idx_t num, 
    const idx_t vec_k_size , const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik[4], const data_t * b_jik, data_t * x_jik)
{
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3];
    for (idx_t k = 0; k < num; k++) {
        data_t tmp =  A0[k] * x_jik[k - vec_ki_size]
                    + A1[k] * x_jik[k - vec_k_size]
                    + A2[k] * x_jik[k - 1];
        x_jik[k] = wgt * (b_jik[k] - tmp) / A3[k];
    }
}

template<typename idx_t, typename data_t>
void inline SOA_forward_zero_3d7_scaled(const idx_t num, 
    const idx_t vec_k_size , const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik[4], const data_t * b_jik, data_t * x_jik, const data_t * sqD_jik)
{
    const data_t* x0 = x_jik - vec_ki_size,
                * x1 = x_jik - vec_k_size, * x2 = x_jik - 1;
    const data_t* sqD0 = sqD_jik - vec_ki_size,
                * sqD1 = sqD_jik - vec_k_size, * sqD2 = sqD_jik - 1;
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3];
    for (idx_t k = 0; k < num; k++) {
        data_t diag_val = A3[k] * sqD_jik[k] * sqD_jik[k];
        data_t tmp =  A0[k] * x0[k] * sqD0[k] + A1[k] * x1[k] * sqD1[k] + A2[k] * x2[k] * sqD2[k];
        x_jik[k] = wgt * (b_jik[k] - tmp * sqD_jik[k]) / diag_val;
    }
}

template<typename idx_t, typename data_t, int stride>
void inline SOA_ALL_3d7(const idx_t num, 
    const idx_t vec_k_size , const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik[7], const data_t * b_jik, data_t * x_jik)
{
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3],
                * A4 = A_jik[4], * A5 = A_jik[5], * A6 = A_jik[6];
    const data_t one_minus_wgt = 1.0 - wgt;
    for (idx_t k = 0; k < num; k += stride) {
        data_t tmp =  A0[k] * x_jik[k - vec_ki_size]
                    + A1[k] * x_jik[k - vec_k_size]
                    + A2[k] * x_jik[k - 1]
                    + A4[k] * x_jik[k + 1]
                    + A5[k] * x_jik[k + vec_k_size]
                    + A6[k] * x_jik[k + vec_ki_size];
        x_jik[k] = one_minus_wgt * x_jik[k] + wgt * (b_jik[k] - tmp) / A3[k];
    }
}

template<typename idx_t, typename data_t, int stride>
void inline SOA_ALL_3d7_scaled(const idx_t num, 
    const idx_t vec_k_size , const idx_t vec_ki_size, const data_t wgt,
    const data_t * A_jik[7], const data_t * b_jik, data_t * x_jik, const data_t * sqD_jik)
{
    const data_t* x0 = x_jik - vec_ki_size, * x6 = x_jik + vec_ki_size,
                * x1 = x_jik - vec_k_size , * x5 = x_jik + vec_k_size,
                * x2 = x_jik - 1          , * x4 = x_jik + 1;
    const data_t* sqD0 = sqD_jik - vec_ki_size, * sqD6 = sqD_jik + vec_ki_size,
                * sqD1 = sqD_jik - vec_k_size , * sqD5 = sqD_jik + vec_k_size,
                * sqD2 = sqD_jik - 1          , * sqD4 = sqD_jik + 1;
    const data_t* A0 = A_jik[0], * A1 = A_jik[1], * A2 = A_jik[2], * A3 = A_jik[3],
                * A4 = A_jik[4], * A5 = A_jik[5], * A6 = A_jik[6];
    const data_t one_minus_wgt = 1.0 - wgt;
    for (idx_t k = 0; k < num; k += stride) {
        data_t diag_val = A3[k] * sqD_jik[k] * sqD_jik[k];
        data_t tmp = A0[k] * x0[k] * sqD0[k] + A1[k] * x1[k] * sqD1[k] + A2[k] * x2[k] * sqD2[k]
                    +A4[k] * x4[k] * sqD4[k] + A5[k] * x5[k] * sqD5[k] + A6[k] * x6[k] * sqD6[k];
        x_jik[k] = one_minus_wgt * x_jik[k] + wgt * (b_jik[k] - tmp * sqD_jik[k]) / diag_val;
    }
}


#endif