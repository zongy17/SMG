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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_spmv_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const data_t * dummy)
{
    const calc_t * x_jNi = x_jik - vec_ki_size, * x_jPi = x_jik + vec_ki_size;
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_spmv_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * A_jik, const calc_t * x_jik, calc_t * y_jik, const data_t * sqD_jik)
{
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const calc_t * sqD_jNi = sqD_jik - vec_ki_size, * sqD_jPi = sqD_jik + vec_ki_size;
    #pragma GCC unroll (4)
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp = 
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
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_zero_3d7_irr(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy,
    const idx_t irr_k, const calc_t irr_contrib)
{
    const calc_t * x_jNi   = x_jik   - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[3];
        calc_t tmp = 
            + L_jik[0] * x_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ]
            + L_jik[2] * x_jik[               k-1];// L * x_{k+1}
        tmp = b_jik[k] - tmp;// b - L*x_{k+1}
        if (k == irr_k) tmp -= irr_contrib;
        // int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        // printf("proc %d irr_k %d contrib %.7e tmp %.7e\n", my_pid, irr_k, irr_contrib, tmp);
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 4;// 下三角部分包含对角线
    }
}


template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t * x_jNi   = x_jik   - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[3];
        calc_t tmp = 
            + L_jik[0] * x_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ]
            + L_jik[2] * x_jik[               k-1];// L * x_{k+1}
        tmp = b_jik[k] - tmp;// b - L*x_{k+1}
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 4;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_zero_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    const calc_t * x_jNi   = x_jik   - vec_ki_size;
    const calc_t * sqD_jNi = sqD_jik - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[3] * sqD_jik[k] * sqD_jik[k];
        calc_t tmp = 
            + L_jik[0] * x_jNi[               k  ] * sqD_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ] * sqD_jik[- vec_k_size + k  ]
            + L_jik[2] * x_jik[               k-1] * sqD_jik[               k-1];// L * x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - L*x_{k+1}
        
        x_jik[k] = wgt * tmp / diag_val;
        L_jik += 4;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_ALL_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[3];// = U_jik[3]
        calc_t tmp = 
        + L_jik[0] * x_jNi[               k    ]
        + L_jik[1] * x_jik[- vec_k_size + k    ]
        + L_jik[2] * x_jik[               k - 1]
        // + A_jik[3] * x_jik[               k    ]
        + U_jik[0] * x_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik += 4;
        U_jik += 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_ALL_3d7_irr(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy,
    const idx_t irr_k, const calc_t irr_contrib)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[3];// = U_jik[3]
        calc_t tmp = 
        + L_jik[0] * x_jNi[               k    ]
        + L_jik[1] * x_jik[- vec_k_size + k    ]
        + L_jik[2] * x_jik[               k - 1]
        // + A_jik[3] * x_jik[               k    ]
        + U_jik[0] * x_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}
        if (k == irr_k) tmp -= irr_contrib;

        // int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
        // printf("proc %d irr_k %d contrib %.7e tmp %.7e\n", my_pid, irr_k, irr_contrib, tmp);

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik += 4;
        U_jik += 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_forward_ALL_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const calc_t * sqD_jNi = sqD_jik - vec_ki_size, * sqD_jPi = sqD_jik + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t diag_val = L_jik[3] * sqD_jik[k] * sqD_jik[k];
        calc_t tmp = 
        + L_jik[0] * x_jNi[               k    ] * sqD_jNi[               k    ]
        + L_jik[1] * x_jik[- vec_k_size + k    ] * sqD_jik[- vec_k_size + k    ]
        + L_jik[2] * x_jik[               k - 1] * sqD_jik[               k - 1]
        + U_jik[0] * x_jik[               k + 1] * sqD_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ] * sqD_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik += 4;
        U_jik += 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t * x_jPi   = x_jik   + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[3];
        calc_t tmp = 
            + U_jik[0] * x_jik[               k + 1]
            + U_jik[1] * x_jik[  vec_k_size + k    ]
            + U_jik[2] * x_jPi[               k    ];// U*x_{k+1}
            tmp = b_jik[k] - tmp;// b - U*x_{k+1}

            x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_zero_3d7_irr(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy,
    const idx_t irr_k, const calc_t irr_contrib)
{
    const calc_t * x_jPi   = x_jik   + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[3];
        calc_t tmp = 
            + U_jik[0] * x_jik[               k + 1]
            + U_jik[1] * x_jik[  vec_k_size + k    ]
            + U_jik[2] * x_jPi[               k    ];// U*x_{k+1}
            tmp = b_jik[k] - tmp;// b - U*x_{k+1}
            if (k == irr_k) tmp -= irr_contrib;

            x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 4;
    }   
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_zero_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    const calc_t * x_jPi   = x_jik   + vec_ki_size;
    const calc_t * sqD_jPi = sqD_jik + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[3] * sqD_jik[k] * sqD_jik[k];
        calc_t tmp = 
        + U_jik[0] * x_jik[               k + 1] * sqD_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ] * sqD_jPi[               k    ];// U*x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - U*x_{k+1}

        x_jik[k] = wgt * tmp / diag_val;
        U_jik -= 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_ALL_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[3];
        calc_t tmp = 
        + L_jik[0] * x_jNi[               k    ]
        + L_jik[1] * x_jik[- vec_k_size + k    ]
        + L_jik[2] * x_jik[               k - 1]
        + U_jik[0] * x_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik -= 4;
        U_jik -= 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_ALL_3d7_irr(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * dummy,
    const idx_t irr_k, const calc_t irr_contrib)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[3];
        calc_t tmp = 
        + L_jik[0] * x_jNi[               k    ]
        + L_jik[1] * x_jik[- vec_k_size + k    ]
        + L_jik[2] * x_jik[               k - 1]
        + U_jik[0] * x_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp;// b - U*x_{k} - L*x_{k+1}
        if (k == irr_k) tmp -= irr_contrib;

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik -= 4;
        U_jik -= 4;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_point_backward_ALL_3d7_scaled(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size, const data_t wgt,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik, const calc_t * sqD_jik)
{
    const calc_t one_minus_weight = 1.0 - wgt;
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    const calc_t * sqD_jNi = sqD_jik - vec_ki_size, * sqD_jPi = sqD_jik + vec_ki_size;
    const idx_t end = - num;
    for (idx_t k = 0; k > end; k--) {
        calc_t diag_val = U_jik[3] * sqD_jik[k] * sqD_jik[k];
        calc_t tmp = 
        + L_jik[0] * x_jNi[               k    ] * sqD_jNi[               k    ]
        + L_jik[1] * x_jik[- vec_k_size + k    ] * sqD_jik[- vec_k_size + k    ]
        + L_jik[2] * x_jik[               k - 1] * sqD_jik[               k - 1]
        + U_jik[0] * x_jik[               k + 1] * sqD_jik[               k + 1]
        + U_jik[1] * x_jik[  vec_k_size + k    ] * sqD_jik[  vec_k_size + k    ]
        + U_jik[2] * x_jPi[               k    ] * sqD_jPi[               k    ];// U*x_{k} + L*x_{k+1}
        tmp = b_jik[k] - tmp * sqD_jik[k];// b - U*x_{k} - L*x_{k+1}

        x_jik[k] *= one_minus_weight;
        x_jik[k] += wgt * tmp / diag_val;
        L_jik -= 4;
        U_jik -= 4;
    }
}

// ================================= LGS =======================================
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_forward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * dummy, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t * x_jNi   = x_jik   - vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp = 
            + L_jik[0] * x_jNi[               k  ]
            + L_jik[1] * x_jik[- vec_k_size + k  ];// L * x_{k+1}
        rhs[k] = b_jik[k] - tmp;// b - L*x_{k+1}
        L_jik += 2;// 下三角部分包含对角线
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_backward_zero_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * dummy, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
            + U_jik[0] * x_jik[ vec_k_size + k  ]
            + U_jik[1] * x_jPi[              k  ];
        rhs[k] = b_jik[k] - tmp;
        U_jik += 2;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_line_ALL_3d7(const idx_t num,
    const idx_t vec_k_size, const idx_t vec_ki_size,
    const data_t * L_jik, const data_t * U_jik, const calc_t * b_jik, const calc_t * x_jik, calc_t * rhs)
{
    const calc_t * x_jNi   = x_jik   - vec_ki_size, * x_jPi   = x_jik   + vec_ki_size;
    for (idx_t k = 0; k < num; k++) {
        calc_t tmp =
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
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_forward_zero_3d7(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t * x_jNi   = x_jik   - dim_1 * dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        calc_t tmp = 
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_backward_zero_3d7(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t * x_jPi   = x_jik   + dim_1 * dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        calc_t para = U_jik[0];
        calc_t tmp = 
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
#define GROUP_LEN 8
#define NEON_LEN 4
// ============================ SPMV =================================
void inline SOA_spmv_3d7_Cal32Stg16(const int num,
    const int vec_k_size , const int vec_ki_size,
    const __fp16 * Diags[2], const float * x3, float * y_jik)
{
    const __fp16* A0_3 = Diags[0],// 0 ~ 3
                * A4_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ,
                * x2 = x3 - 1          , * x4 = x3 + 1          ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0,
                x0_32_1, x1_32_1, x2_32_1, x3_32_1;
    float32x4_t tmp_0, tmp_1;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float16x8x3_t A4_6_16;
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
        // A4~A6
        A4_6_16 = vld3q_f16(A4_6); A4_6 += GROUP_LEN * 3; __builtin_prefetch(A4_6, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A4_6_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A4_6_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A4_6_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A4_6_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A4_6_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A4_6_16.val[2]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x0_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5), x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x2_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlaq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlaq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlaq_f32(tmp_1, A2_32_1, x2_32_1);
        vst1q_f32(y_jik, tmp_0); vst1q_f32(y_jik + NEON_LEN, tmp_1);
        y_jik += GROUP_LEN; __builtin_prefetch(y_jik,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float16x4x3_t A4_6_16;
        tmp_0 = vdupq_n_f32(0.0);
        // A0~A3
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 此时只剩前半段有效
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
        // A4~A6
        A4_6_16 = vld3_f16(A4_6); A4_6 += NEON_LEN * 3; __builtin_prefetch(A4_6, 0, 0);
        A0_32_0 = vcvt_f32_f16(A4_6_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A4_6_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A4_6_16.val[2]);
        x0_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        x2_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);
        tmp_0 = vmlaq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlaq_f32(tmp_0, A2_32_0 , x2_32_0);
        vst1q_f32(y_jik, tmp_0);
        y_jik += NEON_LEN; __builtin_prefetch(y_jik,1);
    }
    for (k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        y_jik[k] = 
            A0_3[0] * x0[k] + A0_3[1] * x1[k] + A0_3[2] * x2[k] + A0_3[3] * x3[k]
        +   A4_6[0] * x4[k] + A4_6[1] * x5[k] + A4_6[2] * x6[k];
        A0_3 += 4;
        A4_6 += 3;
    }
}

// ============================ PGS ==================================
void inline SOA_point_forward_zero_3d7_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[1], const float * b3, float * x3, const float * dummy)
{// 这里的四条对角线，包含主对角线, 0,1,2,3
    const __fp16* A0_3 = Diags[0];
    const float * x0 = x3 - vec_ki_size,
                * x1 = x3 - vec_k_size ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0,
                x0_32_1, x1_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        // 本柱的x不用读入到向量寄存器内
        // 先把非本柱的算好
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        A3_32_0 = vdivq_f32(vwgts, A3_32_0); A3_32_1 = vdivq_f32(vwgts, A3_32_1);
        vst1q_f32(x3, tmp_0);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1]) * vgetq_lane_f32(A3_32_0, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_0, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_0, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_0, 3);
        x3 += NEON_LEN;
        vst1q_f32(x3, tmp_1);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_1, 0) * x3[-1]) * vgetq_lane_f32(A3_32_1, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_1, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_1, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_1, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_1, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_1, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_1, 3);
        x3 += NEON_LEN; __builtin_prefetch(x3,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 此时只剩前半段有效
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        // 先把非本柱的算好
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        A3_32_0 = vdivq_f32(vwgts, A3_32_0);
        vst1q_f32(x3, tmp_0);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1]) * vgetq_lane_f32(A3_32_0, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_0, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_0, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_0, 3);
        x3 += NEON_LEN; __builtin_prefetch(x3,1);
    }
    for (k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        float diag_val = A0_3[3];
        float tmp = 
        + A0_3[0] * x0[k  ] + A0_3[1] * x1[k  ] + A0_3[2] * x3[k-1];
        tmp = b3[k] - tmp;// b - L*x_{k+1}
        x3[k] = weight * tmp / diag_val;
        A0_3 += 4;
    }
}

void inline SOA_point_forward_zero_3d7_Cal32Stg16_irr(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[1], const float * b3, float * x3, const float * dummy,
    const float * iR/* irregular contribution */)
{
    const __fp16* A0_3 = Diags[0];
    const float * x0 = x3 - vec_ki_size,
                * x1 = x3 - vec_k_size ;
    
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0,
                A0_32_1, A1_32_1, A2_32_1, A3_32_1;
    float32x4_t x0_32_0, x1_32_0,
                x0_32_1, x1_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, irg_0, irg_1;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        irg_0   = vld1q_f32(iR); iR += NEON_LEN; irg_1   = vld1q_f32(iR); iR += NEON_LEN; __builtin_prefetch(iR + NEON_LEN, 0);
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + NEON_LEN, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        // 本柱的x不用读入到向量寄存器内
        // 先把非本柱的算好
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        A3_32_0 = vdivq_f32(vwgts, A3_32_0); A3_32_1 = vdivq_f32(vwgts, A3_32_1);
        tmp_0 = vsubq_f32(tmp_0, irg_0); tmp_1 = vsubq_f32(tmp_1, irg_1);
        vst1q_f32(x3, tmp_0);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1]) * vgetq_lane_f32(A3_32_0, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_0, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_0, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_0, 3);
        x3 += NEON_LEN;
        vst1q_f32(x3, tmp_1);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_1, 0) * x3[-1]) * vgetq_lane_f32(A3_32_1, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_1, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_1, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_1, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_1, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_1, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_1, 3);
        x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, irg_0;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 此时只剩前半段有效
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        irg_0   = vld1q_f32(iR); iR += NEON_LEN; __builtin_prefetch(iR + NEON_LEN, 0);
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + NEON_LEN, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        // 先把非本柱的算好
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        A3_32_0 = vdivq_f32(vwgts, A3_32_0);
        tmp_0 = vsubq_f32(tmp_0, irg_0);
        vst1q_f32(x3, tmp_0);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1]) * vgetq_lane_f32(A3_32_0, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_0, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_0, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_0, 3);
        x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN,1);
    }
    for (k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        float diag_val = A0_3[3];
        float tmp = 
        + A0_3[0] * x0[k  ] + A0_3[1] * x1[k  ] + A0_3[2] * x3[k-1];
        tmp = b3[k] - tmp - iR[k];// b - L*x_{k+1}
        x3[k] = weight * tmp / diag_val;
        A0_3 += 4;
    }
    /*
    for (int k = 0; k < num; k++) {
        float diag_val = A0_3[3];
        float tmp = 
        + A0_3[0] * x0[k  ] + A0_3[1] * x1[k  ] + A0_3[2] * x3[k-1];
        tmp = b3[k] - tmp - iR[k];// b - L*x_{k+1}
        // if (k == irr_k) tmp -= irr_contrib;
        x3[k] = weight * tmp / diag_val;
        A0_3 += 4;
    }*/
}

void inline SOA_point_forward_ALL_3d7_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[2], const float * b3, float * x3, const float * dummy)
{// 0,1,2,3为第0组对角线，3,4,5,6为第1组对角线
    const __fp16* A0_3 = Diags[0], * A3_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, diag_32_0,
                A0_32_1, A1_32_1, A2_32_1, diag_32_1;
    float32x4_t x0_32_0, x1_32_0,
                x0_32_1, x1_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1;
        float A2_buf[GROUP_LEN], A4_buf[GROUP_LEN];// 暂存 wgt*A2/A3 和 wgt*A4/A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// A0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A2
        diag_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); diag_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A3
        diag_32_0 = vdivq_f32(vwgts, diag_32_0); diag_32_1 = vdivq_f32(vwgts, diag_32_1);
        // 此时diag_32存的是w/A3
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A2_buf, A2_32_0); vst1q_f32(A2_buf + NEON_LEN, A2_32_1);
        
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        // U半部分
        A0_3_16 = vld4q_f16(A3_6); A3_6 += GROUP_LEN * 4; __builtin_prefetch(A3_6 + GROUP_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A4_buf, A2_32_0); vst1q_f32(A4_buf + NEON_LEN, A2_32_1);

        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);// x5
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);// x6
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);// 此时tmp存的是b-非本柱的a*x
        float * x_jik = x3;
        res_0 = vld1q_f32(x3); x3 += NEON_LEN     ; res_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; res_1 = vmulq_f32(res_1, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0); res_1 = vmlaq_f32(res_1, tmp_1, diag_32_1);// 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x_jik[-1] + A4_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x_jik[ 0] + A4_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x_jik[ 1] + A4_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x_jik[ 2] + A4_buf[3] * x_jik[4]);
        x_jik[4] = vgetq_lane_f32(res_1, 0) - (A2_buf[4] * x_jik[ 3] + A4_buf[4] * x_jik[5]);
        x_jik[5] = vgetq_lane_f32(res_1, 1) - (A2_buf[5] * x_jik[ 4] + A4_buf[5] * x_jik[6]);
        x_jik[6] = vgetq_lane_f32(res_1, 2) - (A2_buf[6] * x_jik[ 5] + A4_buf[6] * x_jik[7]);
        x_jik[7] = vgetq_lane_f32(res_1, 3) - (A2_buf[7] * x_jik[ 6] + A4_buf[7] * x_jik[8]);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, res_0;
        float A2_buf[NEON_LEN], A4_buf[NEON_LEN];
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        diag_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        diag_32_0 = vdivq_f32(vwgts, diag_32_0);
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A2_buf, A2_32_0);
        
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);

        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        // U半部分
        A0_3_16 = vld4_f16(A3_6); A3_6 += NEON_LEN * 4; __builtin_prefetch(A3_6 + NEON_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A4_buf, A2_32_0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);// x5
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);// x6
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        float * x_jik = x3;
        res_0 = vld1q_f32(x3); x3 += NEON_LEN     ; __builtin_prefetch(x3, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0);
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x_jik[-1] + A4_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x_jik[ 0] + A4_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x_jik[ 1] + A4_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x_jik[ 2] + A4_buf[3] * x_jik[4]);
    }
    for ( k = 0; k < num - max_nk; k++) {
        float diag_val = A0_3[3];
        float tmp = 
            + A0_3[0] * x0[k  ] + A0_3[1] * x1[k] + A0_3[2] * x3[k-1]
            + A3_6[1] * x3[k+1] + A3_6[2] * x5[k] + A3_6[3] * x6[k  ];
        tmp = b3[k] - tmp;// b - L*x_{k+1}
        x3[k] *= (1.0 - weight);
        x3[k] += weight * tmp / diag_val;
        A0_3 += 4; A3_6 += 4;
    }
}

void inline SOA_point_forward_ALL_3d7_Cal32Stg16_irr(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[2], const float * b3, float * x3, const float * dummy,
    const float * iR)
{// ()
    const __fp16* A0_3 = Diags[0], * A3_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, diag_32_0,
                A0_32_1, A1_32_1, A2_32_1, diag_32_1;
    float32x4_t x0_32_0, x1_32_0,
                x0_32_1, x1_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1, irg_0, irg_1;
        float A2_buf[GROUP_LEN], A4_buf[GROUP_LEN];// 暂存 wgt*A2/A3 和 wgt*A4/A3
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// A0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A2
        diag_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); diag_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A3
        diag_32_0 = vdivq_f32(vwgts, diag_32_0); diag_32_1 = vdivq_f32(vwgts, diag_32_1);
        // 此时diag_32存的是w/A3
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A2_buf, A2_32_0); vst1q_f32(A2_buf + NEON_LEN, A2_32_1);
        
        irg_0   = vld1q_f32(iR); iR += NEON_LEN; irg_1   = vld1q_f32(iR); iR += NEON_LEN; __builtin_prefetch(iR + NEON_LEN, 0);
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + NEON_LEN, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        // U半部分
        A0_3_16 = vld4q_f16(A3_6); A3_6 += GROUP_LEN * 4; __builtin_prefetch(A3_6 + GROUP_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A4_buf, A2_32_0); vst1q_f32(A4_buf + NEON_LEN, A2_32_1);

        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);// x5
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);// x6
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);// 此时tmp存的是b-非本柱的a*x
        tmp_0 = vsubq_f32(tmp_0, irg_0); tmp_1 = vsubq_f32(tmp_1, irg_1);
        float * x_jik = x3;
        res_0 = vld1q_f32(x3); x3 += NEON_LEN     ; res_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3 + NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; res_1 = vmulq_f32(res_1, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0); res_1 = vmlaq_f32(res_1, tmp_1, diag_32_1);// 此时res存的是(1-w)*x + w/A9*(b-非本柱的a*x)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x_jik[-1] + A4_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x_jik[ 0] + A4_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x_jik[ 1] + A4_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x_jik[ 2] + A4_buf[3] * x_jik[4]);
        x_jik[4] = vgetq_lane_f32(res_1, 0) - (A2_buf[4] * x_jik[ 3] + A4_buf[4] * x_jik[5]);
        x_jik[5] = vgetq_lane_f32(res_1, 1) - (A2_buf[5] * x_jik[ 4] + A4_buf[5] * x_jik[6]);
        x_jik[6] = vgetq_lane_f32(res_1, 2) - (A2_buf[6] * x_jik[ 5] + A4_buf[6] * x_jik[7]);
        x_jik[7] = vgetq_lane_f32(res_1, 3) - (A2_buf[7] * x_jik[ 6] + A4_buf[7] * x_jik[8]);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, res_0, irg_0;
        float A2_buf[NEON_LEN], A4_buf[NEON_LEN];
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);
        diag_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        diag_32_0 = vdivq_f32(vwgts, diag_32_0);
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A2_buf, A2_32_0);
        
        irg_0   = vld1q_f32(iR); iR += NEON_LEN; __builtin_prefetch(iR + NEON_LEN, 0);
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + NEON_LEN, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);

        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        // U半部分
        A0_3_16 = vld4_f16(A3_6); A3_6 += NEON_LEN * 4; __builtin_prefetch(A3_6 + NEON_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A4_buf, A2_32_0);
        x1_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);// x5
        x0_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);// x6
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vsubq_f32(tmp_0, irg_0);
        float * x_jik = x3;
        res_0 = vld1q_f32(x3); x3 += NEON_LEN     ; __builtin_prefetch(x3 + NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0);
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x_jik[-1] + A4_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x_jik[ 0] + A4_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x_jik[ 1] + A4_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x_jik[ 2] + A4_buf[3] * x_jik[4]);
    }
    for ( k = 0; k < num - max_nk; k++) {
        float diag_val = A0_3[3];
        float tmp = 
            + A0_3[0] * x0[k  ] + A0_3[1] * x1[k] + A0_3[2] * x3[k-1]
            + A3_6[1] * x3[k+1] + A3_6[2] * x5[k] + A3_6[3] * x6[k  ];
        tmp = b3[k] - tmp - iR[k];
        x3[k] *= (1.0 - weight);
        x3[k] += weight * tmp / diag_val;
        A0_3 += 4; A3_6 += 4;
    }
}


void inline SOA_point_backward_ALL_3d7_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[2], const float * b3, float * x3, const float * dummy)
{// 0,1,2,3为第0组对角线，3,4,5,6为第1组对角线
    const __fp16* A0_3 = Diags[0], * A3_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, diag_32_0,
                A0_32_1, A1_32_1, A2_32_1, diag_32_1;
    float32x4_t x0_32_0, x1_32_0,
                x0_32_1, x1_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1;
        float A2_buf[GROUP_LEN], A4_buf[GROUP_LEN];// 暂存 wgt*A2/A3 和 wgt*A4/A3
        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// A0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A2
        diag_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); diag_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A3
        diag_32_0 = vdivq_f32(vwgts, diag_32_0); diag_32_1 = vdivq_f32(vwgts, diag_32_1);
        // 此时diag_32存的是w/A3
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A2_buf, A2_32_0); vst1q_f32(A2_buf + NEON_LEN, A2_32_1);
        // 注意这里载入的顺序
        b3 -= NEON_LEN; tmp_1 = vld1q_f32(b3)  ; b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3)  ; __builtin_prefetch(b3 - GROUP_LEN, 0);
        x0 -= NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - GROUP_LEN, 0);
        x1 -= NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        // U半部分
        A3_6 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A3_6); __builtin_prefetch(A3_6 - GROUP_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A4_buf, A2_32_0); vst1q_f32(A4_buf + NEON_LEN, A2_32_1);

        x5 -= NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - GROUP_LEN, 0);// x5
        x6 -= NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x0_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - GROUP_LEN, 0);// x6
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);// 此时tmp存的是b-非本柱的a*x
        x3 -= NEON_LEN; res_1 = vld1q_f32(x3); x3 -= NEON_LEN; res_0 = vld1q_f32(x3); __builtin_prefetch(x3 - GROUP_LEN, 1);
        res_1 = vmulq_f32(res_1, vone_minus_wgts); res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_1 = vmlaq_f32(res_1, tmp_1, diag_32_1); res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0); // 此时res存的是(1-w)*x + w*(b-非本柱的a*x)
        x3[7] = vgetq_lane_f32(res_1, 3) - (A2_buf[7] * x3[ 6] + A4_buf[7] * x3[8]);
        x3[6] = vgetq_lane_f32(res_1, 2) - (A2_buf[6] * x3[ 5] + A4_buf[6] * x3[7]);
        x3[5] = vgetq_lane_f32(res_1, 1) - (A2_buf[5] * x3[ 4] + A4_buf[5] * x3[6]);
        x3[4] = vgetq_lane_f32(res_1, 0) - (A2_buf[4] * x3[ 3] + A4_buf[4] * x3[5]);
        x3[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x3[ 2] + A4_buf[3] * x3[4]);
        x3[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x3[ 1] + A4_buf[2] * x3[3]);
        x3[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x3[ 0] + A4_buf[1] * x3[2]);
        x3[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x3[-1] + A4_buf[0] * x3[1]);
    }// 循环结束时 k==min_gk
    assert(k == min_gk);
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, res_0;
        float A2_buf[NEON_LEN], A4_buf[NEON_LEN];// 暂存 wgt*A2/A3 和 wgt*A4/A3
        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// A0
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// A1
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// A2
        diag_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// A3
        diag_32_0 = vdivq_f32(vwgts, diag_32_0);
        // 此时diag_32存的是w/A3
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A2_buf, A2_32_0);
        // 注意这里载入的顺序
        b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3)  ; __builtin_prefetch(b3 - NEON_LEN, 0);
        x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - NEON_LEN, 0);
        x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        // U半部分
        A3_6 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A3_6); __builtin_prefetch(A3_6 - NEON_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A4_buf, A2_32_0);

        x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - NEON_LEN, 0);// x5
        x6 -= NEON_LEN; x0_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - NEON_LEN, 0);// x6
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);// 此时tmp存的是b-非本柱的a*x
        x3 -= NEON_LEN; res_0 = vld1q_f32(x3); __builtin_prefetch(x3 - NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0); // 此时res存的是(1-w)*x + w*(b-非本柱的a*x)
        x3[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x3[ 2] + A4_buf[3] * x3[4]);
        x3[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x3[ 1] + A4_buf[2] * x3[3]);
        x3[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x3[ 0] + A4_buf[1] * x3[2]);
        x3[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x3[-1] + A4_buf[0] * x3[1]);
    }// 循环结束时 k==min_nk
    assert(k == min_nk);
    A0_3 -= 4; A3_6 -= 4;
    x0 -= min_nk; x1 -= min_nk;
    x3 -= min_nk;
    x5 -= min_nk; x6 -= min_nk; b3 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A0_3[3];
        float tmp = 
            + A0_3[0] * x0[k  ] + A0_3[1] * x1[k] + A0_3[2] * x3[k-1]
            + A3_6[1] * x3[k+1] + A3_6[2] * x5[k] + A3_6[3] * x6[k  ];
        tmp = b3[k] - tmp;
        x3[k] *= (1.0 - weight);
        x3[k] += weight * tmp / diag_val;
        A0_3 -= 4; A3_6 -= 4;
    }
}

void inline SOA_point_backward_ALL_3d7_Cal32Stg16_irr(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[2], const float * b3, float * x3, const float * dummy,
    const float * iR)
{
    const __fp16* A0_3 = Diags[0], * A3_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, diag_32_0,
                A0_32_1, A1_32_1, A2_32_1, diag_32_1;
    float32x4_t x0_32_0, x1_32_0,
                x0_32_1, x1_32_1;
    float32x4_t vwgts = vdupq_n_f32(weight);
    float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    iR += num;
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, res_0, res_1, irg_0, irg_1;
        float A2_buf[GROUP_LEN], A4_buf[GROUP_LEN];// 暂存 wgt*A2/A3 和 wgt*A4/A3
        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// A0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A2
        diag_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); diag_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A3
        diag_32_0 = vdivq_f32(vwgts, diag_32_0); diag_32_1 = vdivq_f32(vwgts, diag_32_1);
        // 此时diag_32存的是w/A3
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A2_buf, A2_32_0); vst1q_f32(A2_buf + NEON_LEN, A2_32_1);
        // 注意这里载入的顺序
        iR -= NEON_LEN; irg_1 = vld1q_f32(iR)  ; iR -= NEON_LEN; irg_0 = vld1q_f32(iR)  ; __builtin_prefetch(iR - GROUP_LEN, 0);
        b3 -= NEON_LEN; tmp_1 = vld1q_f32(b3)  ; b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3)  ; __builtin_prefetch(b3 - GROUP_LEN, 0);
        x0 -= NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - GROUP_LEN, 0);
        x1 -= NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        // U半部分
        A3_6 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A3_6); __builtin_prefetch(A3_6 - GROUP_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0); A2_32_1 = vmulq_f32(A2_32_1, diag_32_1);
        vst1q_f32(A4_buf, A2_32_0); vst1q_f32(A4_buf + NEON_LEN, A2_32_1);

        x5 -= NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - GROUP_LEN, 0);// x5
        x6 -= NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x0_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - GROUP_LEN, 0);// x6
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_1 = vsubq_f32(tmp_1, irg_1); tmp_0 = vsubq_f32(tmp_0, irg_0);
        x3 -= NEON_LEN; res_1 = vld1q_f32(x3); x3 -= NEON_LEN; res_0 = vld1q_f32(x3); __builtin_prefetch(x3 - GROUP_LEN, 1);
        res_1 = vmulq_f32(res_1, vone_minus_wgts); res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_1 = vmlaq_f32(res_1, tmp_1, diag_32_1); res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0); // 此时res存的是(1-w)*x + w*(b-非本柱的a*x)
        x3[7] = vgetq_lane_f32(res_1, 3) - (A2_buf[7] * x3[ 6] + A4_buf[7] * x3[8]);
        x3[6] = vgetq_lane_f32(res_1, 2) - (A2_buf[6] * x3[ 5] + A4_buf[6] * x3[7]);
        x3[5] = vgetq_lane_f32(res_1, 1) - (A2_buf[5] * x3[ 4] + A4_buf[5] * x3[6]);
        x3[4] = vgetq_lane_f32(res_1, 0) - (A2_buf[4] * x3[ 3] + A4_buf[4] * x3[5]);
        x3[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x3[ 2] + A4_buf[3] * x3[4]);
        x3[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x3[ 1] + A4_buf[2] * x3[3]);
        x3[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x3[ 0] + A4_buf[1] * x3[2]);
        x3[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x3[-1] + A4_buf[0] * x3[1]);
    }// 循环结束时 k==min_gk
    assert(k == min_gk);
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, res_0, irg_0;
        float A2_buf[NEON_LEN], A4_buf[NEON_LEN];// 暂存 wgt*A2/A3 和 wgt*A4/A3
        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// A0
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// A1
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// A2
        diag_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// A3
        diag_32_0 = vdivq_f32(vwgts, diag_32_0);
        // 此时diag_32存的是w/A3
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A2_buf, A2_32_0);
        // 注意这里载入的顺序
        iR -= NEON_LEN; irg_0 = vld1q_f32(iR)  ; __builtin_prefetch(iR - NEON_LEN, 0);
        b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3)  ; __builtin_prefetch(b3 - NEON_LEN, 0);
        x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - NEON_LEN, 0);
        x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); 
        // U半部分
        A3_6 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A3_6); __builtin_prefetch(A3_6 - NEON_LEN * 16, 0, 0);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// A4
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// A5
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// A6
        A2_32_0 = vmulq_f32(A2_32_0, diag_32_0);
        vst1q_f32(A4_buf, A2_32_0);

        x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - NEON_LEN, 0);// x5
        x6 -= NEON_LEN; x0_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - NEON_LEN, 0);// x6
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_0 = vsubq_f32(tmp_0, irg_0);
        x3 -= NEON_LEN; res_0 = vld1q_f32(x3); __builtin_prefetch(x3 - NEON_LEN, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_0 = vmlaq_f32(res_0, tmp_0, diag_32_0); // 此时res存的是(1-w)*x + w*(b-非本柱的a*x)
        x3[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x3[ 2] + A4_buf[3] * x3[4]);
        x3[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x3[ 1] + A4_buf[2] * x3[3]);
        x3[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x3[ 0] + A4_buf[1] * x3[2]);
        x3[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x3[-1] + A4_buf[0] * x3[1]);
    }// 循环结束时 k==min_nk
    assert(k == min_nk);
    A0_3 -= 4; A3_6 -= 4;
    x0 -= min_nk; x1 -= min_nk;
    x3 -= min_nk;
    x5 -= min_nk; x6 -= min_nk; b3 -= min_nk; iR -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A0_3[3];
        float tmp = 
            + A0_3[0] * x0[k  ] + A0_3[1] * x1[k] + A0_3[2] * x3[k-1]
            + A3_6[1] * x3[k+1] + A3_6[2] * x5[k] + A3_6[3] * x6[k  ];
        tmp = b3[k] - tmp - iR[k];
        x3[k] *= (1.0 - weight);
        x3[k] += weight * tmp / diag_val;
        A0_3 -= 4; A3_6 -= 4;
    }
}

// ========================= PGS (Scaled) ==============================

void inline SOA_point_forward_zero_3d7_Cal32Stg16_scaled(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[1], const float * b3, float * x3, const float * sqD3)
{// (0,1,2,3)
    const __fp16* A0_3 = Diags[0];
    const float * x0 = x3 - vec_ki_size, * sqD0 = sqD3 - vec_ki_size,
                * x1 = x3 - vec_k_size , * sqD1 = sqD3 - vec_k_size, * sqD2 = sqD3 - 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, D0_32_0, D1_32_0, D2_32_0, D3_32_0;
    float32x4_t vwgts = vdupq_n_f32(weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1, D0_32_1, D1_32_1, D2_32_1, D3_32_1;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应本柱的 x2
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        D0_32_0 = vld1q_f32(sqD0); sqD0 += NEON_LEN; D0_32_1 = vld1q_f32(sqD0); sqD0 += NEON_LEN; __builtin_prefetch(sqD0, 0);
        D1_32_0 = vld1q_f32(sqD1); sqD1 += NEON_LEN; D1_32_1 = vld1q_f32(sqD1); sqD1 += NEON_LEN; __builtin_prefetch(sqD1, 0);
        D2_32_0 = vld1q_f32(sqD2); sqD2 += NEON_LEN; D2_32_1 = vld1q_f32(sqD2); sqD2 += NEON_LEN; __builtin_prefetch(sqD2, 0);
        D3_32_0 = vld1q_f32(sqD3); sqD3 += NEON_LEN; D3_32_1 = vld1q_f32(sqD3); sqD3 += NEON_LEN; __builtin_prefetch(sqD3, 0);
        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0); A0_32_1 = vmulq_f32(A0_32_1, D0_32_1);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); A1_32_1 = vmulq_f32(A1_32_1, D1_32_1);// 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); A2_32_1 = vmulq_f32(A2_32_1, D2_32_1);//   本柱的 Abarj*Qj
        A3_32_0 = vmulq_f32(A3_32_0, D3_32_0); A3_32_1 = vmulq_f32(A3_32_1, D3_32_1);//   自己的 Abari*Qi
        A3_32_0 = vdivq_f32(vwgts  , A3_32_0); A3_32_1 = vdivq_f32(vwgts  , A3_32_1);// w/(Abari*Qi)

        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        tmp_0   = vdivq_f32(tmp_0  , D3_32_0);  tmp_1   = vdivq_f32(tmp_1  , D3_32_1);// b/Qi
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);// b/Qi - 非本柱的 Abarj*xj*Qj

        vst1q_f32(x3, tmp_0);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1]) * vgetq_lane_f32(A3_32_0, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_0, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_0, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_0, 3);
        x3 += NEON_LEN;
        vst1q_f32(x3, tmp_1);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_1, 0) * x3[-1]) * vgetq_lane_f32(A3_32_1, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_1, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_1, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_1, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_1, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_1, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_1, 3);
        x3 += NEON_LEN; __builtin_prefetch(x3,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应本柱的 x2
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        D0_32_0 = vld1q_f32(sqD0); sqD0 += NEON_LEN; __builtin_prefetch(sqD0, 0);
        D1_32_0 = vld1q_f32(sqD1); sqD1 += NEON_LEN; __builtin_prefetch(sqD1, 0);
        D2_32_0 = vld1q_f32(sqD2); sqD2 += NEON_LEN; __builtin_prefetch(sqD2, 0);
        D3_32_0 = vld1q_f32(sqD3); sqD3 += NEON_LEN; __builtin_prefetch(sqD3, 0);
        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0);// 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0);//   本柱的 Abarj*Qj
        A3_32_0 = vmulq_f32(A3_32_0, D3_32_0);//   自己的 Abari*Qi
        A3_32_0 = vdivq_f32(vwgts  , A3_32_0);// w/(Abari*Qi)

        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        tmp_0   = vdivq_f32(tmp_0  , D3_32_0);// b/Qi
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);// b/Qi - 非本柱的 Abarj*xj*Qj

        vst1q_f32(x3, tmp_0);
        x3[0] = (x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1]) * vgetq_lane_f32(A3_32_0, 0);
        x3[1] = (x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0]) * vgetq_lane_f32(A3_32_0, 1);
        x3[2] = (x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1]) * vgetq_lane_f32(A3_32_0, 2);
        x3[3] = (x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2]) * vgetq_lane_f32(A3_32_0, 3);
        x3 += NEON_LEN; __builtin_prefetch(x3,1);
    }
    for (k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        float diag_val = A0_3[3] * sqD3[k];
        float tmp = 
        + A0_3[0]*x0[k]*sqD0[k] + A0_3[1]*x1[k]*sqD1[k] + A0_3[2]*x3[k-1]*sqD2[k];
        tmp = b3[k]/sqD3[k] - tmp;// b - L*x_{k+1}
        x3[k] = weight * tmp / diag_val;
        A0_3 += 4;
    }
}

void inline SOA_point_forward_ALL_3d7_Cal32Stg16_scaled(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[2], const float * b3, float * x3, const float * sqD3)
{// (0,1,2,3) (3,4,5,6)
    const __fp16* A0_3 = Diags[0], * A3_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    const float * sqD0 = sqD3 - vec_ki_size, * sqD6 = sqD3 + vec_ki_size,
                * sqD1 = sqD3 - vec_k_size , * sqD5 = sqD3 + vec_k_size ,
                * sqD2 = sqD3 - 1          , * sqD4 = sqD3 + 1          ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, D0_32_0, D1_32_0, D2_32_0, D3_32_0;
    const float32x4_t vwgts = vdupq_n_f32(weight);
    const float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, res_0, tmp_1, res_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1, D0_32_1, D1_32_1, D2_32_1, D3_32_1;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4q_f16(A0_3); A0_3 += GROUP_LEN * 4; __builtin_prefetch(A0_3 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应本柱的 x2
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        D0_32_0 = vld1q_f32(sqD0); sqD0 += NEON_LEN; D0_32_1 = vld1q_f32(sqD0); sqD0 += NEON_LEN; __builtin_prefetch(sqD0, 0);
        D1_32_0 = vld1q_f32(sqD1); sqD1 += NEON_LEN; D1_32_1 = vld1q_f32(sqD1); sqD1 += NEON_LEN; __builtin_prefetch(sqD1, 0);
        D2_32_0 = vld1q_f32(sqD2); sqD2 += NEON_LEN; D2_32_1 = vld1q_f32(sqD2); sqD2 += NEON_LEN; __builtin_prefetch(sqD2, 0);
        D3_32_0 = vld1q_f32(sqD3); sqD3 += NEON_LEN; D3_32_1 = vld1q_f32(sqD3); sqD3 += NEON_LEN; __builtin_prefetch(sqD3, 0);

        A3_32_0 = vmulq_f32(A3_32_0, D3_32_0); A3_32_1 = vmulq_f32(A3_32_1, D3_32_1);//   自己的 Abari*Qi
        A3_32_0 = vdivq_f32(vwgts  , A3_32_0); A3_32_1 = vdivq_f32(vwgts  , A3_32_1);// w/(Abari*Qi)

        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0); A0_32_1 = vmulq_f32(A0_32_1, D0_32_1);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); A1_32_1 = vmulq_f32(A1_32_1, D1_32_1);// 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); A2_32_1 = vmulq_f32(A2_32_1, D2_32_1);//   本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, A3_32_0); A2_32_1 = vmulq_f32(A2_32_1, A3_32_1);// 本柱的 w*Abarj*Qj/(Abari*Qi)
        float A2_buf[GROUP_LEN];
        vst1q_f32(A2_buf, A2_32_0); vst1q_f32(A2_buf + NEON_LEN, A2_32_1);

        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        tmp_0   = vdivq_f32(tmp_0  , D3_32_0);  tmp_1   = vdivq_f32(tmp_1  , D3_32_1);// b/Qi
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);// b/Qi - 非本柱的 Abarj*xj*Qj
        // A3 ~ A6
        A0_3_16 = vld4q_f16(A3_6); A3_6 += GROUP_LEN * 4; __builtin_prefetch(A3_6 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应x5
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// 对应x6
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应本柱的 x4
        D0_32_0 = vld1q_f32(sqD5); sqD5 += NEON_LEN; D0_32_1 = vld1q_f32(sqD5); sqD5 += NEON_LEN; __builtin_prefetch(sqD5, 0);
        D1_32_0 = vld1q_f32(sqD6); sqD6 += NEON_LEN; D1_32_1 = vld1q_f32(sqD6); sqD6 += NEON_LEN; __builtin_prefetch(sqD6, 0);
        D2_32_0 = vld1q_f32(sqD4); sqD4 += NEON_LEN; D2_32_1 = vld1q_f32(sqD4); sqD4 += NEON_LEN; __builtin_prefetch(sqD4, 0);
        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0); A0_32_1 = vmulq_f32(A0_32_1, D0_32_1);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); A1_32_1 = vmulq_f32(A1_32_1, D1_32_1);// 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); A2_32_1 = vmulq_f32(A2_32_1, D2_32_1);//   本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, A3_32_0); A2_32_1 = vmulq_f32(A2_32_1, A3_32_1);// 本柱的 w*Abarj*Qj/(Abari*Qi)
        float A4_buf[GROUP_LEN];
        vst1q_f32(A4_buf, A2_32_0); vst1q_f32(A4_buf + NEON_LEN, A2_32_1);

        x0_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x0_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        x1_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x1_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);// b/Qi - 非本柱的 Abarj*xj*Qj
        
        float * x_jik = x3;
        res_0 = vld1q_f32(x3); x3 += NEON_LEN     ; res_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ; res_1 = vmulq_f32(res_1, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, A3_32_0)  ; res_1 = vmlaq_f32(res_1, tmp_1, A3_32_1);// 此时res存的是(1-w)*x + w/(Abari*Qi)*(b/Qi - 非本柱的Abarj*Qj*xj)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x_jik[-1] + A4_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x_jik[ 0] + A4_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x_jik[ 1] + A4_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x_jik[ 2] + A4_buf[3] * x_jik[4]);
        x_jik[4] = vgetq_lane_f32(res_1, 0) - (A2_buf[4] * x_jik[ 3] + A4_buf[4] * x_jik[5]);
        x_jik[5] = vgetq_lane_f32(res_1, 1) - (A2_buf[5] * x_jik[ 4] + A4_buf[5] * x_jik[6]);
        x_jik[6] = vgetq_lane_f32(res_1, 2) - (A2_buf[6] * x_jik[ 5] + A4_buf[6] * x_jik[7]);
        x_jik[7] = vgetq_lane_f32(res_1, 3) - (A2_buf[7] * x_jik[ 6] + A4_buf[7] * x_jik[8]);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, res_0;
        // A0~A3，其中A3为对角线
        A0_3_16 = vld4_f16(A0_3); A0_3 += NEON_LEN * 4; __builtin_prefetch(A0_3 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应本柱的 x2
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        D0_32_0 = vld1q_f32(sqD0); sqD0 += NEON_LEN; __builtin_prefetch(sqD0, 0);
        D1_32_0 = vld1q_f32(sqD1); sqD1 += NEON_LEN; __builtin_prefetch(sqD1, 0);
        D2_32_0 = vld1q_f32(sqD2); sqD2 += NEON_LEN; __builtin_prefetch(sqD2, 0);
        D3_32_0 = vld1q_f32(sqD3); sqD3 += NEON_LEN; __builtin_prefetch(sqD3, 0);
        A3_32_0 = vmulq_f32(A3_32_0, D3_32_0);//   自己的 Abari*Qi
        A3_32_0 = vdivq_f32(vwgts  , A3_32_0);// w/(Abari*Qi)

        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0);// 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0);//   本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, A3_32_0);
        float A2_buf[NEON_LEN];
        vst1q_f32(A2_buf, A2_32_0);

        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN;  __builtin_prefetch(b3, 0);
        tmp_0   = vdivq_f32(tmp_0  , D3_32_0);// b/Qi
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);// b/Qi - 非本柱的 Abarj*xj*Qj
        // A3 ~ A6
        A0_3_16 = vld4_f16(A3_6); A3_6 += NEON_LEN * 4; __builtin_prefetch(A3_6 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应x5
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// 对应x6
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应本柱的 x4
        D0_32_0 = vld1q_f32(sqD5); sqD5 += NEON_LEN; __builtin_prefetch(sqD5, 0);
        D1_32_0 = vld1q_f32(sqD6); sqD6 += NEON_LEN; __builtin_prefetch(sqD6, 0);
        D2_32_0 = vld1q_f32(sqD4); sqD4 += NEON_LEN; __builtin_prefetch(sqD4, 0);
        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0);// 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0);//   本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, A3_32_0);
        float A4_buf[NEON_LEN];
        vst1q_f32(A4_buf, A2_32_0);

        x0_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        x1_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);// b/Qi - 非本柱的 Abarj*xj*Qj
        
        float * x_jik = x3;
        res_0 = vld1q_f32(x3); x3 += NEON_LEN     ; __builtin_prefetch(x3, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts) ;
        res_0 = vmlaq_f32(res_0, tmp_0, A3_32_0)  ;// 此时res存的是(1-w)*x + w/(Abari*Qi)*(b/Qi - 非本柱的Abarj*Qj*xj)
        x_jik[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x_jik[-1] + A4_buf[0] * x_jik[1]);
        x_jik[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x_jik[ 0] + A4_buf[1] * x_jik[2]);
        x_jik[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x_jik[ 1] + A4_buf[2] * x_jik[3]);
        x_jik[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x_jik[ 2] + A4_buf[3] * x_jik[4]);
    }
    for (k = 0; k < num - max_nk; k++) {// 做完剩下的元素
        float diag_val = A0_3[3] * sqD3[k];
        float tmp = 
        + A0_3[0]*x0[k]*sqD0[k] + A0_3[1]*x1[k]*sqD1[k] + A0_3[2]*x3[k-1]*sqD2[k]
        + A3_6[1]*x3[k+1]*sqD4[k] + A3_6[2]*x5[k]*sqD5[k] + A3_6[3]*x6[k]*sqD6[k];
        tmp = b3[k]/sqD3[k] - tmp;// b - L*x_{k+1}
        x3[k] *= (1.0 - weight);
        x3[k] += weight * tmp / diag_val;
        A0_3 += 4; A3_6 += 4;
    }
}

void inline SOA_point_backward_ALL_3d7_Cal32Stg16_scaled(const int num,
    const int vec_k_size, const int vec_ki_size, const float weight,
    const __fp16 * Diags[2], const float * b3, float * x3, const float * sqD3)
{// (0,1,2,3) (3,4,5,6)
    const __fp16* A0_3 = Diags[0], * A3_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    const float * sqD0 = sqD3 - vec_ki_size, * sqD6 = sqD3 + vec_ki_size,
                * sqD1 = sqD3 - vec_k_size , * sqD5 = sqD3 + vec_k_size ,
                * sqD2 = sqD3 - 1          , * sqD4 = sqD3 + 1          ;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, D0_32_0, D1_32_0, D2_32_0, D3_32_0;
    const float32x4_t vwgts = vdupq_n_f32(weight);
    const float32x4_t vone_minus_wgts = vdupq_n_f32(1.0 - weight);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, res_0, tmp_1, res_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1, D0_32_1, D1_32_1, D2_32_1, D3_32_1;
        // A0~A3，其中A3为对角线
        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应本柱的 x2
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);
        sqD0 -= NEON_LEN; D0_32_1 = vld1q_f32(sqD0); sqD0 -= NEON_LEN; D0_32_0 = vld1q_f32(sqD0); __builtin_prefetch(sqD0, 0);
        sqD1 -= NEON_LEN; D1_32_1 = vld1q_f32(sqD1); sqD1 -= NEON_LEN; D1_32_0 = vld1q_f32(sqD1); __builtin_prefetch(sqD1, 0);
        sqD2 -= NEON_LEN; D2_32_1 = vld1q_f32(sqD2); sqD2 -= NEON_LEN; D2_32_0 = vld1q_f32(sqD2); __builtin_prefetch(sqD2, 0);
        sqD3 -= NEON_LEN; D3_32_1 = vld1q_f32(sqD3); sqD3 -= NEON_LEN; D3_32_0 = vld1q_f32(sqD3); __builtin_prefetch(sqD3, 0);
        A3_32_1 = vmulq_f32(A3_32_1, D3_32_1); A3_32_0 = vmulq_f32(A3_32_0, D3_32_0); //   自己的 Abari*Qi
        A3_32_1 = vdivq_f32(vwgts  , A3_32_1); A3_32_0 = vdivq_f32(vwgts  , A3_32_0); // w/(Abari*Qi)
        A0_32_1 = vmulq_f32(A0_32_1, D0_32_1); A0_32_0 = vmulq_f32(A0_32_0, D0_32_0); 
        A1_32_1 = vmulq_f32(A1_32_1, D1_32_1); A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); // 非本柱的 Abarj*Qj
        A2_32_1 = vmulq_f32(A2_32_1, D2_32_1); A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); //   本柱的 Abarj*Qj
        A2_32_1 = vmulq_f32(A2_32_1, A3_32_1); A2_32_0 = vmulq_f32(A2_32_0, A3_32_0); // 本柱的 w*Abarj*Qj/(Abari*Qi)
        float A2_buf[GROUP_LEN];
        vst1q_f32(A2_buf, A2_32_0); vst1q_f32(A2_buf + NEON_LEN, A2_32_1);

        b3 -= NEON_LEN; tmp_1   = vld1q_f32(b3); b3 -= NEON_LEN; tmp_0   = vld1q_f32(b3); __builtin_prefetch(b3, 0);
        tmp_1   = vdivq_f32(tmp_1  , D3_32_1);  tmp_0   = vdivq_f32(tmp_0  , D3_32_0);  // b/Qi
        x0 -= NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0, 0);
        x1 -= NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1, 0);// 非本柱的 xj
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); // b/Qi - 非本柱的 Abarj*xj*Qj
        // A3 ~ A6
        A3_6 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A3_6); __builtin_prefetch(A3_6 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应x5
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// 对应x6
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应本柱的 x4
        sqD5 -= NEON_LEN; D0_32_1 = vld1q_f32(sqD5); sqD5 -= NEON_LEN; D0_32_0 = vld1q_f32(sqD5); __builtin_prefetch(sqD5, 0);
        sqD6 -= NEON_LEN; D1_32_1 = vld1q_f32(sqD6); sqD6 -= NEON_LEN; D1_32_0 = vld1q_f32(sqD6); __builtin_prefetch(sqD6, 0);
        sqD4 -= NEON_LEN; D2_32_1 = vld1q_f32(sqD4); sqD4 -= NEON_LEN; D2_32_0 = vld1q_f32(sqD4); __builtin_prefetch(sqD4, 0);
        A0_32_1 = vmulq_f32(A0_32_1, D0_32_1); A0_32_0 = vmulq_f32(A0_32_0, D0_32_0);
        A1_32_1 = vmulq_f32(A1_32_1, D1_32_1); A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); // 非本柱的 Abarj*Qj
        A2_32_1 = vmulq_f32(A2_32_1, D2_32_1); A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); //   本柱的 Abarj*Qj
        A2_32_1 = vmulq_f32(A2_32_1, A3_32_1); A2_32_0 = vmulq_f32(A2_32_0, A3_32_0); // 本柱的 w*Abarj*Qj/(Abari*Qi)
        float A4_buf[GROUP_LEN];
        vst1q_f32(A4_buf, A2_32_0); vst1q_f32(A4_buf + NEON_LEN, A2_32_1);

        x5 -= NEON_LEN; x0_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x0_32_0 = vld1q_f32(x5); __builtin_prefetch(x5, 0);
        x6 -= NEON_LEN; x1_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x1_32_0 = vld1q_f32(x6); __builtin_prefetch(x6, 0);// 非本柱的 xj
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); // b/Qi - 非本柱的 Abarj*xj*Qj
        
        x3 -= NEON_LEN; res_1 = vld1q_f32(x3); x3 -= NEON_LEN; res_0 = vld1q_f32(x3); __builtin_prefetch(x3, 1);
        res_1 = vmulq_f32(res_1, vone_minus_wgts); res_0 = vmulq_f32(res_0, vone_minus_wgts);
        res_1 = vmlaq_f32(res_1, tmp_1, A3_32_1) ; res_0 = vmlaq_f32(res_0, tmp_0, A3_32_0) ; // 此时res存的是(1-w)*x + w/(Abari*Qi)*(b/Qi - 非本柱的Abarj*Qj*xj)
        x3[7] = vgetq_lane_f32(res_1, 3) - (A2_buf[7] * x3[ 6] + A4_buf[7] * x3[8]);
        x3[6] = vgetq_lane_f32(res_1, 2) - (A2_buf[6] * x3[ 5] + A4_buf[6] * x3[7]);
        x3[5] = vgetq_lane_f32(res_1, 1) - (A2_buf[5] * x3[ 4] + A4_buf[5] * x3[6]);
        x3[4] = vgetq_lane_f32(res_1, 0) - (A2_buf[4] * x3[ 3] + A4_buf[4] * x3[5]);
        x3[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x3[ 2] + A4_buf[3] * x3[4]);
        x3[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x3[ 1] + A4_buf[2] * x3[3]);
        x3[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x3[ 0] + A4_buf[1] * x3[2]);
        x3[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x3[-1] + A4_buf[0] * x3[1]);
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0, res_0;
        // A0~A3，其中A3为对角线
        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应本柱的 x2
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);
        sqD0 -= NEON_LEN; D0_32_0 = vld1q_f32(sqD0); __builtin_prefetch(sqD0, 0);
        sqD1 -= NEON_LEN; D1_32_0 = vld1q_f32(sqD1); __builtin_prefetch(sqD1, 0);
        sqD2 -= NEON_LEN; D2_32_0 = vld1q_f32(sqD2); __builtin_prefetch(sqD2, 0);
        sqD3 -= NEON_LEN; D3_32_0 = vld1q_f32(sqD3); __builtin_prefetch(sqD3, 0);
        A3_32_0 = vmulq_f32(A3_32_0, D3_32_0); //   自己的 Abari*Qi
        A3_32_0 = vdivq_f32(vwgts  , A3_32_0); // w/(Abari*Qi)
        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0); 
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); // 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); //   本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, A3_32_0); // 本柱的 w*Abarj*Qj/(Abari*Qi)
        float A2_buf[NEON_LEN];
        vst1q_f32(A2_buf, A2_32_0);

        b3 -= NEON_LEN; tmp_0   = vld1q_f32(b3); __builtin_prefetch(b3, 0);
        tmp_0   = vdivq_f32(tmp_0  , D3_32_0);  // b/Qi
        x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0, 0);
        x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); // b/Qi - 非本柱的 Abarj*xj*Qj
        // A3 ~ A6
        A3_6 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A3_6); __builtin_prefetch(A3_6 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应x5
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// 对应x6
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应本柱的 x4
        sqD5 -= NEON_LEN; D0_32_0 = vld1q_f32(sqD5); __builtin_prefetch(sqD5, 0);
        sqD6 -= NEON_LEN; D1_32_0 = vld1q_f32(sqD6); __builtin_prefetch(sqD6, 0);
        sqD4 -= NEON_LEN; D2_32_0 = vld1q_f32(sqD4); __builtin_prefetch(sqD4, 0);
        A0_32_0 = vmulq_f32(A0_32_0, D0_32_0);
        A1_32_0 = vmulq_f32(A1_32_0, D1_32_0); // 非本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, D2_32_0); //   本柱的 Abarj*Qj
        A2_32_0 = vmulq_f32(A2_32_0, A3_32_0); // 本柱的 w*Abarj*Qj/(Abari*Qi)
        float A4_buf[NEON_LEN];
        vst1q_f32(A4_buf, A2_32_0);

        x5 -= NEON_LEN; x0_32_0 = vld1q_f32(x5); __builtin_prefetch(x5, 0);
        x6 -= NEON_LEN; x1_32_0 = vld1q_f32(x6); __builtin_prefetch(x6, 0);// 非本柱的 xj
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); // b/Qi - 非本柱的 Abarj*xj*Qj
        
        x3 -= NEON_LEN; res_0 = vld1q_f32(x3); __builtin_prefetch(x3, 1);
        res_0 = vmulq_f32(res_0, vone_minus_wgts);
        res_0 = vmlaq_f32(res_0, tmp_0, A3_32_0) ; // 此时res存的是(1-w)*x + w/(Abari*Qi)*(b/Qi - 非本柱的Abarj*Qj*xj)
        x3[3] = vgetq_lane_f32(res_0, 3) - (A2_buf[3] * x3[ 2] + A4_buf[3] * x3[4]);
        x3[2] = vgetq_lane_f32(res_0, 2) - (A2_buf[2] * x3[ 1] + A4_buf[2] * x3[3]);
        x3[1] = vgetq_lane_f32(res_0, 1) - (A2_buf[1] * x3[ 0] + A4_buf[1] * x3[2]);
        x3[0] = vgetq_lane_f32(res_0, 0) - (A2_buf[0] * x3[-1] + A4_buf[0] * x3[1]);
    }
    A0_3 -= 4; A3_6 -= 4;
    x0 -= min_nk; x1 -= min_nk; sqD0 -= min_nk; sqD1 -= min_nk; sqD2 -= min_nk;
    x3 -= min_nk; sqD3 -= min_nk; b3 -= min_nk;
    x5 -= min_nk; x6 -= min_nk; sqD5 -= min_nk; sqD6 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {// 做完剩下的元素
        float diag_val = A0_3[3] * sqD3[k];
        float tmp = 
        + A0_3[0]*x0[k]*sqD0[k] + A0_3[1]*x1[k]*sqD1[k] + A0_3[2]*x3[k-1]*sqD2[k]
        + A3_6[1]*x3[k+1]*sqD4[k] + A3_6[2]*x5[k]*sqD5[k] + A3_6[3]*x6[k]*sqD6[k];
        tmp = b3[k]/sqD3[k] - tmp;// b - L*x_{k+1}
        x3[k] *= (1.0 - weight);
        x3[k] += weight * tmp / diag_val;
        A0_3 -= 4; A3_6 -= 4;
    }
}

// ============================ LGS ===================================
void inline SOA_line_forward_zero_3d7_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[1], const float * b3, const float * x3, float * rhs)
{// (0,1)
    const __fp16* A0_1 = Diags[0];
    const float * x0 = x3 - vec_ki_size,
                * x1 = x3 - vec_k_size ;
    float32x4_t A0_32_0, A1_32_0;
    float32x4_t x0_32_0, x1_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x2_t A0_1_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, A0_32_1, A1_32_1;
        tmp_0 = vld1q_f32(b3); b3 += NEON_LEN; tmp_1 = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + GROUP_LEN, 0);
        // A0 ~ A1
        A0_1_16 = vld2q_f16(A0_1); A0_1 += GROUP_LEN * 2; __builtin_prefetch(A0_1 + GROUP_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_1_16.val[1]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);

        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; vst1q_f32(rhs, tmp_1); rhs += NEON_LEN; __builtin_prefetch(rhs + GROUP_LEN, 1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x2_t A0_1_16;
        float32x4_t tmp_0;
        tmp_0 = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + NEON_LEN, 0);
        // A0 ~ A1
        A0_1_16 = vld2_f16(A0_1); A0_1 += NEON_LEN * 2; __builtin_prefetch(A0_1 + NEON_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_1_16.val[1]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; __builtin_prefetch(rhs + NEON_LEN, 1);
    }
    for (k = 0; k < num - max_nk; k++) {
        rhs[k] = b3[k] - A0_1[0] * x0[k] - A0_1[1] * x1[k];
        A0_1 += 2;
    }
}

void inline SOA_line_forward_ALL_3d7_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[2], const float * b3, const float * x3, float * rhs)
{// (0,1) (5,6)
    const __fp16* A0_1 = Diags[0], * A5_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    float32x4_t A0_32_0, A1_32_0;
    float32x4_t x0_32_0, x1_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = num & (~(GROUP_LEN - 1)), max_nk = num & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x2_t A0_1_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, A0_32_1, A1_32_1;
        tmp_0 = vld1q_f32(b3); b3 += NEON_LEN; tmp_1 = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + GROUP_LEN, 0);
        // A0 ~ A1
        A0_1_16 = vld2q_f16(A0_1); A0_1 += GROUP_LEN * 2; __builtin_prefetch(A0_1 + GROUP_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_1_16.val[1]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        // A5 ~ A6
        A0_1_16 = vld2q_f16(A5_6); A5_6 += GROUP_LEN * 2; __builtin_prefetch(A5_6 + GROUP_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_1_16.val[1]);
        x0_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x0_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + GROUP_LEN, 0);
        x1_32_0 = vld1q_f32(x6); x6 += NEON_LEN; x1_32_1 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + GROUP_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);

        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; vst1q_f32(rhs, tmp_1); rhs += NEON_LEN; __builtin_prefetch(rhs + GROUP_LEN, 1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x2_t A0_1_16;
        float32x4_t tmp_0;
        tmp_0 = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3 + NEON_LEN, 0);
        // A0 ~ A1
        A0_1_16 = vld2_f16(A0_1); A0_1 += NEON_LEN * 2; __builtin_prefetch(A0_1 + NEON_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_1_16.val[1]);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        // A5 ~ A6
        A0_1_16 = vld2_f16(A5_6); A5_6 += NEON_LEN * 2; __builtin_prefetch(A5_6 + NEON_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_1_16.val[1]);
        x0_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5 + NEON_LEN, 0);
        x1_32_0 = vld1q_f32(x6); x6 += NEON_LEN; __builtin_prefetch(x6 + NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);

        vst1q_f32(rhs, tmp_0); rhs += NEON_LEN; __builtin_prefetch(rhs + NEON_LEN, 1);
    }
    for (k = 0; k < num - max_nk; k++) {
        rhs[k] = b3[k] - A0_1[0] * x0[k] - A0_1[1] * x1[k] - A5_6[0] * x5[k] - A5_6[1] * x6[k];
        A0_1 += 2; A5_6 += 2;
    }
}

void inline SOA_line_backward_ALL_3d7_Cal32Stg16(const int num,
    const int vec_k_size, const int vec_ki_size,
    const __fp16 * Diags[2], const float * b3, const float * x3, float * rhs)
{// (0,1) (5,6)
    const __fp16* A0_1 = Diags[0], * A5_6 = Diags[1];
    const float * x0 = x3 - vec_ki_size, * x6 = x3 + vec_ki_size,
                * x1 = x3 - vec_k_size , * x5 = x3 + vec_k_size ;
    float32x4_t A0_32_0, A1_32_0;
    float32x4_t x0_32_0, x1_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = num, min_gk = num & (GROUP_LEN - 1), min_nk = num & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x2_t A0_1_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, A0_32_1, A1_32_1;
        b3 -= NEON_LEN; tmp_1 = vld1q_f32(b3); b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3); __builtin_prefetch(b3 - GROUP_LEN, 0);
        // A0 ~ A1
        A0_1 -= GROUP_LEN * 2;
        A0_1_16 = vld2q_f16(A0_1); __builtin_prefetch(A0_1 - GROUP_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_1_16.val[1]);
        x0 -= NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - GROUP_LEN, 0);
        x1 -= NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        // A5 ~ A6
        A5_6 -= GROUP_LEN * 2;
        A0_1_16 = vld2q_f16(A5_6); __builtin_prefetch(A5_6 - GROUP_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_1_16.val[0]);// A5
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_1_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_1_16.val[1]);
        x5 -= NEON_LEN; x0_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x0_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - GROUP_LEN, 0);
        x6 -= NEON_LEN; x1_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x1_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);

        rhs -= NEON_LEN; vst1q_f32(rhs, tmp_1); rhs -= NEON_LEN; vst1q_f32(rhs, tmp_0); __builtin_prefetch(rhs - GROUP_LEN, 1);
        // printf("inside %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n", rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5], rhs[6], rhs[7]);
    }
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x2_t A0_1_16;
        float32x4_t tmp_0;
        b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3); __builtin_prefetch(b3 - NEON_LEN, 0);
        // A0 ~ A1
        A0_1 -= NEON_LEN * 2;
        A0_1_16 = vld2_f16(A0_1); __builtin_prefetch(A0_1 - NEON_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_1_16.val[1]);
        x0 -= NEON_LEN; x0_32_0 = vld1q_f32(x0); __builtin_prefetch(x0 - NEON_LEN, 0);
        x1 -= NEON_LEN; x1_32_0 = vld1q_f32(x1); __builtin_prefetch(x1 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        // A5 ~ A6
        A5_6 -= NEON_LEN * 2;
        A0_1_16 = vld2_f16(A5_6); __builtin_prefetch(A5_6 - NEON_LEN * 8, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_1_16.val[0]);
        A1_32_0 = vcvt_f32_f16(A0_1_16.val[1]);
        x5 -= NEON_LEN; x0_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - NEON_LEN, 0);
        x6 -= NEON_LEN; x1_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); 
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);

        rhs -= NEON_LEN; vst1q_f32(rhs, tmp_0); __builtin_prefetch(rhs - NEON_LEN, 1);
    }
    A0_1 -= 2; A5_6 -= 2;
    x0 -= min_nk; x1 -= min_nk; x5 -= min_nk; x6 -= min_nk; b3 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        rhs[k] = b3[k] - A0_1[0] * x0[k] - A0_1[1] * x1[k] - A5_6[0] * x5[k] - A5_6[1] * x6[k];
        A0_1 -= 2; A5_6 -= 2;
    }
    
}

// ========================== BILU ===================================
void inline SOA_ilu_forward_zero_3d7_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[1], const float * b3, float * x3)
{// L(0,1,2)
    const __fp16 * A0_2 = Diags[0];
    const float * x0 = x3 - dim_1 * dim_2,
                * x1 = x3 -         dim_2;
    float32x4_t A0_32_0, A1_32_0, A2_32_0;
    float32x4_t x0_32_0, x1_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = dim_2 & (~(GROUP_LEN - 1)), max_nk = dim_2 & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x3_t A0_2_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, A0_32_1, A1_32_1, A2_32_1;
        A0_2_16 = vld3q_f16(A0_2); A0_2 += GROUP_LEN * 3; __builtin_prefetch(A0_2 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_2_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_2_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_2_16.val[2]);// 对应本柱的x2
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; tmp_1   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        // 本柱的x不用读入到向量寄存器内
        // 先把非本柱的算好
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        vst1q_f32(x3, tmp_0);
        x3[0] = x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1];
        x3[1] = x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0];
        x3[2] = x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1];
        x3[3] = x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2];
        x3 += NEON_LEN;
        vst1q_f32(x3, tmp_1);
        x3[0] = x3[0] - vgetq_lane_f32(A2_32_1, 0) * x3[-1];
        x3[1] = x3[1] - vgetq_lane_f32(A2_32_1, 1) * x3[ 0];
        x3[2] = x3[2] - vgetq_lane_f32(A2_32_1, 2) * x3[ 1];
        x3[3] = x3[3] - vgetq_lane_f32(A2_32_1, 3) * x3[ 2];
        x3 += NEON_LEN; __builtin_prefetch(x3,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x3_t A0_2_16;
        float32x4_t tmp_0;
        // A0~A3，其中A3为对角线
        A0_2_16 = vld3_f16(A0_2); A0_2 += NEON_LEN * 3; __builtin_prefetch(A0_2 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_2_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(A0_2_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(A0_2_16.val[2]);// 对应本柱的x2
        tmp_0   = vld1q_f32(b3); b3 += NEON_LEN; __builtin_prefetch(b3, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        // 先把非本柱的算好
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        vst1q_f32(x3, tmp_0);
        x3[0] = x3[0] - vgetq_lane_f32(A2_32_0, 0) * x3[-1];
        x3[1] = x3[1] - vgetq_lane_f32(A2_32_0, 1) * x3[ 0];
        x3[2] = x3[2] - vgetq_lane_f32(A2_32_0, 2) * x3[ 1];
        x3[3] = x3[3] - vgetq_lane_f32(A2_32_0, 3) * x3[ 2];
        x3 += NEON_LEN; __builtin_prefetch(x3,1);
    }
    for (k = 0; k < dim_2 - max_nk; k++) {// 做完剩下的元素
        float tmp = 
        + A0_2[0]*x0[k  ] + A0_2[1]*x1[k  ] + A0_2[2]*x3[k-1];
        x3[k] = b3[k] - tmp;// b - L*x_{k+1}
        A0_2 += 3;
    }
}

void inline SOA_ilu_backward_zero_3d7_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[1], const float * b3, float * x3)
{// U(0,1,2,3) 其中U(0)是主对角元
    const __fp16* A0_3 = Diags[0];
    const float * x6 = x3 + dim_1 * dim_2,
                * x5 = x3 +         dim_2;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0;
    float32x4_t vones = vdupq_n_f32(1.0);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = dim_2, min_gk = dim_2 & (GROUP_LEN - 1), min_nk = dim_2 & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1;
        float A4_buf[GROUP_LEN];// 暂存本柱的 A4/A3
        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应本柱的 x4
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// x5
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// x6
        A0_32_0 = vdivq_f32(vones, A0_32_0)                 ; A0_32_1 = vdivq_f32(vones, A0_32_1);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0)               ; A1_32_1 = vmulq_f32(A1_32_1, A0_32_1);// A4/A3
        vst1q_f32(A4_buf, A1_32_0)                          ; vst1q_f32(A4_buf + NEON_LEN, A1_32_1);
        // 注意这里载入的顺序
        b3 -= NEON_LEN; tmp_1 = vld1q_f32(b3)  ; b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3)  ; __builtin_prefetch(b3 - GROUP_LEN, 0);
        x5 -= NEON_LEN; x1_32_1 = vld1q_f32(x5); x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - GROUP_LEN, 0);
        x6 -= NEON_LEN; x0_32_1 = vld1q_f32(x6); x6 -= NEON_LEN; x0_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x0_32_0);// 此时tmp存的是b-非本柱的a*x
        x3 -= GROUP_LEN; __builtin_prefetch(x3 - GROUP_LEN, 1);
        tmp_1 = vmulq_f32(tmp_1, A0_32_1)                   ; tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x3[7] = vgetq_lane_f32(tmp_1, 3) - A4_buf[7] * x3[8];
        x3[6] = vgetq_lane_f32(tmp_1, 2) - A4_buf[6] * x3[7];
        x3[5] = vgetq_lane_f32(tmp_1, 1) - A4_buf[5] * x3[6];
        x3[4] = vgetq_lane_f32(tmp_1, 0) - A4_buf[4] * x3[5];
        x3[3] = vgetq_lane_f32(tmp_0, 3) - A4_buf[3] * x3[4];
        x3[2] = vgetq_lane_f32(tmp_0, 2) - A4_buf[2] * x3[3];
        x3[1] = vgetq_lane_f32(tmp_0, 1) - A4_buf[1] * x3[2];
        x3[0] = vgetq_lane_f32(tmp_0, 0) - A4_buf[0] * x3[1];
    }// 循环结束时 k==min_gk
    assert(k == min_gk);
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        float A4_buf[NEON_LEN];// 暂存本柱的 A4/A3
        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应本柱的 x4
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// x5
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// x6
        A0_32_0 = vdivq_f32(vones, A0_32_0);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0);// A4/A3
        vst1q_f32(A4_buf, A1_32_0);
        // 注意这里载入的顺序
        b3 -= NEON_LEN; tmp_0 = vld1q_f32(b3)  ; __builtin_prefetch(b3 - NEON_LEN, 0);
        x5 -= NEON_LEN; x1_32_0 = vld1q_f32(x5); __builtin_prefetch(x5 - NEON_LEN, 0);
        x6 -= NEON_LEN; x0_32_0 = vld1q_f32(x6); __builtin_prefetch(x6 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x0_32_0);// 此时tmp存的是b-非本柱的a*x
        x3 -= NEON_LEN; __builtin_prefetch(x3 - NEON_LEN, 1);
        tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x3[3] = vgetq_lane_f32(tmp_0, 3) - A4_buf[3] * x3[4];
        x3[2] = vgetq_lane_f32(tmp_0, 2) - A4_buf[2] * x3[3];
        x3[1] = vgetq_lane_f32(tmp_0, 1) - A4_buf[1] * x3[2];
        x3[0] = vgetq_lane_f32(tmp_0, 0) - A4_buf[0] * x3[1];
    }// 循环结束时 k==min_nk
    assert(k == min_nk);
    A0_3 -= 4;
    x3 -= min_nk; b3 -= min_nk;
    x5 -= min_nk; x6 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A0_3[0];
        float tmp = 
            + A0_3[1]*x3[k+1] + A0_3[2]*x5[k] + A0_3[3]*x6[k];
        x3[k] = (b3[k] - tmp) / diag_val;
        A0_3 -= 4;
    }
}


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
#undef NEON_LEN
#undef GROUP_LEN

#endif