#ifndef SMG_KERNELS_3D15_HPP
#define SMG_KERNELS_3D15_HPP

// ===================================================================
// ========================  3d19  kernels  ==========================
// ===================================================================
 /*
                                                    内存读入后的格式
                    /-----14 ------/                    
                  5 |    8       11|                   
                /---|-- 2 ------/  |
               |    |           |  |
               |    | ----13 ---|--|
   z   y       |  4 |    7      |10|
   ^  ^        | ---|-- 1 ------|  |
   | /         |    |           |  |
   |/          |    /-----12 ---|--/ 
   O-------> x |  3      6      | 9 
               |/------ 0 ------|/
            */

// =================================== SPMV =============================

// =================================== PGS =============================

// =================================== LGS =============================

// =================================== BILU ==================================
template<typename idx_t, typename data_t>
void inline AOS_ilu_forward_zero_3d15(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t* x_jNiZ = x_jik  - dim_1 * dim_2,
                * x_jZiN = x_jik  - dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        data_t tmp = 
            + L_jik[0] * x_jNiZ[k-1]
            + L_jik[1] * x_jNiZ[k  ]
            + L_jik[2] * x_jNiZ[k+1]
            + L_jik[3] * x_jZiN[k-1]
            + L_jik[4] * x_jZiN[k  ]
            + L_jik[5] * x_jZiN[k+1]
            + L_jik[6] * x_jik [k-1];// L * x_{k+1}
        x_jik[k] = b_jik[k] - tmp;
        L_jik += 7;
    }
}

template<typename idx_t, typename data_t>
void inline AOS_ilu_backward_zero_3d15(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const data_t * b_jik, data_t * x_jik)
{
    const data_t* x_jPiZ  = x_jik   + dim_1 * dim_2,
                * x_jZiP  = x_jik   + dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        data_t para = U_jik[0];
        data_t tmp = 
            + U_jik[1] * x_jik [k+1]
            + U_jik[2] * x_jZiP[k-1]
            + U_jik[3] * x_jZiP[k  ]
            + U_jik[4] * x_jZiP[k+1]
            + U_jik[5] * x_jPiZ[k-1]
            + U_jik[6] * x_jPiZ[k  ]
            + U_jik[7] * x_jPiZ[k+1];
        x_jik[k] = (b_jik[k] - tmp) / para;
        U_jik -= 8;
    }
}

#endif