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
template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_forward_zero_3d15(const idx_t dim_2, const idx_t dim_1,
    const data_t * L_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t* x_jNiZ = x_jik  - dim_1 * dim_2,
                * x_jZiN = x_jik  - dim_2;
    for (idx_t k = 0; k < dim_2; k++) {
        calc_t tmp = 
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

template<typename idx_t, typename data_t, typename calc_t>
void inline AOS_ilu_backward_zero_3d15(const idx_t dim_2, const idx_t dim_1,
    const data_t * U_jik, const calc_t * b_jik, calc_t * x_jik)
{
    const calc_t* x_jPiZ  = x_jik   + dim_1 * dim_2,
                * x_jZiP  = x_jik   + dim_2;
    const idx_t end = - dim_2;
    for (idx_t k = 0; k > end; k--) {
        calc_t para = U_jik[0];
        calc_t tmp = 
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

/* 
---------------------------------------------------

    Structure of Array: diagonals separated !!!

-------------------------------------------------------
*/
#define GROUP_LEN 8
#define NEON_LEN 4
// ========================== BILU ===================================
void inline SOA_ilu_forward_zero_3d15_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[2], const float * b7, float * x7)
{// L(0,1,2) (3,4,5,6)
    const __fp16 * A0_2 = Diags[0], * A3_6 = Diags[1];
    const float * x1 = x7 - dim_1 * dim_2,
                * x4 = x7 -         dim_2;
    const float * x0 = x1 - 1, * x2 = x1 + 1, * x3 = x4 - 1, * x5 = x4 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0;
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = 0, max_gk = dim_2 & (~(GROUP_LEN - 1)), max_nk = dim_2 & (~(NEON_LEN - 1));
    for ( ; k < max_gk; k += GROUP_LEN) {
        float16x8x3_t A0_2_16;
        float32x4_t tmp_0, tmp_1, x0_32_1, x1_32_1, x2_32_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1;
        A0_2_16 = vld3q_f16(A0_2); A0_2 += GROUP_LEN * 3; __builtin_prefetch(A0_2 + GROUP_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_2_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_2_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_2_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_2_16.val[2]);// 对应x2
        tmp_0   = vld1q_f32(b7); b7 += NEON_LEN; tmp_1   = vld1q_f32(b7); b7 += NEON_LEN; __builtin_prefetch(b7, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; x0_32_1 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; x1_32_1 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; x2_32_1 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2, 0);
        // 本柱的x不用读入到向量寄存器内
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        float16x8x4_t A3_6_16;
        A3_6_16 = vld4q_f16(A3_6); A3_6 += GROUP_LEN * 4; __builtin_prefetch(A3_6 + GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A3_6_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A3_6_16.val[0]);// 对应x3
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A3_6_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A3_6_16.val[1]);// 对应x4
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A3_6_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A3_6_16.val[2]);// 对应x5
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A3_6_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A3_6_16.val[3]);// 对应本柱的x6
        x0_32_0 = vld1q_f32(x3); x3 += NEON_LEN; x0_32_1 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 0);
        x1_32_0 = vld1q_f32(x4); x4 += NEON_LEN; x1_32_1 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x2_32_0 = vld1q_f32(x5); x5 += NEON_LEN; x2_32_1 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0); tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0); tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0); tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1);
        vst1q_f32(x7, tmp_0);
        x7[0] = x7[0] - vgetq_lane_f32(A3_32_0, 0) * x7[-1];
        x7[1] = x7[1] - vgetq_lane_f32(A3_32_0, 1) * x7[ 0];
        x7[2] = x7[2] - vgetq_lane_f32(A3_32_0, 2) * x7[ 1];
        x7[3] = x7[3] - vgetq_lane_f32(A3_32_0, 3) * x7[ 2];
        x7 += NEON_LEN;
        vst1q_f32(x7, tmp_1);
        x7[0] = x7[0] - vgetq_lane_f32(A3_32_1, 0) * x7[-1];
        x7[1] = x7[1] - vgetq_lane_f32(A3_32_1, 1) * x7[ 0];
        x7[2] = x7[2] - vgetq_lane_f32(A3_32_1, 2) * x7[ 1];
        x7[3] = x7[3] - vgetq_lane_f32(A3_32_1, 3) * x7[ 2];
        x7 += NEON_LEN; __builtin_prefetch(x7,1);
    }
    for ( ; k < max_nk; k += NEON_LEN) {
        float16x4x3_t A0_2_16;
        float32x4_t tmp_0;
        A0_2_16 = vld3_f16(A0_2); A0_2 += NEON_LEN * 3; __builtin_prefetch(A0_2 + NEON_LEN * 12, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_2_16.val[0]);// 对应x0
        A1_32_0 = vcvt_f32_f16(A0_2_16.val[1]);// 对应x1
        A2_32_0 = vcvt_f32_f16(A0_2_16.val[2]);// 对应x2
        tmp_0   = vld1q_f32(b7); b7 += NEON_LEN; __builtin_prefetch(b7, 0);
        x0_32_0 = vld1q_f32(x0); x0 += NEON_LEN; __builtin_prefetch(x0, 0);
        x1_32_0 = vld1q_f32(x1); x1 += NEON_LEN; __builtin_prefetch(x1, 0);
        x2_32_0 = vld1q_f32(x2); x2 += NEON_LEN; __builtin_prefetch(x2, 0);
        // 本柱的x不用读入到向量寄存器内
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        float16x4x4_t A3_6_16;
        A3_6_16 = vld4_f16(A3_6); A3_6 += NEON_LEN * 4; __builtin_prefetch(A3_6 + NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A3_6_16.val[0]);// 对应x3
        A1_32_0 = vcvt_f32_f16(A3_6_16.val[1]);// 对应x4
        A2_32_0 = vcvt_f32_f16(A3_6_16.val[2]);// 对应x5
        A3_32_0 = vcvt_f32_f16(A3_6_16.val[3]);// 对应本柱的x6
        x0_32_0 = vld1q_f32(x3); x3 += NEON_LEN; __builtin_prefetch(x3, 0);
        x1_32_0 = vld1q_f32(x4); x4 += NEON_LEN; __builtin_prefetch(x4, 0);
        x2_32_0 = vld1q_f32(x5); x5 += NEON_LEN; __builtin_prefetch(x5, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        vst1q_f32(x7, tmp_0);
        x7[0] = x7[0] - vgetq_lane_f32(A3_32_0, 0) * x7[-1];
        x7[1] = x7[1] - vgetq_lane_f32(A3_32_0, 1) * x7[ 0];
        x7[2] = x7[2] - vgetq_lane_f32(A3_32_0, 2) * x7[ 1];
        x7[3] = x7[3] - vgetq_lane_f32(A3_32_0, 3) * x7[ 2];
        x7 += NEON_LEN; __builtin_prefetch(x7,1);
    }
    for (k = 0; k < dim_2 - max_nk; k++) {// 做完剩下的元素
        float tmp = 
        + A0_2[0]*x0[k] + A0_2[1]*x1[k] + A0_2[2]*x2[k]
        + A3_6[0]*x3[k] + A3_6[1]*x4[k] + A3_6[2]*x5[k] + A3_6[3]*x7[k-1];
        x7[k] = b7[k] - tmp;// b - L*x_{k+1}
        A0_2 += 3; A3_6 += 4;
    }
}

void inline SOA_ilu_backward_zero_3d15_Cal32Stg16(const int dim_2, const int dim_1,
    const __fp16 * Diags[2], const float * b7, float * x7)
{// U:(0,1,2,3)(4,5,6,7) 
    const __fp16* A0_3 = Diags[0], * A4_7 = Diags[1];
    const float * x13 = x7 + dim_1 * dim_2,
                * x10 = x7 +         dim_2;
    const float * x12 = x13 - 1, * x14 = x13 + 1, * x9 = x10 - 1, * x11 = x10 + 1;
    float32x4_t A0_32_0, A1_32_0, A2_32_0, A3_32_0;
    float32x4_t x0_32_0, x1_32_0, x2_32_0, x3_32_0;
    float32x4_t vones = vdupq_n_f32(1.0);
    static_assert(GROUP_LEN == 8 && NEON_LEN == 4);
    int k = dim_2, min_gk = dim_2 & (GROUP_LEN - 1), min_nk = dim_2 & (NEON_LEN-1);
    for ( ; k > min_gk; k -= GROUP_LEN) {
        float16x8x4_t A0_3_16;
        float32x4_t tmp_0, tmp_1, A0_32_1, A1_32_1, A2_32_1, A3_32_1, x0_32_1, x1_32_1, x2_32_1, x3_32_1;
        A4_7 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A4_7); __builtin_prefetch(A4_7 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 对应x11
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应x12
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// 对应x13
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// 对应x14
        b7 -= NEON_LEN; tmp_1 = vld1q_f32(b7)  ; b7 -= NEON_LEN; tmp_0 = vld1q_f32(b7)  ; __builtin_prefetch(b7 - GROUP_LEN, 0);
        x11-= NEON_LEN; x0_32_1 = vld1q_f32(x11); x11 -= NEON_LEN; x0_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - GROUP_LEN, 0);
        x12-= NEON_LEN; x1_32_1 = vld1q_f32(x12); x12 -= NEON_LEN; x1_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - GROUP_LEN, 0);
        x13-= NEON_LEN; x2_32_1 = vld1q_f32(x13); x13 -= NEON_LEN; x2_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - GROUP_LEN, 0);
        x14-= NEON_LEN; x3_32_1 = vld1q_f32(x14); x14 -= NEON_LEN; x3_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A0_32_1, x0_32_1); tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A1_32_1, x1_32_1); tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);

        A0_3 -= GROUP_LEN * 4;
        A0_3_16 = vld4q_f16(A0_3); __builtin_prefetch(A0_3 - GROUP_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[0])); A0_32_1 = vcvt_high_f32_f16(A0_3_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[1])); A1_32_1 = vcvt_high_f32_f16(A0_3_16.val[1]);// 对应本柱的 x8
        A2_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[2])); A2_32_1 = vcvt_high_f32_f16(A0_3_16.val[2]);// x9
        A3_32_0 = vcvt_f32_f16(vget_low_f16(A0_3_16.val[3])); A3_32_1 = vcvt_high_f32_f16(A0_3_16.val[3]);// x10
        A0_32_0 = vdivq_f32(vones, A0_32_0)                 ; A0_32_1 = vdivq_f32(vones, A0_32_1);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0)               ; A1_32_1 = vmulq_f32(A1_32_1, A0_32_1);// A8/A7
        float A8_buf[GROUP_LEN];// 暂存本柱的 A8/A7
        vst1q_f32(A8_buf, A1_32_0)                          ; vst1q_f32(A8_buf + NEON_LEN, A1_32_1);
        x9 -= NEON_LEN; x2_32_1 = vld1q_f32(x9 ); x9  -= NEON_LEN; x2_32_0 = vld1q_f32(x9 ); __builtin_prefetch(x9  - GROUP_LEN, 0);
        x10-= NEON_LEN; x3_32_1 = vld1q_f32(x10); x10 -= NEON_LEN; x3_32_0 = vld1q_f32(x10); __builtin_prefetch(x10 - GROUP_LEN, 0);
        tmp_1 = vmlsq_f32(tmp_1, A2_32_1, x2_32_1); tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_1 = vmlsq_f32(tmp_1, A3_32_1, x3_32_1); tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_1 = vmulq_f32(tmp_1, A0_32_1)                   ; tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x7 -= GROUP_LEN; __builtin_prefetch(x7 - GROUP_LEN, 1);
        x7[7] = vgetq_lane_f32(tmp_1, 3) - A8_buf[7] * x7[8];
        x7[6] = vgetq_lane_f32(tmp_1, 2) - A8_buf[6] * x7[7];
        x7[5] = vgetq_lane_f32(tmp_1, 1) - A8_buf[5] * x7[6];
        x7[4] = vgetq_lane_f32(tmp_1, 0) - A8_buf[4] * x7[5];
        x7[3] = vgetq_lane_f32(tmp_0, 3) - A8_buf[3] * x7[4];
        x7[2] = vgetq_lane_f32(tmp_0, 2) - A8_buf[2] * x7[3];
        x7[1] = vgetq_lane_f32(tmp_0, 1) - A8_buf[1] * x7[2];
        x7[0] = vgetq_lane_f32(tmp_0, 0) - A8_buf[0] * x7[1];
    }// 循环结束时 k==min_gk
    assert(k == min_gk);
    for ( ; k > min_nk; k -= NEON_LEN) {
        float16x4x4_t A0_3_16;
        float32x4_t tmp_0;
        A4_7 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A4_7); __builtin_prefetch(A4_7 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 对应x11
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应x12
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// 对应x13
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// 对应x14
        b7 -= NEON_LEN; tmp_0 = vld1q_f32(b7)  ; __builtin_prefetch(b7 - NEON_LEN, 0);
        x11 -= NEON_LEN; x0_32_0 = vld1q_f32(x11); __builtin_prefetch(x11 - NEON_LEN, 0);
        x12 -= NEON_LEN; x1_32_0 = vld1q_f32(x12); __builtin_prefetch(x12 - NEON_LEN, 0);
        x13 -= NEON_LEN; x2_32_0 = vld1q_f32(x13); __builtin_prefetch(x13 - NEON_LEN, 0);
        x14 -= NEON_LEN; x3_32_0 = vld1q_f32(x14); __builtin_prefetch(x14 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A0_32_0 , x0_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A1_32_0 , x1_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);

        A0_3 -= NEON_LEN * 4;
        A0_3_16 = vld4_f16(A0_3); __builtin_prefetch(A0_3 - NEON_LEN * 16, 0, 0);
        A0_32_0 = vcvt_f32_f16(A0_3_16.val[0]);// 主对角元
        A1_32_0 = vcvt_f32_f16(A0_3_16.val[1]);// 对应本柱的 x8
        A2_32_0 = vcvt_f32_f16(A0_3_16.val[2]);// x9
        A3_32_0 = vcvt_f32_f16(A0_3_16.val[3]);// x10
        A0_32_0 = vdivq_f32(vones, A0_32_0);// 此时对角线在分母
        A1_32_0 = vmulq_f32(A1_32_0, A0_32_0);// A8/A7
        float A8_buf[NEON_LEN];// 暂存本柱的 A8/A7
        vst1q_f32(A8_buf, A1_32_0);
        x9  -= NEON_LEN; x2_32_0 = vld1q_f32(x9 ); __builtin_prefetch(x9  - NEON_LEN, 0);
        x10 -= NEON_LEN; x3_32_0 = vld1q_f32(x10); __builtin_prefetch(x10 - NEON_LEN, 0);
        tmp_0 = vmlsq_f32(tmp_0, A2_32_0 , x2_32_0);
        tmp_0 = vmlsq_f32(tmp_0, A3_32_0 , x3_32_0);// 此时tmp存的是b-非本柱的a*x
        tmp_0 = vmulq_f32(tmp_0, A0_32_0);
        x7 -= NEON_LEN; __builtin_prefetch(x7 - NEON_LEN, 1);
        x7[3] = vgetq_lane_f32(tmp_0, 3) - A8_buf[3] * x7[4];
        x7[2] = vgetq_lane_f32(tmp_0, 2) - A8_buf[2] * x7[3];
        x7[1] = vgetq_lane_f32(tmp_0, 1) - A8_buf[1] * x7[2];
        x7[0] = vgetq_lane_f32(tmp_0, 0) - A8_buf[0] * x7[1];
    }// 循环结束时 k==min_nk
    assert(k == min_nk);
    A0_3 -= 4; A4_7 -= 4;
    x7 -= min_nk; b7 -= min_nk;
    x9 -= min_nk; x10 -= min_nk; x11 -= min_nk; x12 -= min_nk; x13 -= min_nk; x14 -= min_nk;
    for (k = min_nk - 1; k >= 0; k--) {
        float diag_val = A0_3[0];
        float tmp = 
            +                  A0_3[1]*x7[k+1] + A0_3[2]*x9[k] + A0_3[3]*x10[k]
            + A4_7[0]*x11[k] + A4_7[1]*x12[k] + A4_7[2]*x13[k] + A4_7[3]*x14[k];
        x7[k] = (b7[k] - tmp) / diag_val;
        A0_3 -= 4; A4_7 -= 4;
    }
}
#undef NEON_LEN
#undef GROUP_LEN


#endif