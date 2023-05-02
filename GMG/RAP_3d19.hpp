#ifndef SMG_RAP_3D19_HPP
#define SMG_RAP_3D19_HPP

#include "../utils/common.hpp"

#define C_IDX(i, j, k)   (k) + (i) * cz + (j) * czcx
#define F_IDX(i, j, k)   (k) + (i) * fz + (j) * fzfx
// 当本进程不在某维度的边界时，可以直接算（前提要求细网格的halo区已填充，但R采用4点时似乎也不用？）
#define CHECK_BDR(CI, CJ, CK) \
    (ilb==false || (CI) >= cibeg) && (iub==false || (CI) < ciend) && \
    (jlb==false || (CJ) >= cjbeg) && (jub==false || (CJ) < cjend) && \
    (klb==false || (CK) >= ckbeg) && (kub==false || (CK) < ckend)

template<typename idx_t, typename data_t>
void RCell2d4_A3d19_PCell2d16(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, const idx_t fx, const idx_t fy, const idx_t fz, 
    // 粗网格的数据，以及各维度的存储大小（含halo区）
        data_t * coar_mat, const idx_t cx, const idx_t cy, const idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [cjbeg, cjend) x [ckbeg, ckend)
    const idx_t cibeg, const idx_t ciend, const idx_t cjbeg, const idx_t cjend, const idx_t ckbeg, const idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // jlb, jub 分别记录本进程在j维度是否在前边界和后边界
    const bool jlb , const bool jub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // 实际就是halo区的宽度，要求粗、细网格的宽度相同
    const idx_t hx, const idx_t hy, const idx_t hz)
{
    for (idx_t i = 0; i < 27 * cx * cy * cz; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }

    const data_t (*AF)[19] = (data_t (*)[19])fine_mat;
    data_t (*AC)[27] = (data_t (*)[27])coar_mat;

    const idx_t czcx = cz * cx;// cz * cx
    const idx_t fzfx = fz * fx;// fz * fx
	#pragma omp parallel for collapse(3) schedule(static)
    for (idx_t J = cjbeg; J < cjend; J++)
    for (idx_t I = cibeg; I < ciend; I++)
    for (idx_t K = ckbeg; K < ckend; K++) {
		if (CHECK_BDR(I - 1,J - 1,K - 1)) {// ingb=0
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[8];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[1];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[3];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][2] = res;
		}
		if (CHECK_BDR(I,J - 1,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.0625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[1];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[1];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][3] = res;
		}
		if (CHECK_BDR(I,J - 1,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.0625000 * ptr[4];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][4] = res;
		}
		if (CHECK_BDR(I,J - 1,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.0625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[8];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[1];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[4];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[3];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][8] = res;
		}
		if (CHECK_BDR(I - 1,J,K - 1)) {// ingb=9
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.5625000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.0625000 * ptr[15];
				tmp += 0.5625000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[5];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][9] = res;
		}
		if (CHECK_BDR(I - 1,J,K)) {// ingb=10
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.5625000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.1875000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.0625000 * ptr[14];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][10] = res;
		}
		if (CHECK_BDR(I - 1,J,K + 1)) {// ingb=11
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.5625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[17];
				tmp += 0.5625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][11] = res;
		}
		if (CHECK_BDR(I,J,K - 1)) {// ingb=12
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.5625000 * ptr[8];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.5625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.5625000 * ptr[8];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.5625000 * ptr[8];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.5625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.5625000 * ptr[8];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][12] = res;
		}
		if (CHECK_BDR(I,J,K)) {// ingb=13
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.5625000 * ptr[14];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.0625000 * ptr[14];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][13] = res;
		}
		if (CHECK_BDR(I,J,K + 1)) {// ingb=14
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][14] = res;
		}
		if (CHECK_BDR(I + 1,J,K - 1)) {// ingb=15
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.5625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.0625000 * ptr[15];
				tmp += 0.5625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][15] = res;
		}
		if (CHECK_BDR(I + 1,J,K)) {// ingb=16
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][16] = res;
		}
		if (CHECK_BDR(I + 1,J,K + 1)) {// ingb=17
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.5625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[17];
				tmp += 0.5625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][17] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K - 1)) {// ingb=18
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[15];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[8];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.1875000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[5];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][18] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K)) {// ingb=19
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.1875000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[14];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][19] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K + 1)) {// ingb=20
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[17];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][20] = res;
		}
		if (CHECK_BDR(I,J + 1,K - 1)) {// ingb=21
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[15];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[15];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.0625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][21] = res;
		}
		if (CHECK_BDR(I,J + 1,K)) {// ingb=22
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.0625000 * ptr[14];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.0625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[14];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][22] = res;
		}
		if (CHECK_BDR(I,J + 1,K + 1)) {// ingb=23
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[17];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[17];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.0625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][23] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K - 1)) {// ingb=24
			data_t res = 0.0;
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[15];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[8];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.1875000 * ptr[11];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][24] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K)) {// ingb=25
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][25] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K + 1)) {// ingb=26
			data_t res = 0.0;
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[17];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.1875000 * ptr[13];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][26] = res;
		}
	}
}

template<typename idx_t, typename data_t>
void RVtx2d9_A3d19_PVtx2d9(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, const idx_t fx, const idx_t fy, const idx_t fz, 
    // 粗网格的数据，以及各维度的存储大小（含halo区）
	data_t * coar_mat, const idx_t cx, const idx_t cy, const idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [cjbeg, cjend) x [ckbeg, ckend)
    const idx_t cibeg, const idx_t ciend, const idx_t cjbeg, const idx_t cjend, const idx_t ckbeg, const idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // jlb, jub 分别记录本进程在j维度是否在前边界和后边界
    const bool jlb , const bool jub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // 实际就是halo区的宽度，要求粗、细网格的宽度相同
    const idx_t hx, const idx_t hy, const idx_t hz,
	// 粗网格相对细的偏移(base offset)
    const idx_t box, const idx_t boy, const idx_t boz)
{
	for (idx_t i = 0; i < 27 * cx * cy * cz; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }

    const data_t (*AF)[19] = (data_t (*)[19])fine_mat;
    data_t (*AC)[27] = (data_t (*)[27])coar_mat;

    const idx_t czcx = cz * cx;// cz * cx
    const idx_t fzfx = fz * fx;// fz * fx

	#pragma omp parallel for collapse(3) schedule(static)
    for (idx_t J = cjbeg; J < cjend; J++)
    for (idx_t I = cibeg; I < ciend; I++)
    for (idx_t K = ckbeg; K < ckend; K++) {
		if (CHECK_BDR(I - 1,J - 1,K - 1)) {// ingb=0
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.5000000 * ptr[6];
				tmp += 1.0000000 * ptr[0];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.5000000 * ptr[0];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.5000000 * ptr[0];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][2] = res;
		}
		if (CHECK_BDR(I,J - 1,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][3] = res;
		}
		if (CHECK_BDR(I,J - 1,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 1.0000000 * ptr[4];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[2];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[4];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[6];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[2];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][4] = res;
		}
		if (CHECK_BDR(I,J - 1,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 1.0000000 * ptr[3];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[4];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.5000000 * ptr[12];
				tmp += 1.0000000 * ptr[4];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.5000000 * ptr[4];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[13];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][8] = res;
		}
		if (CHECK_BDR(I - 1,J,K - 1)) {// ingb=9
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[15];
				tmp += 1.0000000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][9] = res;
		}
		if (CHECK_BDR(I - 1,J,K)) {// ingb=10
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[9];
				tmp += 1.0000000 * ptr[14];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[16];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.5000000 * ptr[14];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[9];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[6];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[6];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][10] = res;
		}
		if (CHECK_BDR(I - 1,J,K + 1)) {// ingb=11
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[17];
				tmp += 1.0000000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][11] = res;
		}
		if (CHECK_BDR(I,J,K - 1)) {// ingb=12
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[11];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[15];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[5];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][12] = res;
		}
		if (CHECK_BDR(I,J,K)) {// ingb=13
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[18];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[9];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[12];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[4];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[9];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[16];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[9];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.2500000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[14];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[9];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[9];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][13] = res;
		}
		if (CHECK_BDR(I,J,K + 1)) {// ingb=14
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.2500000 * ptr[10];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[13];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[10];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[17];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[10];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[3];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.2500000 * ptr[10];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[7];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[10];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][14] = res;
		}
		if (CHECK_BDR(I + 1,J,K - 1)) {// ingb=15
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[15];
				tmp += 1.0000000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][15] = res;
		}
		if (CHECK_BDR(I + 1,J,K)) {// ingb=16
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.2500000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[9];
				tmp += 1.0000000 * ptr[18];
				tmp += 0.5000000 * ptr[12];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[16];
				tmp += 1.0000000 * ptr[12];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[9];
				tmp += 1.0000000 * ptr[4];
				tmp += 0.5000000 * ptr[12];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][16] = res;
		}
		if (CHECK_BDR(I + 1,J,K + 1)) {// ingb=17
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[13];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[13];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[17];
				tmp += 1.0000000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[13];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][17] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K - 1)) {// ingb=18
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][18] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K)) {// ingb=19
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[16];
				tmp += 0.5000000 * ptr[14];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.5000000 * ptr[6];
				tmp += 1.0000000 * ptr[14];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[14];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.5000000 * ptr[14];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][19] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K + 1)) {// ingb=20
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][20] = res;
		}
		if (CHECK_BDR(I,J + 1,K - 1)) {// ingb=21
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[15];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[15];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 1.0000000 * ptr[15];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[15];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][21] = res;
		}
		if (CHECK_BDR(I,J + 1,K)) {// ingb=22
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 1.0000000 * ptr[18];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[16];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.2500000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 1.0000000 * ptr[16];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[6];
				tmp += 1.0000000 * ptr[14];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[16];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][22] = res;
		}
		if (CHECK_BDR(I,J + 1,K + 1)) {// ingb=23
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[17];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[17];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 1.0000000 * ptr[17];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[17];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][23] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K - 1)) {// ingb=24
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][24] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K)) {// ingb=25
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[16];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.5000000 * ptr[12];
				tmp += 1.0000000 * ptr[18];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][25] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K + 1)) {// ingb=26
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.5000000 * ptr[13];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][26] = res;
		}
	}
}


template<typename idx_t, typename data_t>
void RVtxOD2d9_A3d19_PVtxOD2d9(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, const idx_t fx, const idx_t fy, const idx_t fz, 
    // 粗网格的数据，以及各维度的存储大小（含halo区）
	data_t * coar_mat, const idx_t cx, const idx_t cy, const idx_t cz,
	// 限制矩阵和插值矩阵，大小规格需要同粗网格矩阵一样
    const data_t * R_mat, const data_t * P_mat,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [cjbeg, cjend) x [ckbeg, ckend)
    const idx_t cibeg, const idx_t ciend, const idx_t cjbeg, const idx_t cjend, const idx_t ckbeg, const idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // jlb, jub 分别记录本进程在j维度是否在前边界和后边界
    const bool jlb , const bool jub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // 实际就是halo区的宽度，要求粗、细网格的宽度相同
    const idx_t hx, const idx_t hy, const idx_t hz,
	// 粗网格相对细的偏移(base offset)
    const idx_t box, const idx_t boy, const idx_t boz)
{
	// 原来的有错，删掉了，需要时重新生成一下
	MPI_Abort(MPI_COMM_WORLD, -20230502);
}

#undef C_IDX
#undef F_IDX
#undef CHECK_BDR

#endif