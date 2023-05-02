#ifndef SMG_RAP_2D9_HPP
#define SMG_RAP_2D9_HPP

#include "../utils/common.hpp"

#define C_IDX(i, k)   (k) + (i) * cz
#define F_IDX(i, k)   (k) + (i) * fz
// 当本进程不在某维度的边界时，可以直接算（前提要求细网格的halo区已填充，但R采用4点时似乎也不用？）
#define CHECK_BDR(CI, CK) \
    (ilb==false || (CI) >= cibeg) && (iub==false || (CI) < ciend) && \
    (klb==false || (CK) >= ckbeg) && (kub==false || (CK) < ckend)

template<typename idx_t, typename data_t>
void RVtx2d9_A2d9_PVtx2d9(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, idx_t fx, idx_t fz,
    // 粗网格的数据，以及各维度的存储大小（含halo区）
    data_t * coar_mat, idx_t cx, idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [ckbeg, ckend)
    idx_t cibeg, idx_t ciend, idx_t ckbeg, idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // halo区的宽度
    const idx_t hx, const idx_t hz,
    // 粗网格相对细的偏移(base offset)
    const idx_t box, const idx_t boz)
{
	for (idx_t i = 0; i < 9 * cx * cz; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }
    const data_t (*AF)[9] = (data_t (*)[9])fine_mat;
    data_t (*AC)[9] = (data_t (*)[9])coar_mat;

	#pragma omp parallel for collapse(2) schedule(static)
	for (idx_t I = cibeg; I < ciend; I++)
	for (idx_t K = ckbeg; K < ckend; K++) {
		if (CHECK_BDR(I - 1,K - 1)) {// ingb=0
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[1];
				tmp += 1.0000000 * ptr[0];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.5000000 * ptr[0];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.5000000 * ptr[0];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[4];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[5];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[4];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[1];
				tmp += 1.0000000 * ptr[2];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,K)][2] = res;
		}
		if (CHECK_BDR(I,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[7];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[4];
				tmp += 1.0000000 * ptr[3];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[1];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,K)][3] = res;
		}
		if (CHECK_BDR(I,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += 1.0000000 * ptr[7];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += 1.0000000 * ptr[5];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 1.0000000 * ptr[4];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += 1.0000000 * ptr[3];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,K)][4] = res;
		}
		if (CHECK_BDR(I,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[7];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[4];
				tmp += 1.0000000 * ptr[5];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[1];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[7];
				tmp += 1.0000000 * ptr[6];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[4];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[5];
				tmp += 1.0000000 * ptr[7];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[4];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[7];
				tmp += 1.0000000 * ptr[8];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,K)][8] = res;
		}
	}
}

template<typename idx_t, typename data_t>
void RVtxOD2d9_A2d9_PVtxOD2d9(
    // 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, idx_t fx, idx_t fz,
    // 粗网格的数据，以及各维度的存储大小（含halo区）
    data_t * coar_mat, idx_t cx, idx_t cz,
	// 限制矩阵和插值矩阵，大小规格需要同粗网格矩阵一样
    const data_t * R_mat, const data_t * P_mat,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [ckbeg, ckend)
    idx_t cibeg, idx_t ciend, idx_t ckbeg, idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // halo区的宽度
    const idx_t hx, const idx_t hz,
    // 粗网格相对细的偏移(base offset)
    const idx_t box, const idx_t boz)
{
	for (idx_t i = 0; i < 9 * cx * cz; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }
    const data_t (*AF)[9] = (data_t (*)[9])fine_mat;
    data_t (*AC)[9] = (data_t (*)[9])coar_mat;
	const data_t (*RC)[9] = (data_t (*)[9])R_mat;
    const data_t (*PC)[9] = (data_t (*)[9])P_mat;

	#pragma omp parallel for collapse(2) schedule(static)
	for (idx_t I = cibeg; I < ciend; I++)
	for (idx_t K = ckbeg; K < ckend; K++) {
		const data_t * Rv = RC[C_IDX(I, K)];
		if (CHECK_BDR(I - 1,K - 1)) {// ingb=0
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I - 1,K - 1)];
			{// u_coord[0]=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += Pv[8] * ptr[4];
				tmp += Pv[7] * ptr[3];
				tmp += Pv[5] * ptr[1];
				tmp += Pv[4] * ptr[0];
				res += tmp * Rv[0];
			}
			{// u_coord[1]=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += Pv[8] * ptr[3];
				tmp += Pv[5] * ptr[0];
				res += tmp * Rv[1];
			}
			{// u_coord[3]=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += Pv[8] * ptr[1];
				tmp += Pv[7] * ptr[0];
				res += tmp * Rv[3];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[8] * ptr[0];
				res += tmp * Rv[4];
			}
			AC[C_IDX(I,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,K)) {// ingb=1
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I - 1,K)];
			{// u_coord[0]=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += Pv[7] * ptr[5];
				tmp += Pv[6] * ptr[4];
				tmp += Pv[4] * ptr[2];
				tmp += Pv[3] * ptr[1];
				res += tmp * Rv[0];
			}
			{// u_coord[1]=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += Pv[7] * ptr[4];
				tmp += Pv[6] * ptr[3];
				tmp += Pv[8] * ptr[5];
				tmp += Pv[4] * ptr[1];
				tmp += Pv[3] * ptr[0];
				tmp += Pv[5] * ptr[2];
				res += tmp * Rv[1];
			}
			{// u_coord[2]=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += Pv[7] * ptr[3];
				tmp += Pv[8] * ptr[4];
				tmp += Pv[4] * ptr[0];
				tmp += Pv[5] * ptr[1];
				res += tmp * Rv[2];
			}
			{// u_coord[3]=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += Pv[7] * ptr[2];
				tmp += Pv[6] * ptr[1];
				res += tmp * Rv[3];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[7] * ptr[1];
				tmp += Pv[6] * ptr[0];
				tmp += Pv[8] * ptr[2];
				res += tmp * Rv[4];
			}
			{// u_coord[5]=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += Pv[7] * ptr[0];
				tmp += Pv[8] * ptr[1];
				res += tmp * Rv[5];
			}
			AC[C_IDX(I,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I - 1,K + 1)];
			{// u_coord[1]=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += Pv[6] * ptr[5];
				tmp += Pv[3] * ptr[2];
				res += tmp * Rv[1];
			}
			{// u_coord[2]=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += Pv[6] * ptr[4];
				tmp += Pv[7] * ptr[5];
				tmp += Pv[3] * ptr[1];
				tmp += Pv[4] * ptr[2];
				res += tmp * Rv[2];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[6] * ptr[2];
				res += tmp * Rv[4];
			}
			{// u_coord[5]=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += Pv[6] * ptr[1];
				tmp += Pv[7] * ptr[2];
				res += tmp * Rv[5];
			}
			AC[C_IDX(I,K)][2] = res;
		}
		if (CHECK_BDR(I,K - 1)) {// ingb=3
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I,K - 1)];
			{// u_coord[0]=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += Pv[5] * ptr[7];
				tmp += Pv[4] * ptr[6];
				tmp += Pv[2] * ptr[4];
				tmp += Pv[1] * ptr[3];
				res += tmp * Rv[0];
			}
			{// u_coord[1]=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += Pv[5] * ptr[6];
				tmp += Pv[2] * ptr[3];
				res += tmp * Rv[1];
			}
			{// u_coord[3]=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += Pv[5] * ptr[4];
				tmp += Pv[4] * ptr[3];
				tmp += Pv[2] * ptr[1];
				tmp += Pv[1] * ptr[0];
				tmp += Pv[8] * ptr[7];
				tmp += Pv[7] * ptr[6];
				res += tmp * Rv[3];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[5] * ptr[3];
				tmp += Pv[2] * ptr[0];
				tmp += Pv[8] * ptr[6];
				res += tmp * Rv[4];
			}
			{// u_coord[6]=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += Pv[5] * ptr[1];
				tmp += Pv[4] * ptr[0];
				tmp += Pv[8] * ptr[4];
				tmp += Pv[7] * ptr[3];
				res += tmp * Rv[6];
			}
			{// u_coord[7]=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += Pv[5] * ptr[0];
				tmp += Pv[8] * ptr[3];
				res += tmp * Rv[7];
			}
			AC[C_IDX(I,K)][3] = res;
		}
		if (CHECK_BDR(I,K)) {// ingb=4
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I,K)];
			{// u_coord[0]=(2*I + box - hx - 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz - 1)];
				tmp += Pv[4] * ptr[8];
				tmp += Pv[3] * ptr[7];
				tmp += Pv[1] * ptr[5];
				tmp += Pv[0] * ptr[4];
				res += tmp * Rv[0];
			}
			{// u_coord[1]=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += Pv[4] * ptr[7];
				tmp += Pv[3] * ptr[6];
				tmp += Pv[5] * ptr[8];
				tmp += Pv[1] * ptr[4];
				tmp += Pv[0] * ptr[3];
				tmp += Pv[2] * ptr[5];
				res += tmp * Rv[1];
			}
			{// u_coord[2]=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += Pv[4] * ptr[6];
				tmp += Pv[5] * ptr[7];
				tmp += Pv[1] * ptr[3];
				tmp += Pv[2] * ptr[4];
				res += tmp * Rv[2];
			}
			{// u_coord[3]=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += Pv[4] * ptr[5];
				tmp += Pv[3] * ptr[4];
				tmp += Pv[1] * ptr[2];
				tmp += Pv[0] * ptr[1];
				tmp += Pv[7] * ptr[8];
				tmp += Pv[6] * ptr[7];
				res += tmp * Rv[3];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[4] * ptr[4];
				tmp += Pv[3] * ptr[3];
				tmp += Pv[5] * ptr[5];
				tmp += Pv[1] * ptr[1];
				tmp += Pv[0] * ptr[0];
				tmp += Pv[2] * ptr[2];
				tmp += Pv[7] * ptr[7];
				tmp += Pv[6] * ptr[6];
				tmp += Pv[8] * ptr[8];
				res += tmp * Rv[4];
			}
			{// u_coord[5]=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += Pv[4] * ptr[3];
				tmp += Pv[5] * ptr[4];
				tmp += Pv[1] * ptr[0];
				tmp += Pv[2] * ptr[1];
				tmp += Pv[7] * ptr[6];
				tmp += Pv[8] * ptr[7];
				res += tmp * Rv[5];
			}
			{// u_coord[6]=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += Pv[4] * ptr[2];
				tmp += Pv[3] * ptr[1];
				tmp += Pv[7] * ptr[5];
				tmp += Pv[6] * ptr[4];
				res += tmp * Rv[6];
			}
			{// u_coord[7]=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += Pv[4] * ptr[1];
				tmp += Pv[3] * ptr[0];
				tmp += Pv[5] * ptr[2];
				tmp += Pv[7] * ptr[4];
				tmp += Pv[6] * ptr[3];
				tmp += Pv[8] * ptr[5];
				res += tmp * Rv[7];
			}
			{// u_coord[8]=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += Pv[4] * ptr[0];
				tmp += Pv[5] * ptr[1];
				tmp += Pv[7] * ptr[3];
				tmp += Pv[8] * ptr[4];
				res += tmp * Rv[8];
			}
			AC[C_IDX(I,K)][4] = res;
		}
		if (CHECK_BDR(I,K + 1)) {// ingb=5
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I,K + 1)];
			{// u_coord[1]=(2*I + box - hx - 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz)];
				tmp += Pv[3] * ptr[8];
				tmp += Pv[0] * ptr[5];
				res += tmp * Rv[1];
			}
			{// u_coord[2]=(2*I + box - hx - 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*K + boz - hz + 1)];
				tmp += Pv[3] * ptr[7];
				tmp += Pv[4] * ptr[8];
				tmp += Pv[0] * ptr[4];
				tmp += Pv[1] * ptr[5];
				res += tmp * Rv[2];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[3] * ptr[5];
				tmp += Pv[0] * ptr[2];
				tmp += Pv[6] * ptr[8];
				res += tmp * Rv[4];
			}
			{// u_coord[5]=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += Pv[3] * ptr[4];
				tmp += Pv[4] * ptr[5];
				tmp += Pv[0] * ptr[1];
				tmp += Pv[1] * ptr[2];
				tmp += Pv[6] * ptr[7];
				tmp += Pv[7] * ptr[8];
				res += tmp * Rv[5];
			}
			{// u_coord[7]=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += Pv[3] * ptr[2];
				tmp += Pv[6] * ptr[5];
				res += tmp * Rv[7];
			}
			{// u_coord[8]=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += Pv[3] * ptr[1];
				tmp += Pv[4] * ptr[2];
				tmp += Pv[6] * ptr[4];
				tmp += Pv[7] * ptr[5];
				res += tmp * Rv[8];
			}
			AC[C_IDX(I,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I + 1,K - 1)];
			{// u_coord[3]=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += Pv[2] * ptr[7];
				tmp += Pv[1] * ptr[6];
				res += tmp * Rv[3];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[2] * ptr[6];
				res += tmp * Rv[4];
			}
			{// u_coord[6]=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += Pv[2] * ptr[4];
				tmp += Pv[1] * ptr[3];
				tmp += Pv[5] * ptr[7];
				tmp += Pv[4] * ptr[6];
				res += tmp * Rv[6];
			}
			{// u_coord[7]=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += Pv[2] * ptr[3];
				tmp += Pv[5] * ptr[6];
				res += tmp * Rv[7];
			}
			AC[C_IDX(I,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,K)) {// ingb=7
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I + 1,K)];
			{// u_coord[3]=(2*I + box - hx,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz - 1)];
				tmp += Pv[1] * ptr[8];
				tmp += Pv[0] * ptr[7];
				res += tmp * Rv[3];
			}
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[1] * ptr[7];
				tmp += Pv[0] * ptr[6];
				tmp += Pv[2] * ptr[8];
				res += tmp * Rv[4];
			}
			{// u_coord[5]=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += Pv[1] * ptr[6];
				tmp += Pv[2] * ptr[7];
				res += tmp * Rv[5];
			}
			{// u_coord[6]=(2*I + box - hx + 1,2*K + boz - hz - 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz - 1)];
				tmp += Pv[1] * ptr[5];
				tmp += Pv[0] * ptr[4];
				tmp += Pv[4] * ptr[8];
				tmp += Pv[3] * ptr[7];
				res += tmp * Rv[6];
			}
			{// u_coord[7]=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += Pv[1] * ptr[4];
				tmp += Pv[0] * ptr[3];
				tmp += Pv[2] * ptr[5];
				tmp += Pv[4] * ptr[7];
				tmp += Pv[3] * ptr[6];
				tmp += Pv[5] * ptr[8];
				res += tmp * Rv[7];
			}
			{// u_coord[8]=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += Pv[1] * ptr[3];
				tmp += Pv[2] * ptr[4];
				tmp += Pv[4] * ptr[6];
				tmp += Pv[5] * ptr[7];
				res += tmp * Rv[8];
			}
			AC[C_IDX(I,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			const data_t * Pv = PC[C_IDX(I + 1,K + 1)];
			{// u_coord[4]=(2*I + box - hx,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz)];
				tmp += Pv[0] * ptr[8];
				res += tmp * Rv[4];
			}
			{// u_coord[5]=(2*I + box - hx,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*K + boz - hz + 1)];
				tmp += Pv[0] * ptr[7];
				tmp += Pv[1] * ptr[8];
				res += tmp * Rv[5];
			}
			{// u_coord[7]=(2*I + box - hx + 1,2*K + boz - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz)];
				tmp += Pv[0] * ptr[5];
				tmp += Pv[3] * ptr[8];
				res += tmp * Rv[7];
			}
			{// u_coord[8]=(2*I + box - hx + 1,2*K + boz - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*K + boz - hz + 1)];
				tmp += Pv[0] * ptr[4];
				tmp += Pv[1] * ptr[5];
				tmp += Pv[3] * ptr[7];
				tmp += Pv[4] * ptr[8];
				res += tmp * Rv[8];
			}
			AC[C_IDX(I,K)][8] = res;
		}
	}
}

template<typename idx_t, typename data_t>
void RCell2d4_A2d9_PCell2d4(
	// 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, idx_t fx, idx_t fz,
    // 粗网格的数据，以及各维度的存储大小（含halo区）
    data_t * coar_mat, idx_t cx, idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [ckbeg, ckend)
    idx_t cibeg, idx_t ciend, idx_t ckbeg, idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // halo区的宽度
    const idx_t hx, const idx_t hz)
{
	for (idx_t i = 0; i < 9 * cx * cz; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }
    const data_t (*AF)[9] = (data_t (*)[9])fine_mat;
    data_t (*AC)[9] = (data_t (*)[9])coar_mat;

	#pragma omp parallel for collapse(2) schedule(static)
	for (idx_t I = cibeg; I < ciend; I++)
	for (idx_t K = ckbeg; K < ckend; K++) {
		if (CHECK_BDR(I - 1,K - 1)) {// ingb=0
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 1.0000000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 1.0000000 * ptr[1];
				tmp += 1.0000000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[0];
				tmp += 1.0000000 * ptr[1];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[2];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][2] = res;
		}
		if (CHECK_BDR(I,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 1.0000000 * ptr[3];
				tmp += 1.0000000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 1.0000000 * ptr[0];
				tmp += 1.0000000 * ptr[3];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][3] = res;
		}
		if (CHECK_BDR(I,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 1.0000000 * ptr[4];
				tmp += 1.0000000 * ptr[5];
				tmp += 1.0000000 * ptr[7];
				tmp += 1.0000000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 1.0000000 * ptr[1];
				tmp += 1.0000000 * ptr[2];
				tmp += 1.0000000 * ptr[4];
				tmp += 1.0000000 * ptr[5];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[3];
				tmp += 1.0000000 * ptr[4];
				tmp += 1.0000000 * ptr[6];
				tmp += 1.0000000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[0];
				tmp += 1.0000000 * ptr[1];
				tmp += 1.0000000 * ptr[3];
				tmp += 1.0000000 * ptr[4];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][4] = res;
		}
		if (CHECK_BDR(I,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[5];
				tmp += 1.0000000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[2];
				tmp += 1.0000000 * ptr[5];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 1.0000000 * ptr[6];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 1.0000000 * ptr[7];
				tmp += 1.0000000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[6];
				tmp += 1.0000000 * ptr[7];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 1.0000000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][8] = res;
		}
	}
}

template<typename idx_t, typename data_t>
void RCell2d4_A2d9_PCell2d16(
	// 细网格的数据，以及各维度的存储大小（含halo区）
    const data_t * fine_mat, idx_t fx, idx_t fz,
    // 粗网格的数据，以及各维度的存储大小（含halo区）
    data_t * coar_mat, idx_t cx, idx_t cz,
    // 本进程在粗网格上实际负责的范围：[cibeg, ciend) x [ckbeg, ckend)
    idx_t cibeg, idx_t ciend, idx_t ckbeg, idx_t ckend,
    // ilb, iub 分别记录本进程在i维度是否在左边界和右边界
    const bool ilb , const bool iub , 
    // klb, kub 分别记录本进程在k维度是否在下边界和上边界
    const bool klb , const bool kub ,
    // halo区的宽度
    const idx_t hx, const idx_t hz)
{
	for (idx_t i = 0; i < 9 * cx * cz; i++) {
        // 初值赋为0，为了边界条件
        coar_mat[i] = 0.0;
    }
    const data_t (*AF)[9] = (data_t (*)[9])fine_mat;
    data_t (*AC)[9] = (data_t (*)[9])coar_mat;

	#pragma omp parallel for collapse(2) schedule(static)
	for (idx_t I = cibeg; I < ciend; I++)
	for (idx_t K = ckbeg; K < ckend; K++) {
		if (CHECK_BDR(I - 1,K - 1)) {// ingb=0
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.5625000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.1875000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.0625000 * ptr[2];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.0625000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[2];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][2] = res;
		}
		if (CHECK_BDR(I,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.0625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][3] = res;
		}
		if (CHECK_BDR(I,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.0625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][4] = res;
		}
		if (CHECK_BDR(I,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.0625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.0625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz)];
				tmp += 0.0625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz)];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*K - hz + 1)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*K - hz + 1)];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,K)][8] = res;
		}
	}
}

#undef C_IDX
#undef F_IDX
#undef CHECK_BDR

#endif