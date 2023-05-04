#ifndef SMG_RAP_3D27_HPP
#define SMG_RAP_3D27_HPP

#include "../utils/common.hpp"

/* assume 粗网格也是3d27，所以遍历27个非零元
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

    
    这样排布的好处
    外维j，中间i，内维k，三维坐标(i,j,k) 的一维寻址为 i*3 + j*9 + k + 13(数组整体偏移)，其中13是由27减1再除以2得到的
    0 <=> (i,j,k)=(-1,-1,-1) 取相反数 ( 1, 1, 1) <=> 26
    1 <=> (i,j,k)=(-1,-1, 0) 取相反数 ( 1, 1, 0) <=> 25
    2 <=> (i,j,k)=(-1,-1, 1) 取相反数 ( 1, 1,-1) <=> 24
    3 <=> (i,j,k)=( 0,-1,-1) 取相反数 ( 0, 1, 1) <=> 23
    4 <=> (i,j,k)=( 0,-1, 0) 取相反数 ( 0, 1, 0) <=> 22
    10<=> (i,j,k)=(-1, 0, 0) 取相反数 ( 1, 0, 0) <=> 16
    12<=> (i,j,k)=( 0, 0,-1) 取相反数 ( 0, 0, 1) <=> 14
    这也就意味着当知道邻居相对于我的偏移（即位于0~26的哪个位置）时，可以直接取一组相反数，就得到我在邻居行的对应列的元素
 */

/*
  README:
  输入前数组fine_mat必须先做update_halo()，该函数忽略通信相关，只视作单进程内粗化。
  fx, fy, fz为fine_mat的总大小（包含halo区宽度），cx, cy, cz为coar_mat的总大小（包含halo区宽度），
  需要全局大小是因为做数组偏移寻址。
  [cibeg, ciend), [ckbeg, ckend) 是本进程粗网格的起止范围（对应于本函数中需要进行处理的粗矩阵coar_mat
  的点范围）
  base_x, base_z是本进程粗网格的第一个（左下角）点（下图中Ⓒ），对应于本进程细网格的第一个（左下角）点
  在总的细网格（包含halo区宽度）中的偏移，注意区别于外层数据结构COAR_TO_FINE_INFO中的fine_base_idx[3]
  如下图示例中：
    cibeg  = 1 (= AC.halo_x), ciend = 7 (= cibeg + AC.local_x), 
    base_x = 1 (= fine_base_idx[0] + AF.halo_x = 0 + 1), 
     
    |< - - - - - - - - - - - - - - - cx  - - - - - - - - - - - - - - - - >|
       |<-------------------------  fx  ------------------------------->|
    C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C      肆
     --F----F----F----F----F----F----F----F----F----F----F----F----F----F--  7 
       |  ==|====|====|====|====|====|====|====|====|====|====|====|==  |    
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  6
    C  | || | C  |    | C  |    | C  |    | C  |    | C  |    | C  | || | C      叁
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  5
       | || |    |    |    |    |    |    |    |    |    |    |    | || |
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  4
    C  | || | C  |    | C  |    | C  |    | C  |    | C  |    | C  | || | C      贰
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  3
       | || |    |    |    |    |    |    |    |    |    |    |    | || |
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  2
    C  | || | C  |    | C  |    | C  |    | C  |    | C  |    | C  | || | C      壹
     --F-||-F----F----F----F----F----F----F----F----F----F----F----F-||-F--  1
       |  ==|====|====|====|====|====|====|====|====|====|====|====|==  |
     --F----F----F----F----F----F----F----F----F----F----F----F----F----F--  0 
    C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C  |    | C      零
       0    1    2    3    4    5    6    7    8    9    10   11   12   13
    零        壹        贰         叁        肆        伍        陆        柒
              ^                                                 ^
              |                                                 |
            cibeg                                            ciend-1

    粗网格点用(I,J,K)表示，细网格点用(i,j,k)表示
    在不计halo区时，粗细映射关系为，I的最邻近左右为2*I, 2*I+1，J的最邻近前后为2*J, 2*J+1，K的最邻近上下为2*K, 2*K+1
    在粗细网格均有宽度为bx, by, bz的halo区时，需要加上对应的偏移
    I的最邻近左右为2*(I-bx)+bx, 2*(I-bx)+bx+1，即2*I-bx, 2*I-bx+1，同理，
    J的最邻近前后为2*(J-by)+by, 2*(J-by)+by+1，即2*J-by, 2*J-by+1
    K的最邻近上下为2*(K-bz)+bz, 2*(K-bz)+bz+1，即2*K-bz, 2*K-bz+1
 */

#define C_IDX(i, j, k)   (k) + (i) * cz + (j) * czcx
#define F_IDX(i, j, k)   (k) + (i) * fz + (j) * fzfx
// 当本进程不在某维度的边界时，可以直接算（前提要求细网格的halo区已填充，但R采用4点时似乎也不用？）
#define CHECK_BDR(CI, CJ, CK) \
    (ilb==false || (CI) >= cibeg) && (iub==false || (CI) < ciend) && \
    (jlb==false || (CJ) >= cjbeg) && (jub==false || (CJ) < cjend) && \
    (klb==false || (CK) >= ckbeg) && (kub==false || (CK) < ckend)

template<typename idx_t, typename data_t>
void RCell2d4_A3d27_PCell2d16(
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

    const data_t (*AF)[27] = (data_t (*)[27])fine_mat;
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
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[0];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[13];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[1];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[1];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[1];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[1];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[14];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.5625000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[11];
				tmp += 0.1875000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[2];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[2];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][2] = res;
		}
		if (CHECK_BDR(I,J - 1,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.5625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.0625000 * ptr[15];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.0625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][3] = res;
		}
		if (CHECK_BDR(I,J - 1,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.5625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.0625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][4] = res;
		}
		if (CHECK_BDR(I,J - 1,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.0625000 * ptr[11];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.5625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.0625000 * ptr[17];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.0625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[15];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.5625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[6];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[13];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.5625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[7];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[17];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[14];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.5625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[8];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][8] = res;
		}
		if (CHECK_BDR(I - 1,J,K - 1)) {// ingb=9
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.5625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[21];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][9] = res;
		}
		if (CHECK_BDR(I - 1,J,K)) {// ingb=10
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.0625000 * ptr[22];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[19];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][10] = res;
		}
		if (CHECK_BDR(I - 1,J,K + 1)) {// ingb=11
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.5625000 * ptr[11];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.5625000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.0625000 * ptr[23];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.5625000 * ptr[11];
				tmp += 0.1875000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.0625000 * ptr[20];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][11] = res;
		}
		if (CHECK_BDR(I,J,K - 1)) {// ingb=12
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.5625000 * ptr[21];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[0];
				tmp += 0.1875000 * ptr[18];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.5625000 * ptr[18];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.5625000 * ptr[21];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.1875000 * ptr[0];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.0625000 * ptr[18];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[0];
				tmp += 0.5625000 * ptr[9];
				tmp += 0.1875000 * ptr[18];
				tmp += 0.5625000 * ptr[3];
				tmp += 0.5625000 * ptr[12];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.0625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][12] = res;
		}
		if (CHECK_BDR(I,J,K)) {// ingb=13
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.5625000 * ptr[13];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[22];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[1];
				tmp += 0.1875000 * ptr[19];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.5625000 * ptr[19];
				tmp += 0.5625000 * ptr[13];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.5625000 * ptr[22];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.5625000 * ptr[13];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.1875000 * ptr[1];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.0625000 * ptr[19];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[1];
				tmp += 0.5625000 * ptr[10];
				tmp += 0.1875000 * ptr[19];
				tmp += 0.5625000 * ptr[4];
				tmp += 0.5625000 * ptr[13];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.0625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][13] = res;
		}
		if (CHECK_BDR(I,J,K + 1)) {// ingb=14
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.5625000 * ptr[14];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.5625000 * ptr[23];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.0625000 * ptr[2];
				tmp += 0.1875000 * ptr[20];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.5625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.5625000 * ptr[11];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.5625000 * ptr[20];
				tmp += 0.5625000 * ptr[14];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.5625000 * ptr[23];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.0625000 * ptr[8];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.5625000 * ptr[14];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.1875000 * ptr[2];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.0625000 * ptr[20];
				tmp += 0.5625000 * ptr[8];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.5625000 * ptr[2];
				tmp += 0.5625000 * ptr[11];
				tmp += 0.1875000 * ptr[20];
				tmp += 0.5625000 * ptr[5];
				tmp += 0.5625000 * ptr[14];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.0625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][14] = res;
		}
		if (CHECK_BDR(I + 1,J,K - 1)) {// ingb=15
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.0625000 * ptr[6];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[3];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.5625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[6];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.0625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[3];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.0625000 * ptr[21];
				tmp += 0.5625000 * ptr[6];
				tmp += 0.5625000 * ptr[15];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][15] = res;
		}
		if (CHECK_BDR(I + 1,J,K)) {// ingb=16
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.0625000 * ptr[7];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.0625000 * ptr[4];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.5625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[7];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.0625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[4];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.0625000 * ptr[22];
				tmp += 0.5625000 * ptr[7];
				tmp += 0.5625000 * ptr[16];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][16] = res;
		}
		if (CHECK_BDR(I + 1,J,K + 1)) {// ingb=17
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.0625000 * ptr[8];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.0625000 * ptr[5];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.5625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[8];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.0625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[5];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.0625000 * ptr[23];
				tmp += 0.5625000 * ptr[8];
				tmp += 0.5625000 * ptr[17];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][17] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K - 1)) {// ingb=18
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[21];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[18];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][18] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K)) {// ingb=19
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[22];
				tmp += 0.1875000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[13];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[19];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][19] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K + 1)) {// ingb=20
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[23];
				tmp += 0.1875000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[14];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.5625000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[11];
				tmp += 0.1875000 * ptr[20];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][20] = res;
		}
		if (CHECK_BDR(I,J + 1,K - 1)) {// ingb=21
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.0625000 * ptr[18];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[18];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.0625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[21];
				tmp += 0.0625000 * ptr[9];
				tmp += 0.1875000 * ptr[18];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.5625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[9];
				tmp += 0.5625000 * ptr[18];
				tmp += 0.1875000 * ptr[12];
				tmp += 0.5625000 * ptr[21];
				tmp += 0.0625000 * ptr[15];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][21] = res;
		}
		if (CHECK_BDR(I,J + 1,K)) {// ingb=22
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.0625000 * ptr[19];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[19];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.0625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.5625000 * ptr[22];
				tmp += 0.0625000 * ptr[10];
				tmp += 0.1875000 * ptr[19];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.5625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[10];
				tmp += 0.5625000 * ptr[19];
				tmp += 0.1875000 * ptr[13];
				tmp += 0.5625000 * ptr[22];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][22] = res;
		}
		if (CHECK_BDR(I,J + 1,K + 1)) {// ingb=23
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.0625000 * ptr[20];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.1875000 * ptr[20];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.0625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.5625000 * ptr[23];
				tmp += 0.0625000 * ptr[11];
				tmp += 0.1875000 * ptr[20];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.5625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.1875000 * ptr[11];
				tmp += 0.5625000 * ptr[20];
				tmp += 0.1875000 * ptr[14];
				tmp += 0.5625000 * ptr[23];
				tmp += 0.0625000 * ptr[17];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][23] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K - 1)) {// ingb=24
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[21];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[15];
				tmp += 0.1875000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[12];
				tmp += 0.1875000 * ptr[21];
				tmp += 0.1875000 * ptr[15];
				tmp += 0.5625000 * ptr[24];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][24] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K)) {// ingb=25
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[22];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[16];
				tmp += 0.1875000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[13];
				tmp += 0.1875000 * ptr[22];
				tmp += 0.1875000 * ptr[16];
				tmp += 0.5625000 * ptr[25];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][25] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K + 1)) {// ingb=26
			data_t res = 0.0;
			{// u_coord=(2*I - hx,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy,K)];
				tmp += 0.0625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy,K)];
				tmp += 0.0625000 * ptr[23];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[17];
				tmp += 0.1875000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I - hx + 1,2*J - hy + 1,K)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I - hx + 1,2*J - hy + 1,K)];
				tmp += 0.0625000 * ptr[14];
				tmp += 0.1875000 * ptr[23];
				tmp += 0.1875000 * ptr[17];
				tmp += 0.5625000 * ptr[26];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][26] = res;
		}
	}

}

template<typename idx_t, typename data_t>
void RVtx2d9_A3d27_PVtx2d9(
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

    const data_t (*AF)[27] = (data_t (*)[27])fine_mat;
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
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[9];
				tmp += 1.0000000 * ptr[0];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.5000000 * ptr[0];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[0];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[0];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][0] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K)) {// ingb=1
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[10];
				tmp += 1.0000000 * ptr[1];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[1];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[1];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][1] = res;
		}
		if (CHECK_BDR(I - 1,J - 1,K + 1)) {// ingb=2
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[11];
				tmp += 1.0000000 * ptr[2];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[11];
				tmp += 0.5000000 * ptr[2];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[2];
				res += tmp * 0.2500000;
			}
			AC[C_IDX(I,J,K)][2] = res;
		}
		if (CHECK_BDR(I,J - 1,K - 1)) {// ingb=3
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[15];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 1.0000000 * ptr[3];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[15];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[3];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[3];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][3] = res;
		}
		if (CHECK_BDR(I,J - 1,K)) {// ingb=4
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[16];
				tmp += 1.0000000 * ptr[7];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[4];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 1.0000000 * ptr[4];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[16];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[4];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[4];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][4] = res;
		}
		if (CHECK_BDR(I,J - 1,K + 1)) {// ingb=5
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[17];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[14];
				tmp += 1.0000000 * ptr[5];
				tmp += 0.2500000 * ptr[11];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[17];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[5];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[5];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][5] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K - 1)) {// ingb=6
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[15];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[6];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[15];
				tmp += 1.0000000 * ptr[6];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.5000000 * ptr[6];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][6] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K)) {// ingb=7
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[16];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[7];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[16];
				tmp += 1.0000000 * ptr[7];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.5000000 * ptr[7];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][7] = res;
		}
		if (CHECK_BDR(I + 1,J - 1,K + 1)) {// ingb=8
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[17];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[8];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[17];
				tmp += 1.0000000 * ptr[8];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.5000000 * ptr[8];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][8] = res;
		}
		if (CHECK_BDR(I - 1,J,K - 1)) {// ingb=9
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.2500000 * ptr[12];
				tmp += 1.0000000 * ptr[18];
				tmp += 0.5000000 * ptr[9];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[21];
				tmp += 1.0000000 * ptr[9];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[12];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.5000000 * ptr[9];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[9];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[9];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][9] = res;
		}
		if (CHECK_BDR(I - 1,J,K)) {// ingb=10
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.2500000 * ptr[13];
				tmp += 1.0000000 * ptr[19];
				tmp += 0.5000000 * ptr[10];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.2500000 * ptr[22];
				tmp += 1.0000000 * ptr[10];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[19];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[13];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.5000000 * ptr[10];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[19];
				tmp += 0.2500000 * ptr[10];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[10];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][10] = res;
		}
		if (CHECK_BDR(I - 1,J,K + 1)) {// ingb=11
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.2500000 * ptr[14];
				tmp += 1.0000000 * ptr[20];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[23];
				tmp += 1.0000000 * ptr[11];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.5000000 * ptr[20];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[14];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.5000000 * ptr[11];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[20];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[11];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][11] = res;
		}
		if (CHECK_BDR(I,J,K - 1)) {// ingb=12
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[24];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[15];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.5000000 * ptr[24];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[21];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[21];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[24];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[12];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.2500000 * ptr[0];
				tmp += 0.2500000 * ptr[18];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.2500000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[3];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[18];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[9];
				tmp += 0.5000000 * ptr[0];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[21];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[0];
				tmp += 0.5000000 * ptr[9];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[12];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][12] = res;
		}
		if (CHECK_BDR(I,J,K)) {// ingb=13
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[25];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[16];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[25];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.2500000 * ptr[22];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[7];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[22];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.5000000 * ptr[19];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[25];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[13];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.2500000 * ptr[1];
				tmp += 0.2500000 * ptr[19];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.2500000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[4];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[19];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[10];
				tmp += 0.5000000 * ptr[1];
				tmp += 0.5000000 * ptr[19];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.2500000 * ptr[22];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[1];
				tmp += 0.5000000 * ptr[10];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[13];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][13] = res;
		}
		if (CHECK_BDR(I,J,K + 1)) {// ingb=14
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[26];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.2500000 * ptr[14];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[17];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.5000000 * ptr[26];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[23];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[14];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[23];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.5000000 * ptr[20];
				tmp += 0.2500000 * ptr[11];
				tmp += 0.5000000 * ptr[26];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[14];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.2500000 * ptr[2];
				tmp += 0.2500000 * ptr[20];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.2500000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[5];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.2500000 * ptr[11];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 1.0000000 * ptr[20];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.2500000 * ptr[14];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 1.0000000 * ptr[11];
				tmp += 0.5000000 * ptr[2];
				tmp += 0.5000000 * ptr[20];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[23];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 1.0000000 * ptr[2];
				tmp += 0.5000000 * ptr[11];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[14];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][14] = res;
		}
		if (CHECK_BDR(I + 1,J,K - 1)) {// ingb=15
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[24];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[15];
				tmp += 0.2500000 * ptr[6];
				tmp += 0.2500000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.2500000 * ptr[15];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.2500000 * ptr[12];
				tmp += 1.0000000 * ptr[24];
				tmp += 0.5000000 * ptr[15];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 0.2500000 * ptr[3];
				tmp += 0.2500000 * ptr[21];
				tmp += 1.0000000 * ptr[15];
				tmp += 0.5000000 * ptr[6];
				tmp += 0.5000000 * ptr[24];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[3];
				tmp += 0.2500000 * ptr[12];
				tmp += 1.0000000 * ptr[6];
				tmp += 0.5000000 * ptr[15];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][15] = res;
		}
		if (CHECK_BDR(I + 1,J,K)) {// ingb=16
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[25];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[16];
				tmp += 0.2500000 * ptr[7];
				tmp += 0.2500000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.2500000 * ptr[16];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.2500000 * ptr[13];
				tmp += 1.0000000 * ptr[25];
				tmp += 0.5000000 * ptr[16];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 0.2500000 * ptr[4];
				tmp += 0.2500000 * ptr[22];
				tmp += 1.0000000 * ptr[16];
				tmp += 0.5000000 * ptr[7];
				tmp += 0.5000000 * ptr[25];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[4];
				tmp += 0.2500000 * ptr[13];
				tmp += 1.0000000 * ptr[7];
				tmp += 0.5000000 * ptr[16];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][16] = res;
		}
		if (CHECK_BDR(I + 1,J,K + 1)) {// ingb=17
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[26];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[17];
				tmp += 0.2500000 * ptr[8];
				tmp += 0.2500000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.2500000 * ptr[17];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy - 1,K + boz)];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.2500000 * ptr[14];
				tmp += 1.0000000 * ptr[26];
				tmp += 0.5000000 * ptr[17];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[14];
				tmp += 0.2500000 * ptr[5];
				tmp += 0.2500000 * ptr[23];
				tmp += 1.0000000 * ptr[17];
				tmp += 0.5000000 * ptr[8];
				tmp += 0.5000000 * ptr[26];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[5];
				tmp += 0.2500000 * ptr[14];
				tmp += 1.0000000 * ptr[8];
				tmp += 0.5000000 * ptr[17];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][17] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K - 1)) {// ingb=18
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[21];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.5000000 * ptr[9];
				tmp += 1.0000000 * ptr[18];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[18];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[18];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][18] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K)) {// ingb=19
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[22];
				tmp += 0.5000000 * ptr[19];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.5000000 * ptr[10];
				tmp += 1.0000000 * ptr[19];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[19];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[19];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][19] = res;
		}
		if (CHECK_BDR(I - 1,J + 1,K + 1)) {// ingb=20
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[23];
				tmp += 0.5000000 * ptr[20];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.5000000 * ptr[11];
				tmp += 1.0000000 * ptr[20];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[20];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[11];
				tmp += 0.5000000 * ptr[20];
				res += tmp * 0.1250000;
			}
			AC[C_IDX(I,J,K)][20] = res;
		}
		if (CHECK_BDR(I,J + 1,K - 1)) {// ingb=21
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[24];
				tmp += 0.2500000 * ptr[21];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[15];
				tmp += 1.0000000 * ptr[24];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[21];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.2500000 * ptr[18];
				tmp += 0.2500000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[12];
				tmp += 1.0000000 * ptr[21];
				tmp += 0.2500000 * ptr[9];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[15];
				tmp += 0.5000000 * ptr[24];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[18];
				tmp += 0.2500000 * ptr[21];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[9];
				tmp += 1.0000000 * ptr[18];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[21];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][21] = res;
		}
		if (CHECK_BDR(I,J + 1,K)) {// ingb=22
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[25];
				tmp += 0.2500000 * ptr[22];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[16];
				tmp += 1.0000000 * ptr[25];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[22];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.2500000 * ptr[19];
				tmp += 0.2500000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[13];
				tmp += 1.0000000 * ptr[22];
				tmp += 0.2500000 * ptr[10];
				tmp += 0.5000000 * ptr[19];
				tmp += 0.2500000 * ptr[16];
				tmp += 0.5000000 * ptr[25];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[19];
				tmp += 0.2500000 * ptr[22];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[10];
				tmp += 1.0000000 * ptr[19];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[22];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][22] = res;
		}
		if (CHECK_BDR(I,J + 1,K + 1)) {// ingb=23
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[26];
				tmp += 0.2500000 * ptr[23];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx - 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[17];
				tmp += 1.0000000 * ptr[26];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[23];
				res += tmp * 0.0625000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.2500000 * ptr[20];
				tmp += 0.2500000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[14];
				tmp += 1.0000000 * ptr[23];
				tmp += 0.2500000 * ptr[11];
				tmp += 0.5000000 * ptr[20];
				tmp += 0.2500000 * ptr[17];
				tmp += 0.5000000 * ptr[26];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.5000000 * ptr[20];
				tmp += 0.2500000 * ptr[23];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.5000000 * ptr[11];
				tmp += 1.0000000 * ptr[20];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[23];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][23] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K - 1)) {// ingb=24
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[24];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[15];
				tmp += 0.5000000 * ptr[24];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[21];
				tmp += 0.5000000 * ptr[24];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[12];
				tmp += 0.5000000 * ptr[21];
				tmp += 0.5000000 * ptr[15];
				tmp += 1.0000000 * ptr[24];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][24] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K)) {// ingb=25
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[25];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[16];
				tmp += 0.5000000 * ptr[25];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[22];
				tmp += 0.5000000 * ptr[25];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[13];
				tmp += 0.5000000 * ptr[22];
				tmp += 0.5000000 * ptr[16];
				tmp += 1.0000000 * ptr[25];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][25] = res;
		}
		if (CHECK_BDR(I + 1,J + 1,K + 1)) {// ingb=26
			data_t res = 0.0;
			{// u_coord=(2*I + box - hx,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[26];
				res += tmp * 0.2500000;
			}
			{// u_coord=(2*I + box - hx,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[17];
				tmp += 0.5000000 * ptr[26];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy,K + boz)];
				tmp += 0.2500000 * ptr[23];
				tmp += 0.5000000 * ptr[26];
				res += tmp * 0.1250000;
			}
			{// u_coord=(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)
				data_t tmp = 0.0;
				const data_t * ptr = AF[F_IDX(2*I + box - hx + 1,2*J + boy - hy + 1,K + boz)];
				tmp += 0.2500000 * ptr[14];
				tmp += 0.5000000 * ptr[23];
				tmp += 0.5000000 * ptr[17];
				tmp += 1.0000000 * ptr[26];
				res += tmp * 0.0625000;
			}
			AC[C_IDX(I,J,K)][26] = res;
		}
	}
}

#undef C_IDX
#undef F_IDX
#undef CHECK_BDR

#endif