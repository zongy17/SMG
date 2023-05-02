#ifndef SMG_GMG_TYPES_HPP
#define SMG_GMG_TYPES_HPP

#include "../utils/par_struct_mat.hpp"
#include "../utils/operator.hpp"
// possible smoother
// #include "../precond/point_Jacobi.hpp"
#include "../precond/point_GS.hpp"
#include "../precond/line_GS.hpp"
#include "../precond/block_ILU.hpp"


typedef enum {FULL_XYZ, SEMI_XY, DECP_XZ} COARSEN_TYPE;
typedef enum {DISCRETIZED, GALERKIN} COARSE_OP_TYPE;
typedef enum {PJ, PGS, LGS, PILU, BILU3d7, BILU3d15, BILU3d19, BILU3d27, GaussElim} RELAX_TYPE;
typedef enum {Vtx_2d9_OpDep, Vtx_2d9, Vtx_2d5, Vtx_inject, Cell_2d4, Cell_2d16, Cell_3d8, Cell_3d64} RSTR_PRLG_TYPE;

template<typename idx_t>
class COAR_TO_FINE_INFO {
public:
    COARSEN_TYPE type;
    // 细网格上的起始索引（以local索引计）：在cell-center的形式时
    // 表示本进程范围内的第0个粗网格cell由本进程范围内的第base_和第base_+1个cell粗化而来
    idx_t fine_base_idx[3];

    idx_t stride[3];// 三个方向上的粗化步长
    COAR_TO_FINE_INFO() {  }
    COAR_TO_FINE_INFO(idx_t b0, idx_t b1, idx_t b2, idx_t s0, idx_t s1, idx_t s2) :
        fine_base_idx{b0, b1, b2}, stride{s0, s1, s2} {  }
};



#endif