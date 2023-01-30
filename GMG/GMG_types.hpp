#ifndef SMG_GMG_TYPES_HPP
#define SMG_GMG_TYPES_HPP

#include "../utils/par_struct_mat.hpp"
#include "../utils/operator.hpp"
// possible smoother
// #include "../precond/point_Jacobi.hpp"
#include "../precond/point_GS.hpp"
#include "../precond/line_GS.hpp"
#include "../precond/block_ILU.hpp"


typedef enum {STANDARD, SEMI_XY} COARSEN_TYPE;
typedef enum {DISCRETIZED, GALERKIN} COARSE_OP_TYPE;
typedef enum {PJ, PGS, LGS, PILU, BILU3d7, BILU3d15, BILU3d19, BILU3d27, GaussElim} RELAX_TYPE;
typedef enum {Rst_8cell, Rst_64cell, Rst_4cell} RESTRICT_TYPE;
typedef enum {Plg_linear_8cell, Plg_linear_64cell, Plg_linear_4cell} PROLONG_TYPE;

template<typename idx_t>
class COAR_TO_FINE_INFO {
public:
    // 细网格上的起始索引（以local索引计）：在cell-center的形式时
    // 表示本进程范围内的第0个粗网格cell由本进程范围内的第base_和第base_+1个cell粗化而来
    idx_t fine_base_idx[3];

    idx_t stride[3];// 三个方向上的粗化步长
    COAR_TO_FINE_INFO() {  }
    COAR_TO_FINE_INFO(idx_t b0, idx_t b1, idx_t b2, idx_t s0, idx_t s1, idx_t s2) :
        fine_base_idx{b0, b1, b2}, stride{s0, s1, s2} {  }
};



#endif