#ifndef SMG_SOLVER_LS_HPP
#define SMG_SOLVER_LS_HPP

// solver
#include "iter_solver/CGSolver.hpp"
#include "iter_solver/GCRSolver.hpp"
#include "iter_solver/GMRESSolver.hpp"

// precond
// #include "precond/point_Jacobi.hpp"
#include "precond/point_GS.hpp"
#include "precond/line_GS.hpp"
#include "precond/plane_ILU.hpp"
#include "precond/dense_LU.hpp"
#include "precond/block_ILU.hpp"


#include "GMG/GMG.hpp"

#endif