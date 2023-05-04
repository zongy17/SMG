#ifndef SMG_PLANE_SOLVER_HPP
#define SMG_PLANE_SOLVER_HPP

#include "precond.hpp"
#include "../utils/common.hpp"

template<typename idx_t, typename data_t, typename calc_t>
class PlaneSolver : public Solver<idx_t, data_t, calc_t> {
public:
    const PLANE_DIRECTION plane_dir = XZ;

    // operator (often as matrix-A)
    const Operator<idx_t, calc_t, calc_t> * oper = nullptr;

    PlaneSolver(PLANE_DIRECTION plane_dir = XZ) : Solver<idx_t, data_t, calc_t>(), plane_dir(plane_dir) {
        assert(plane_dir == XZ);
    }

    virtual void SetOperator(const Operator<idx_t, calc_t, calc_t> & op) {
#ifdef COMPRESS
        assert(((const par_structMatrix<idx_t, calc_t, calc_t>&)op).compressed == false);// 暂时先不处理PlaneSolver等的压缩情况
#endif
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];

        Setup();
    }

    // 子类重写
    virtual void Setup() = 0;
};

#endif