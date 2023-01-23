#ifndef SMG_PLANE_SOLVER_HPP
#define SMG_PLANE_SOLVER_HPP

#include "precond.hpp"
#include "../utils/common.hpp"

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
class PlaneSolver : public Solver<idx_t, data_t, oper_t, res_t> {
public:
    const PLANE_DIRECTION plane_dir = XZ;

    // operator (often as matrix-A)
    const Operator<idx_t, oper_t, res_t> * oper = nullptr;

    PlaneSolver(PLANE_DIRECTION plane_dir = XZ) : Solver<idx_t, data_t, oper_t, res_t>(), plane_dir(plane_dir) {
        assert(plane_dir == XZ);
    }

    virtual void SetOperator(const Operator<idx_t, oper_t, res_t> & op) {
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