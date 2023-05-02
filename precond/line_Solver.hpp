#ifndef SMG_LINE_SOLVER_HPP
#define SMG_LINE_SOLVER_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"
#include "../utils/tridiag.hpp"
// #include "../utils/cyc_tridiag.hpp"

template<typename idx_t, typename data_t, typename calc_t>
class LineSolver : public Solver<idx_t, data_t, calc_t> {
public:
    // when line is along periodic direction, comm is needed
    // MPI_Comm cart_comm = MPI_COMM_NULL;
    // int my_pid = MPI_PROC_NULL, prev_id = MPI_PROC_NULL, next_id = MPI_PROC_NULL;
    // direction to update vals simultaneously (default as 0)
    // 0 : inner-most (vertical), 1: middle, 2: outer-most
    const LINE_DIRECTION line_dir = VERT;

    // operator (often as matrix-A)
    const Operator<idx_t, calc_t, calc_t> * oper = nullptr;
    idx_t num_solvers = 0;
    bool setup_called = false;
    TridiagSolver<idx_t, data_t, calc_t> ** tri_solver = nullptr;

    LineSolver(LINE_DIRECTION line_dir, const StructCommPackage * comm_pkg = nullptr) : 
        Solver<idx_t, data_t, calc_t>(), line_dir(line_dir) {
        assert(line_dir == VERT);
    }

    virtual void SetOperator(const Operator<idx_t, calc_t, calc_t> & op) {
        oper = & op;

        this->input_dim[0] = op.input_dim[0];
        this->input_dim[1] = op.input_dim[1];
        this->input_dim[2] = op.input_dim[2];

        this->output_dim[0] = op.output_dim[0];
        this->output_dim[1] = op.output_dim[1];
        this->output_dim[2] = op.output_dim[2];
                    
        // 根据松弛方向进行setup
        Setup();
        // do something when needed
        post_setup();
    }

    virtual void Setup();
    virtual void post_setup() {}

    virtual ~LineSolver();
};

template<typename idx_t, typename data_t, typename calc_t>
LineSolver<idx_t, data_t, calc_t>::~LineSolver() {
    if (tri_solver != nullptr) {
        for (idx_t i = 0; i < num_solvers; i++)
            delete tri_solver[i];
        delete[] tri_solver;
    }
}

template<typename idx_t, typename data_t, typename calc_t>
void LineSolver<idx_t, data_t, calc_t>::Setup() 
{
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);

    if (setup_called) {
        if (my_pid == 0) printf("  LineSolver::Setup() already done ===> return \n");
        return ;
    }
    setup_called = true;

    assert(this->oper != nullptr);

    const seq_structMatrix<idx_t, calc_t, calc_t> * mat = ((par_structMatrix<idx_t, calc_t, calc_t>*)(this->oper))->local_matrix;
    idx_t n_size;
    assert (line_dir == VERT);
    assert(tri_solver == nullptr);

    const idx_t num_diag = mat->num_diag;
    const idx_t hx = mat->halo_x , hy = mat->halo_y , hz = mat->halo_z ;
    const idx_t lx = mat->local_x, ly = mat->local_y, lz = mat->local_z;

    num_solvers = lx * ly;
    n_size = lz;
    if (my_pid == 0) printf("  LineSolver length: %d\n", n_size);
    tri_solver = new TridiagSolver<idx_t, data_t, calc_t> * [num_solvers];

    if (num_diag != 7 && num_diag != 19 && num_diag != 27 && num_diag != 9) {
        printf("INVALID num_diag of %d to extract vals from matrix\n", num_diag);
        MPI_Abort(MPI_COMM_WORLD, -4000);
    }

    const idx_t diag_id = num_diag >> 1;
    #pragma omp parallel
    {
        calc_t  * a = new calc_t [n_size],
                * b = new calc_t [n_size],             
                * c = new calc_t [n_size]; 
        
        #pragma omp for collapse(2) schedule(static)
        for (idx_t j = 0; j < ly; j++)
        for (idx_t i = 0; i < lx; i++) {
            idx_t real_j = j + hy;
            idx_t real_i = i + hx;
            const idx_t sid = j * lx + i;
            tri_solver[sid] = new TridiagSolver<idx_t, data_t, calc_t>;
            for (idx_t k = 0; k < lz; k++) {
                // 注意位置偏移！
                idx_t real_k = k + hz;
                const calc_t * src_ptr = mat->data + real_j * mat->slice_dki_size
                    + real_i * mat->slice_dk_size + real_k * mat->num_diag + diag_id;
                c[k] = src_ptr[ 1];
                b[k] = src_ptr[ 0];
                a[k] = src_ptr[-1];
            }
            tri_solver[sid]->Setup(n_size, a, b, c);// ????
        }
    }
}

#endif
