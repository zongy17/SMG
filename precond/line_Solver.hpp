#ifndef SMG_LINE_SOLVER_HPP
#define SMG_LINE_SOLVER_HPP

#include "precond.hpp"
#include "../utils/par_struct_mat.hpp"
#include "../utils/tridiag.hpp"
// #include "../utils/cyc_tridiag.hpp"

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
class LineSolver : public Solver<idx_t, data_t, oper_t, res_t> {
public:
    // when line is along periodic direction, comm is needed
    // MPI_Comm cart_comm = MPI_COMM_NULL;
    // int my_pid = MPI_PROC_NULL, prev_id = MPI_PROC_NULL, next_id = MPI_PROC_NULL;
    // direction to update vals simultaneously (default as 0)
    // 0 : inner-most (vertical), 1: middle, 2: outer-most
    const LINE_DIRECTION line_dir = VERT;

    // operator (often as matrix-A)
    const Operator<idx_t, oper_t, res_t> * oper = nullptr;
    idx_t num_solvers = 0;
    bool setup_called = false;
    TridiagSolver<idx_t, data_t, res_t> ** tri_solver = nullptr;

    LineSolver(LINE_DIRECTION line_dir, const StructCommPackage & comm_pkg) : 
        Solver<idx_t, data_t, oper_t, res_t>(), line_dir(line_dir) {
        assert(line_dir == VERT);
    }

    virtual void SetOperator(const Operator<idx_t, oper_t, res_t> & op) {
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
    void extract_vals_from_mat();

    virtual ~LineSolver();
};

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
LineSolver<idx_t, data_t, oper_t, res_t>::~LineSolver() {
    if (tri_solver != nullptr) {
        for (idx_t i = 0; i < num_solvers; i++)
            delete tri_solver[i];
        delete[] tri_solver;
    }
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineSolver<idx_t, data_t, oper_t, res_t>::Setup() 
{
    int my_pid; MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);

    if (setup_called) {
        if (my_pid == 0) printf("  LineSolver::Setup() already done ===> return \n");
        return ;
    }
    setup_called = true;

    assert(this->oper != nullptr);

    const seq_structMatrix<idx_t, oper_t, res_t> * mat = ((par_structMatrix<idx_t, oper_t, res_t>*)(this->oper))->local_matrix;
    idx_t n_size;
    switch (line_dir)
    {
    case VERT:
        assert(tri_solver == nullptr);
        num_solvers = mat->local_x * mat->local_y;
        n_size = mat->local_z;
        if (my_pid == 0) printf("  LineSolver length: %d\n", n_size);
        tri_solver = new TridiagSolver<idx_t, data_t, res_t> * [num_solvers];
        for (idx_t i = 0; i < num_solvers; i++) {
            tri_solver[i] = new TridiagSolver<idx_t, data_t, res_t>;
            tri_solver[i]->alloc(n_size);
        }
        break;
    default:
        printf("INVALID line_dir of %c to setup TridiagSolver\n", line_dir);
        MPI_Abort(MPI_COMM_WORLD, -4000);
        break;
    }

    if (sizeof(oper_t) != sizeof(data_t)) {
        if (my_pid == 0) {
            printf("  \033[1;31mWarning\033[0m: LineSolver Setup() using oper_t of %ld bytes, but data_t of %ld bytes", sizeof(oper_t), sizeof(data_t));
            printf(" ===> only allocated space: instead consider to use higher precision to Setup then copy with truncation.\n");
        }
        return ;
    }

    extract_vals_from_mat();

    switch (line_dir)
    {
    case VERT:
        for (idx_t i = 0; i < num_solvers; i++) {
            tri_solver[i]->Setup();
        }
        break;
    default:
        printf("INVALID line_dir of %c to setup TridiagSolver\n", line_dir);
        MPI_Abort(MPI_COMM_WORLD, -4000);
        break;
    }
}

template<typename idx_t, typename data_t, typename oper_t, typename res_t>
void LineSolver<idx_t,data_t, oper_t, res_t>::extract_vals_from_mat() 
{
    const seq_structMatrix<idx_t, oper_t, res_t> & mat = *(((par_structMatrix<idx_t, oper_t, res_t>*)(this->oper))->local_matrix);
#define MATIDX(d, k, i, j) (d) + (k) * mat.num_diag + (i) * mat.slice_dk_size + (j) * mat.slice_dki_size
    assert(line_dir == VERT);
    switch (line_dir)
    {
    case VERT:
        assert(tri_solver[0]->Get_n_size() == mat.local_z);
        for (idx_t j = 0; j < mat.local_y; j++) {
            idx_t real_j = j + mat.halo_y;
            for (idx_t i = 0; i < mat.local_x; i++) {
                idx_t real_i = i + mat.halo_x;
                data_t * a = tri_solver[j * mat.local_x + i]->Get_a();
                data_t * b = tri_solver[j * mat.local_x + i]->Get_b();
                data_t * c = tri_solver[j * mat.local_x + i]->Get_c();

                const idx_t kbeg = mat.halo_z, kend = kbeg + mat.local_z;
                assert(tri_solver[0]->Get_n_size() == kend - kbeg);
                if (mat.num_diag == 7) {
                    for (idx_t k = 0; k < mat.local_z; k++) {
                        // 注意位置偏移！
                        idx_t real_k = k + kbeg;
                        c[k] = mat.data[MATIDX( 4, real_k, real_i, real_j)];
                        b[k] = mat.data[MATIDX( 3, real_k, real_i, real_j)];
                        a[k] = mat.data[MATIDX( 2, real_k, real_i, real_j)];
                    }
                }
                else if (mat.num_diag == 19) {
                    for (idx_t k = 0; k < mat.local_z; k++) {
                        // 注意位置偏移！
                        idx_t real_k = k + kbeg;
                        c[k] = mat.data[MATIDX(10, real_k, real_i, real_j)];
                        b[k] = mat.data[MATIDX( 9, real_k, real_i, real_j)];
                        a[k] = mat.data[MATIDX( 8, real_k, real_i, real_j)];
                    }
                }
                else if (mat.num_diag == 27) {
                    for (idx_t k = 0; k < mat.local_z; k++) {
                        // 注意位置偏移！
                        idx_t real_k = k + kbeg;
                        c[k] = mat.data[MATIDX(14, real_k, real_i, real_j)];
                        b[k] = mat.data[MATIDX(13, real_k, real_i, real_j)];
                        a[k] = mat.data[MATIDX(12, real_k, real_i, real_j)];
                    }
                }
                else {
                    printf("INVALID num_diag of %d to extract vals from matrix\n", mat.num_diag);
                    MPI_Abort(MPI_COMM_WORLD, -4000);
                }
            }
        }
        break;
    default:
        printf("INVALID dir of %c to extract vals from matrix\n", line_dir);
        MPI_Abort(MPI_COMM_WORLD, -4000);
    }
#undef MATIDX
}

#endif
