#include "utils/par_struct_mat.hpp"
#include "Solver_ls.hpp"
#include "adapator/Adaptor_64_for_32.hpp"

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int num_procs, my_pid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    int cnt = 1;
    std::string case_name = std::string(argv[cnt++]);
    // 三维划分
    int case_idx   = atoi(argv[cnt++]);
    int case_idy   = atoi(argv[cnt++]);
    int case_idz   = atoi(argv[cnt++]);
    int num_proc_x = atoi(argv[cnt++]);
    int num_proc_y = atoi(argv[cnt++]);
    int num_proc_z = atoi(argv[cnt++]);
    

    std::string its_name = std::string(argv[cnt++]);
    int restart    = atoi(argv[cnt++]);

    if (my_pid == 0) printf("\033[1;35mNum Proc along X: %d, along Y: %d, along Z: %d\033[0m \n", num_proc_x, num_proc_y, num_proc_z);
    if (my_pid == 0) printf("Max threads: %d\n", omp_get_max_threads());

    {   IDX_TYPE num_diag = -1;
        if (strcmp(case_name.c_str(), "LASER" ) == 0) {
            assert(sizeof(KSP_TYPE) == 8);
            num_diag = 7;
        }
        else if (strcmp(case_name.c_str(), "GRAPES") == 0) {
            assert(sizeof(KSP_TYPE) == 4);
            assert(num_proc_z == 1);
            num_diag = 19;
        } else if (strstr(case_name.c_str(), "DEMO")) {
            if      (strstr(case_name.c_str(), "07"))
                num_diag = 7;
            else if (strstr(case_name.c_str(), "19"))
                num_diag = 19;
            else if (strstr(case_name.c_str(), "27"))
                num_diag = 27;
            else MPI_Abort(MPI_COMM_WORLD, -71927);
            if (my_pid == 0) printf("DEMO program of %d\n", num_diag);
        }
        assert(num_diag != -1);
        par_structVector<IDX_TYPE, KSP_TYPE> * x = nullptr, * b = nullptr, * y = nullptr;
        par_structMatrix<IDX_TYPE, KSP_TYPE, KSP_TYPE> * A = nullptr;
        std::string pathname;
        IterativeSolver<IDX_TYPE, KSP_TYPE, PC_TYPE> * solver = nullptr;
        Solver<IDX_TYPE, PC_TYPE, KSP_TYPE> * precond = nullptr;
        std::string data_path = "/storage/hpcauser/zongyi/HUAWEI/SMG/data";

        x = new par_structVector<IDX_TYPE, KSP_TYPE          >
                (MPI_COMM_WORLD,           case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z, num_diag!=7);
        A = new par_structMatrix<IDX_TYPE, KSP_TYPE, KSP_TYPE>
                (MPI_COMM_WORLD, num_diag, case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z);
        y = new par_structVector<IDX_TYPE, KSP_TYPE>(*x);
        b = new par_structVector<IDX_TYPE, KSP_TYPE>(*x);

        pathname = data_path + "/" + case_name + 
            "/" + std::to_string(case_idx) + "x" + std::to_string(case_idy) + "x" + std::to_string(case_idz);
        if (my_pid == 0) printf("%s\n", pathname.c_str());

        x->set_val(0.0, true);// 迪利克雷边界条件
        b->set_val(0.0, true);
        y->set_val(0.0, true);
        A->set_val(0.0, true);

        if (strstr(case_name.c_str(), "DEMO")) {
            if (A->num_diag == 7) {
                // srand(time(0));
                // rand
                A->set_diag_val(0, -1.0);
                A->set_diag_val(1, -1.0);
                A->set_diag_val(2, -1.0);
                A->set_diag_val(3,  6.0);
                A->set_diag_val(4, -1.0);
                A->set_diag_val(5, -1.0);
                A->set_diag_val(6, -1.0);
                // A->set_diag_val(0, -1.0);
                // A->set_diag_val(1, -2.0);
                // A->set_diag_val(2, -4.0);
                // A->set_diag_val(3, 14.0);
                // A->set_diag_val(4, -4.0);
                // A->set_diag_val(5, -2.0);
                // A->set_diag_val(6, -1.0);
            }
            else if (A->num_diag == 19) {
                //                             A->set_diag_val(0, -1.0);
                // A->set_diag_val(1, -1.0);   A->set_diag_val(2, -1.0); A->set_diag_val(3, -1.0);
                //                             A->set_diag_val(4, -1.0);
                
                // A->set_diag_val(5, -1.0); A->set_diag_val(6, -1.0); A->set_diag_val(7, -1.0);
                // A->set_diag_val(8, -1.0); A->set_diag_val(9, 18.0); A->set_diag_val(10,-1.0);
                // A->set_diag_val(11,-1.0); A->set_diag_val(12,-1.0); A->set_diag_val(13,-1.0);
                
                //                             A->set_diag_val(14,-1.0);
                // A->set_diag_val(15,-1.0);   A->set_diag_val(16,-1.0); A->set_diag_val(17,-1.0);
                //                             A->set_diag_val(18,-1.0);

                                            A->set_diag_val(0, -0.5);
                A->set_diag_val(1, -0.25);  A->set_diag_val(2, -1.0); A->set_diag_val(3, -0.5);
                                            A->set_diag_val(4, -4.0);
                
                A->set_diag_val(5, -1.0); A->set_diag_val(6, -2.0); A->set_diag_val(7, -2.0);
                A->set_diag_val(8,-10.0); A->set_diag_val(9, 36.0); A->set_diag_val(10,-10.0);
                A->set_diag_val(11,-4.0); A->set_diag_val(12,-0.5); A->set_diag_val(13,-0.125);
                
                                            A->set_diag_val(0, -0.5);
                A->set_diag_val(1, -0.25);  A->set_diag_val(2, -2.0); A->set_diag_val(3, -0.125);
                                            A->set_diag_val(4, -1.0);
            }
            else if (A->num_diag == 27) {
                A->set_diag_val(0, -1.0); A->set_diag_val(1, -1.0); A->set_diag_val(2, -1.0);
                A->set_diag_val(3, -1.0); A->set_diag_val(4, -1.0); A->set_diag_val(5, -1.0);
                A->set_diag_val(6, -1.0); A->set_diag_val(7, -1.0); A->set_diag_val(8, -1.0);

                A->set_diag_val(9, -1.0); A->set_diag_val(10,-1.0); A->set_diag_val(11,-1.0);
                A->set_diag_val(12,-1.0); A->set_diag_val(13,30.0); A->set_diag_val(14,-1.0);
                A->set_diag_val(15,-1.0); A->set_diag_val(16,-1.0); A->set_diag_val(17,-1.0);

                A->set_diag_val(18,-1.0); A->set_diag_val(19,-1.0); A->set_diag_val(20,-1.0);
                A->set_diag_val(21,-1.0); A->set_diag_val(22,-1.0); A->set_diag_val(23,-1.0);
                A->set_diag_val(24,-1.0); A->set_diag_val(25,-1.0); A->set_diag_val(26,-1.0);
            }
            A->set_boundary();
            A->update_halo();
            b->set_val(1.0, false);// 右端项为全1向量
            // {
            //     const int   jbeg = b->local_vector->halo_y, jend = b->local_vector->local_y,
            //                 ibeg = b->local_vector->halo_x, iend = b->local_vector->local_x,
            //                 kbeg = b->local_vector->halo_z, kend = b->local_vector->local_z;
            //     #pragma omp parallel for collapse(3) schedule(static)
            //     for (int j = jbeg; j < jend; j++)
            //     for (int i = ibeg; i < iend; i++)
            //     for (int k = kbeg; k < kend; k++) {
            //         int linear = ((j - jbeg) * b->local_vector->local_x + i - ibeg)
            //             * b->local_vector->local_z + k - kbeg;
            //         b->local_vector->data[j * b->local_vector->slice_ki_size
            //             + i * b->local_vector->slice_k_size + k] = (linear % 100) / 10.0;
            //     }
            // }
            // x->set_val(1.0, false);
            // A->Mult(*x, *b, false);
            // vec_scale(1.001, *b);
        } else {
            b->read_data(pathname, "array_b");
            // if (strcmp(case_name.c_str(), "GRAPES" ) == 0) x->read_data(pathname, "array_x");
            A->read_data(pathname);
        }
        
        double fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*b, *b);
        if (my_pid == 0) printf(" (b , b ) = %.27e\n", fine_dot);
        fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*x, *x);
        if (my_pid == 0) printf(" (x , x ) = %.27e\n", fine_dot);

        assert(A->check_Dirichlet());
        A->Mult(*b, *y, false);

        fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*y, *y);
        if (my_pid == 0) printf(" (Ab, Ab) = %.27e\n", fine_dot);

#ifdef PROFILE
        {
            par_structMatrix<IDX_TYPE, PC_TYPE, KSP_TYPE> A_low
                (MPI_COMM_WORLD, num_diag, case_idx, case_idy, case_idz, num_proc_x, num_proc_y, num_proc_z);
            int tot_len = A->num_diag 
                    * (A->local_matrix->local_x + A->local_matrix->halo_x * 2)
                    * (A->local_matrix->local_y + A->local_matrix->halo_y * 2)
                    * (A->local_matrix->local_z + A->local_matrix->halo_z * 2);
            for (int i = 0; i < tot_len; i++)
                A_low.local_matrix->data[i] = A->local_matrix->data[i];
            assert(A_low.check_Dirichlet());
            A_low.separate_Diags();
            A_low.Mult(*b, *y, false);
            fine_dot = vec_dot<IDX_TYPE, KSP_TYPE, double>(*y, *y);
            if (my_pid == 0) printf(" (Ab, Ab) = %.27e\n", fine_dot);
        }
#endif
        // MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Abort(MPI_COMM_WORLD, -20221106);


#ifdef WRITE_AOS
        A->write_struct_AOS_bin(pathname, "mat.AOS.bin");
        b->write_data(pathname, "b.AOS.bin");
        x->write_data(pathname, "x.AOS.bin");
        A->write_CSR_bin(pathname);
        b->write_CSR_bin(pathname, "b.bin");
        x->write_CSR_bin(pathname, "x0.bin");
#endif
        
#if KSP_BIT==64 && PC_BIT==32
        precond = new Adaptor_64_for_32(argv + cnt, argc - cnt, true);
#else
        static_assert(  (KSP_BIT==64 && PC_BIT==64) || 
                        (KSP_BIT==32              ) );
        std::string prc_name = "";
        if (argc >= 8)
            prc_name = std::string(argv[cnt++]);
        if (strstr(prc_name.c_str(), "PGS")) {
            SCAN_TYPE type = SYMMETRIC;
            if      (strstr(prc_name.c_str(), "F")) type = FORWARD;
            else if (strstr(prc_name.c_str(), "B")) type = BACKWARD;
            else if (strstr(prc_name.c_str(), "S")) type = SYMMETRIC;
            if (my_pid == 0) printf("  using \033[1;35mpointwise-GS %d\033[0m as preconditioner\n", type);
            precond = new PointGS<IDX_TYPE, PC_TYPE, KSP_TYPE>(type);
        } else if (strstr(prc_name.c_str(), "LGS")) {
            SCAN_TYPE type = SYMMETRIC;
            if      (strstr(prc_name.c_str(), "F")) type = FORWARD;
            else if (strstr(prc_name.c_str(), "B")) type = BACKWARD;
            else if (strstr(prc_name.c_str(), "S")) type = SYMMETRIC;
            if (my_pid == 0) printf("  using \033[1;35mlinewise-GS %d\033[0m as preconditioner\n", type);
            precond = new LineGS<IDX_TYPE, PC_TYPE, KSP_TYPE>(type, VERT, x->comm_pkg);
        } else if (prc_name == "GMG") {
            IDX_TYPE num_discrete = atoi(argv[cnt++]);
            IDX_TYPE num_Galerkin = atoi(argv[cnt++]);
            std::unordered_map<std::string, RELAX_TYPE> trans_smth;
            trans_smth["PGS"]= PGS;
            trans_smth["LGS"]= LGS;
            trans_smth["BILU3d7"] = BILU3d7;
            trans_smth["BILU3d15"] = BILU3d15;
            trans_smth["BILU3d19"] = BILU3d19;
            trans_smth["BILU3d27"] = BILU3d27;
            trans_smth["PILU"] = PILU;
            trans_smth["LU"] = GaussElim;
            std::vector<RELAX_TYPE> rel_types;
            for (IDX_TYPE i = 0; i < num_discrete + num_Galerkin + 1; i++) {
                rel_types.push_back(trans_smth[argv[cnt++]]);
                // if (my_pid == 0) printf("i %d type %d\n", i, rel_types[i]);
            }
            precond = new GeometricMultiGrid<IDX_TYPE, PC_TYPE, KSP_TYPE>
                (num_discrete, num_Galerkin, {}, rel_types);
        } else if (strstr(prc_name.c_str(), "BILU")) {
            BlockILU_type type_3d = ILU_3D27;
            if     (strstr(prc_name.c_str(), "3d7" )) type_3d = ILU_3D7 ;
            else if(strstr(prc_name.c_str(), "3d15")) type_3d = ILU_3D15;
            else if(strstr(prc_name.c_str(), "3d19")) type_3d = ILU_3D19;
            else if(strstr(prc_name.c_str(), "3d27")) type_3d = ILU_3D27;
            else {
                if (my_pid == 0) printf("Error: unsupported types of Blockwise-ILU\n");
                MPI_Abort(MPI_COMM_WORLD, -20230121);
            }
            if (my_pid == 0) printf("  using \033[1;35mblock-ILU type %d\033[0m as preconditioner\n", type_3d);
            precond = new BlockILU<IDX_TYPE, PC_TYPE, KSP_TYPE>(type_3d);
        } else if (prc_name == "PILU") {
            if (my_pid == 0) printf("  using \033[1;35mplane-ILU\033[0m as preconditioner\n");
            precond = new PlaneILU<IDX_TYPE, PC_TYPE, KSP_TYPE>(ILU_2D9);
        } else {
            if (my_pid == 0) printf("NO preconditioner was set.\n");
        }
#endif
        

        if (its_name == "GCR") {
            solver = new GCRSolver<IDX_TYPE, KSP_TYPE, PC_TYPE>;
            ((GCRSolver<IDX_TYPE, KSP_TYPE, PC_TYPE>*)solver)->SetInnerIterMax(restart);
        } else if (its_name == "CG") {
            solver = new CGSolver<IDX_TYPE, KSP_TYPE, PC_TYPE>();
        } else if (its_name == "GMRES") {
            solver = new GMRESSolver<IDX_TYPE, KSP_TYPE, PC_TYPE>();
            ((GMRESSolver<IDX_TYPE, KSP_TYPE, PC_TYPE>*)solver)->SetRestartlen(restart);
        } else {
            if (my_pid == 0) printf("INVALID iterative solver name of %s\nOnly GCR, CG, GMRES, FGMRES available\n", its_name.c_str());
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        solver->SetMaxIter(200);
        if (strcmp(case_name.c_str(), "GRAPES" ) == 0) {// 使用绝对残差
            if (its_name == "GCR") solver->SetAbsTol(1e-10);// 注意：这一版的GCR里用的是点积结果（不开根）判敛，实际是范数的平方
            else solver->SetAbsTol(1e-5);
        } else {// 其它算例使用相对残差
            if (its_name == "GCR") solver->SetRelTol(1e-18);
            else solver->SetRelTol(1e-9);
        }
        if (precond != nullptr) solver->SetPreconditioner(*precond);
        solver->SetOperator(*A);
        
#ifdef DEBUG
        const GeometricMultiGrid<IDX_TYPE, PC_TYPE> & GMG = *((GeometricMultiGrid<IDX_TYPE, PC_TYPE>*)precond);
        
        if (GMG.relax_types[GMG.num_levs-1] == GaussElim) {
            // LU分解
            const PlaneDenseLU<IDX_TYPE, PC_TYPE> & PLU = 
                *((PlaneDenseLU<IDX_TYPE, PC_TYPE>*)(GMG.smoother[GMG.num_levs-1]));
            // PLU.SetOperator(*(GMG.A_array[GMG.num_levs-1]));
            par_structVector<IDX_TYPE, PC_TYPE> rhs(*(GMG.U_array[GMG.num_levs-1])), sol(rhs);
            rhs.set_val(1.0);
            sol.set_val(0.0);
            PLU.Mult(rhs, sol, false);

            double dd = vec_dot<IDX_TYPE, PC_TYPE, double>(sol, sol);
            if (my_pid == 0) printf("(sol, sol) = %.20e\n", dd);
        }

        // 打印矩阵出来看看
        if (my_pid == 0) {
            IDX_TYPE j = 55;
            char filename[100];
            {// 原矩阵
                seq_structMatrix<IDX_TYPE, KSP_TYPE> & mat = *(A->local_matrix);
                assert(mat.num_diag == 19);
                for (int id = 0; id < mat.num_diag; id++) {
                    sprintf(filename, "C_data/Ap.%d.%06d.txt\0", id, my_pid);
                    FILE * fp = fopen(filename, "w+");
                    int kbeg = 0, kend = kbeg + mat.local_z + 2 * mat.halo_z;
                    int ibeg = 0, iend = ibeg + mat.local_x + 2 * mat.halo_x;
                    for (int k = kbeg; k < kend; k++) {
                        for (int i = ibeg; i < iend; i++)
                            fprintf(fp, "%.10e ", mat.data[j * mat.slice_dki_size + i * mat.slice_dk_size + k * mat.num_diag + id]);
                        fprintf(fp, "\n");
                    }
                    fclose(fp);
                }
            }

            for (int lev = 0; lev < num_levs; lev++) {
                const par_structMatrix<IDX_TYPE,PC_TYPE> & par_mat = *(GMG.A_array[lev]);
                const seq_structMatrix<IDX_TYPE,PC_TYPE> &     mat = *(par_mat.local_matrix);
                assert(mat.num_diag == 9);
                int kbeg = 0, kend = kbeg + mat.local_z + 2 * mat.halo_z;
                int ibeg = 0, iend = ibeg + mat.local_x + 2 * mat.halo_x;
                for (int id = 0; id < mat.num_diag; id++) {
                    sprintf(filename, "C_data/A%d.%d.%06d.txt\0", lev, id, my_pid);
                    FILE * fp = fopen(filename, "w+");
                    for (int k = kbeg; k < kend; k++) {
                        for (int i = ibeg; i < iend; i++)
                            fprintf(fp, "%.10e ", mat.data[j * mat.slice_dki_size + i * mat.slice_dk_size + k * mat.num_diag + id]);
                        fprintf(fp, "\n");
                    }
                    // fprintf(fp, "\n\n");
                    fclose(fp);
                }
            }

            // if (GMG.relax_types[0] == PILU) {
            //     for (int lev = 0; lev < num_levs; lev++) {
            //         const PlaneILU<IDX_TYPE, PC_TYPE> & ilu = *((PlaneILU<IDX_TYPE, PC_TYPE>*)GMG.smoother[lev]);
            //         const seq_structMatrix<IDX_TYPE, PC_TYPE> & L = *(ilu.L);
            //         const seq_structMatrix<IDX_TYPE, PC_TYPE> & U = *(ilu.U);
            //         int kbeg = 0, kend = kbeg + GMG.A_array[lev]->local_matrix->local_z + 2 * GMG.A_array[lev]->local_matrix->halo_z;
            //         int ibeg = 0, iend = ibeg + GMG.A_array[lev]->local_matrix->local_x + 2 * GMG.A_array[lev]->local_matrix->halo_x;
            //         sprintf(filename, "C_data/L%d.%06d.txt\0", lev, my_pid);
            //         FILE * fp = fopen(filename, "w+");
            //         for (int i = ibeg; i < iend; i++) {
            //             for (int k = kbeg; k < kend; k++) {
            //                 fprintf(fp, "%3d %3d ", i, k);
            //                 for (int d = 0; d < L.num_diag; d++)
            //                     fprintf(fp, "%.6e ", L.data[j * L.slice_dki_size + i * L.slice_dk_size + k * L.num_diag + d]);
            //                 fprintf(fp, "\n");
            //             }
            //         }
            //         fclose(fp);

            //         sprintf(filename, "C_data/U%d.%06d.txt\0", lev, my_pid);
            //         fp = fopen(filename, "w+");
            //         for (int i = ibeg; i < iend; i++) {
            //             for (int k = kbeg; k < kend; k++) {
            //                 fprintf(fp, "%3d %3d ", i, k);
            //                 for (int d = 0; d < U.num_diag; d++)
            //                     fprintf(fp, "%.6e ", U.data[j * U.slice_dki_size + i * U.slice_dk_size + k * U.num_diag + d]);
            //                 fprintf(fp, "\n");
            //             }
            //         }
            //         fclose(fp);

            //         // 拿出来做一次求解
            //         IDX_TYPE outer_dim = 2 * L.halo_x + L.local_x, inner_dim = 2 * L.halo_z + L.local_z;
            //         seq_structVector<IDX_TYPE, PC_TYPE> rhs(L.local_x, L.local_y, L.local_z, L.halo_x, L.halo_y, L.halo_z);
            //         rhs = 1.0;
            //         rhs.set_halo(1.0);
            //         seq_structVector<IDX_TYPE, PC_TYPE> tmp(rhs), sol(rhs);

            //         struct_trsv_forward_2d( L.data   + j * L.slice_dki_size, 
            //                                 rhs.data + j * rhs.slice_ki_size,
            //                                 tmp.data + j * tmp.slice_ki_size,
            //                                 outer_dim, inner_dim, L.num_diag, ilu.stencil_offset);

            //         sprintf(filename, "C_data/tmp%d.%06d.txt\0", lev, my_pid);
            //         fp = fopen(filename, "w+");
            //         for (int i = ibeg; i < iend; i++) {
            //             for (int k = kbeg; k < kend; k++) {
            //                 fprintf(fp, "%3d %3d %.6e\n", i, k, tmp.data[j * tmp.slice_ki_size 
            //                     + i * tmp.slice_k_size + k]);
            //             }
            //         }
            //         fclose(fp);

            //         struct_trsv_backward_2d(U.data   + j * U.slice_dki_size, 
            //                                 tmp.data + j * tmp.slice_ki_size,
            //                                 sol.data + j * sol.slice_ki_size,
            //                                 outer_dim, inner_dim, U.num_diag, ilu.stencil_offset + 2 * L.num_diag);

            //         sprintf(filename, "C_data/x%d.%06d.txt\0", lev, my_pid);
            //         fp = fopen(filename, "w+");
            //         for (int i = ibeg; i < iend; i++) {
            //             for (int k = kbeg; k < kend; k++) {
            //                 fprintf(fp, "%3d %3d %.6e\n", i, k, sol.data[j * sol.slice_ki_size 
            //                     + i * sol.slice_k_size + k]);
            //             }
            //         }
            //         fclose(fp);
            //     }
            // }
            printf("\nEnter\n");
            MPI_Abort(MPI_COMM_WORLD, -555);
            // sprintf(filename, "printA_3d19.%06d.txt\0", my_pid);
            // fp = fopen(filename, "w+");
            // const seq_structMatrix<IDX_TYPE,PC_TYPE> & mat = *(A.local_matrix);
            // assert(mat.num_diag == 19);
            // IDX_TYPE j = mat.local_y / 2;
            // int kbeg = 0, kend = kbeg + mat.local_z + 2 * mat.halo_z;
            // int ibeg = 0, iend = ibeg + mat.local_x + 2 * mat.halo_x;
            // for (int id = 0; id < mat.num_diag; id++) {
            //     fprintf(fp, "diag %d\n", id);
            //     for (int k = kend - 1; k >= kbeg; k--) {
            //         for (int i = ibeg; i < iend; i++)
            //             fprintf(fp, "%.8e ", mat.data[j * mat.slice_dki_size + i * mat.slice_dk_size + k * mat.num_diag + id]);
            //         fprintf(fp, "\n");
            //     }
            //     fprintf(fp, "\n\n");
            // }
            // fclose(fp);
        }
#endif

#ifdef TRC
        if (my_pid == 0) printf("\033[1;35mTruncate to __fp16...\033[0m\n");
        solver->truncate();
#endif

#ifdef STAT_NNZ
        {
            int num_inter = 25;
            double lb[num_inter], ub[num_inter];
            int nnz_cnt[num_inter];
            lb[0] = 1e-12;
            for (int i = 0; i < num_inter; i++) {
                ub[i] = lb[i] * 10.0;
                if (i < num_inter - 1)
                    lb[i + 1] = ub[i];
                nnz_cnt[i] = 0;
            }
            int zero_cnt = 0;
            int num_theta = 10;
            double lb_theta[num_theta] = {1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
            double ub_theta[num_theta];
            int theta_cnt[num_theta];
            for (int i = 0; i < num_theta; i++) {
                ub_theta[i] = lb_theta[i] * 10.0;
                theta_cnt[i] = 0;
            }

            assert(num_procs == 1);
            const int jbeg = A->local_matrix->halo_y, jend = jbeg + A->local_matrix->local_y;
            const int ibeg = A->local_matrix->halo_x, iend = ibeg + A->local_matrix->local_x;
            const int kbeg = A->local_matrix->halo_z, kend = kbeg + A->local_matrix->local_z;
            const int nd = A->num_diag; assert(nd % 2 == 1);
            const int diag_id = (nd - 1) / 2;
            int tot_nnz = 0;
            for (int j = jbeg; j < jend; j++)
            for (int i = ibeg; i < iend; i++)
            for (int k = kbeg; k < kend; k++) {
                KSP_TYPE abs_nnz[nd];
                const KSP_TYPE* ptr = A->local_matrix->data + j * A->local_matrix->slice_dki_size + i * A->local_matrix->slice_dk_size + k * nd;
                for (int d = 0; d < nd; d++)
                    abs_nnz[d] = fabs(ptr[d]);
                for (int d = 0; d < nd; d++) {
                    if (abs_nnz[d] == 0.0) {
                        zero_cnt++;
                        continue;
                    }
                    for (int t = 0; t < num_inter; t++) {
                        if (abs_nnz[d] >= lb[t] && abs_nnz[d] < ub[t]) {
                            nnz_cnt[t] ++;
                            break;
                        }
                    }
                }
                double max_offd = 0.0, min_offd = 1e30;
                for (int d = 0; d < nd; d++) {
                    if (abs_nnz[d] == 0.0) continue;
                    tot_nnz ++;
                    if (nd == diag_id) continue;
                    max_offd = std::max(max_offd, abs_nnz[d]);
                    min_offd = std::min(min_offd, abs_nnz[d]);
                }
                double theta = min_offd / max_offd;
                for (int t = 0; t < num_theta; t++) {
                    if (theta >= lb_theta[t] && theta < ub_theta[t]) {
                        theta_cnt[t] ++;
                        break;
                    }
                }
            }
            printf("Offd nnz\n");
            for (int i = 0; i < num_inter; i++) {
                double ratio = (double) nnz_cnt[i] / (double) tot_nnz;
                printf("[%.2e,%.2e): %d %.6f\n", lb[i], ub[i], nnz_cnt[i], ratio);
            }
            printf("theta\n");
            int tot_ndof = (jend - jbeg) * (iend - ibeg) * (kend - kbeg);
            for (int i = 0; i < num_theta; i++) {
                double ratio = (double) theta_cnt[i] / (double) tot_ndof;
                printf("[%.2e,%.2e): %d %.6f\n", lb_theta[i], ub_theta[i], theta_cnt[i], ratio);
            }
        }
#endif

        // if (case_name == "LASER" && prc_name == "GMG") {
        //     assert(num_Galerkin >= 0);
        //     PC_TYPE  wgts[num_Galerkin+1];
        //     for (int i = 0; i < num_Galerkin+1; i++) wgts[i] = 1.2;
        //     ((GeometricMultiGrid<IDX_TYPE, PC_TYPE, KSP_TYPE>*)precond)->SetRelaxWeights(wgts, num_Galerkin+1);
        // }

        double t1 = wall_time();
        solver->Mult(*b, *x, false);
        t1 = wall_time() - t1;
        if (my_pid == 0) printf("Solve costs %.6f s\n", t1);
        // x->write_data(pathname, "array_x_exact." + std::to_string(solver->final_iter));
        double min_times[NUM_KRYLOV_RECORD], max_times[NUM_KRYLOV_RECORD], avg_times[NUM_KRYLOV_RECORD];
        MPI_Allreduce(solver->part_times, min_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(solver->part_times, max_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(solver->part_times, avg_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        for (int i = 0; i < NUM_KRYLOV_RECORD; i++)
            avg_times[i] /= num_procs;
        if (my_pid == 0) {
            printf("prec time min/avg/max %.3e %.3e %.3e\n", min_times[PREC], avg_times[PREC], max_times[PREC]);
            printf("oper time min/avg/max %.3e %.3e %.3e\n", min_times[OPER], avg_times[OPER], max_times[OPER]);
            printf("axpy time min/avg/max %.3e %.3e %.3e\n", min_times[AXPY], avg_times[AXPY], max_times[AXPY]);
            printf("dot  tune min/avg/max %.3e %.3e %.3e\n", min_times[DOT ], avg_times[DOT ], max_times[DOT ]);
        }

        A->Mult(*x, *y, false);
        vec_add(*b, -1.0, *y, *y);
        double true_r_norm = vec_dot<IDX_TYPE, KSP_TYPE, double>(*y, *y);
        true_r_norm = sqrt(true_r_norm);

        double b_norm = vec_dot<IDX_TYPE, KSP_TYPE, double>(*b, *b);
        b_norm = sqrt(b_norm);
         if (my_pid == 0) printf("\033[1;35mtrue ||r|| = %20.16e ||r||/||b||= %20.16e\033[0m\n", 
            true_r_norm, true_r_norm / b_norm);

        if (b != nullptr) {delete b; b = nullptr;}
        if (x != nullptr) {delete x; x = nullptr;}
        if (y != nullptr) {delete y; y = nullptr;}
        if (A != nullptr) {delete A; A = nullptr;}
        if (solver != nullptr) {delete solver; solver = nullptr;}
        if (precond != nullptr) {delete precond; precond = nullptr;}
    }

    MPI_Finalize();
    return 0;
}