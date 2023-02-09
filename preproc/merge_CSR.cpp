#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
// typedef int idx_t;
typedef long long idx_t;
typedef float data_t;

int main(int argc, char* argv[])
{
    const idx_t gx = atoi(argv[1]),
                    gy = atoi(argv[2]),
                    gz = atoi(argv[3]);
    const idx_t glb_nrows = gx * gy * gz;
    const int file_cnt = atoi(argv[4]);
    const idx_t ly = gy / file_cnt; assert(gy % file_cnt == 0);
    const idx_t loc_nrows = gx * ly * gz;

    idx_t * row_ptr = new idx_t [glb_nrows + 1];
    row_ptr[0] = 0;
    idx_t acc_nnz[file_cnt + 1];// 累计的非零元数目
    acc_nnz[0] = 0;

    idx_t * dist_buf = new idx_t [loc_nrows + 1];
    for (int f = 0; f < file_cnt; f++) {
        std::string filename = "Ai.bin." + std::to_string(f);
        FILE * fp = fopen(filename.c_str(), "rb");
        idx_t size = fread(dist_buf, sizeof(idx_t), loc_nrows + 1, fp); assert(size == loc_nrows + 1);
        fclose(fp);

        assert(dist_buf[0] == 0);
        acc_nnz[f+1] = acc_nnz[f] + dist_buf[loc_nrows];
        printf("%d loc nnz %lld\n", f, acc_nnz[f+1]);

        idx_t beg_row_idx = f * loc_nrows;
        idx_t last_nnz = row_ptr[beg_row_idx];
        for (idx_t i = 1; i < loc_nrows + 1; i++) {
            row_ptr[i + beg_row_idx] = dist_buf[i] + last_nnz;
        }
    }
    delete dist_buf; dist_buf = nullptr;

    idx_t glb_nnz = row_ptr[glb_nrows];
    printf("glb nnz %lld\n", glb_nnz);

    idx_t * col_idx = new idx_t [glb_nnz];
    data_t * vals = new data_t [glb_nnz];
    for (int f = 0; f < file_cnt; f++) {
        std::string filename = "Aj.bin." + std::to_string(f);
        FILE * fp = fopen(filename.c_str(), "rb");
        idx_t size = fread(col_idx + acc_nnz[f], sizeof(idx_t), acc_nnz[f+1] - acc_nnz[f], fp);
        if (size != acc_nnz[f+1] - acc_nnz[f]) {
            printf("hope %lld actual %lld\n", acc_nnz[f+1] - acc_nnz[f], size);
            exit(1);
        }
        fclose(fp);

        filename = "Av.bin." + std::to_string(f);
        fp = fopen(filename.c_str(), "rb");
        size = fread(vals + acc_nnz[f], sizeof(data_t), acc_nnz[f+1] - acc_nnz[f], fp);
        if (size != acc_nnz[f+1] - acc_nnz[f]) {
            printf("hope %lld actual %lld\n", acc_nnz[f+1] - acc_nnz[f], size);
            exit(1);
        }
        fclose(fp);
    }

    data_t * b = new data_t [glb_nrows];
    data_t * x = new data_t [glb_nrows];
    for (int f = 0; f < file_cnt; f++) {
        std::string filename = "b.bin." + std::to_string(f);
        FILE * fp = fopen(filename.c_str(), "rb");
        idx_t size = fread(b + f * loc_nrows, sizeof(data_t), loc_nrows, fp); assert(size == loc_nrows);
        fclose(fp);

        filename = "x0.bin." + std::to_string(f);
        fp = fopen(filename.c_str(), "rb");
        size = fread(x + f * loc_nrows, sizeof(data_t), loc_nrows, fp); assert(size == loc_nrows);
        fclose(fp);
    }

    
    FILE * fp = fopen("Ai.bin", "wb+");
    fwrite(row_ptr, sizeof(idx_t), glb_nrows + 1, fp);
    fclose(fp);
    
    fp = fopen("Aj.bin", "wb+");
    fwrite(col_idx, sizeof(idx_t), glb_nnz, fp);
    fclose(fp);
    fp = fopen("Av.bin", "wb+");
    fwrite(vals, sizeof(data_t), glb_nnz, fp);
    fclose(fp);

    fp = fopen("b.bin", "wb+");
    fwrite(b, sizeof(data_t), glb_nrows, fp);
    fclose(fp);
    fp = fopen("x0.bin", "wb+");
    fwrite(x, sizeof(data_t), glb_nrows, fp);
    fclose(fp);

    // 做一次SPMV检验
    double dot_Ab = 0.0;
    for (idx_t i = 0; i < glb_nrows; i++) {
        idx_t pbeg = row_ptr[i], pend = row_ptr[i+1];
        data_t tmp = 0.0;
        for (idx_t p = pbeg; p < pend; p++) {
            tmp += vals[p] * b[col_idx[p]];
        }
        x[i] = tmp;
        dot_Ab += tmp * tmp;
    }
    printf("(A*b, A*b) = %.10e\n", dot_Ab);
    
    delete row_ptr; delete col_idx; delete vals;
    delete x; delete b;
    return 0;
}