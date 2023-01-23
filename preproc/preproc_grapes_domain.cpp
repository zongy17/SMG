#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <cstring>

typedef float data_t;

int main(int argc, char * argv[])
{
    int cnt = 1;
    const int   gx = atoi(argv[cnt++]),
                gy = atoi(argv[cnt++]),
                gz = atoi(argv[cnt++]);
    const int   file_gx = gx + 2,
                file_gy = gy + 2;
    
    const char * filenames[] = {
        "varA1",
        "varA2",
        "varA3",
        "varA4",
        "varA5",
        "varA6",
        "varA7",
        "varA8",
        "varA9",
        "varA10",
        "varA11",
        "varA12",
        "varA13",
        "varA14",
        "varA15",
        "varA16",
        "varA17",
        "varA18",
        "varA19",
        "varB",
        "varX0",
        "varX"
    };
    const int num_files = sizeof(filenames) / sizeof(char *);
    const int map_GRAPES_to_miniapp[19] = {
        // 0  1   2  3   4   5  6  7  8   9 10  11 12  13  14 15  16 17  18 
        9, 6, 12, 2, 16, 18, 4, 0, 14, 8, 5, 11, 1, 15, 10, 7, 13, 3, 17
    };

    const size_t in_len = file_gx * file_gy * gz;
    const size_t out_len = gx * gy * gz;
    data_t * in_buf = new data_t [in_len];
    data_t * out_buf= new data_t [out_len];

    assert( file_gx - gx == 2 &&
            file_gy - gy == 2);
    for (int f = 0; f < num_files; f++) {
        FILE * fp = fopen(filenames[f], "rb");
        size_t ret = fread(in_buf, sizeof(data_t), in_len, fp); assert(ret == in_len);
        fclose(fp);
        // transform
        int id = -1;
        if (f < 19) id = map_GRAPES_to_miniapp[f];
        // 读入数据的顺序从内到外为 i, j, k
        for (int k = 0; k < gz; k++)
        for (int j = 0; j < gy; j++)
        for (int i = 0; i < gx; i++) {
            int fj = j + 1;
            int fi = i + 1;
            out_buf[k + gz * (i + j * gx)] = in_buf[fi + file_gx * (fj + file_gy * k)];
        }

        std::string outfilename;
        if (id != -1) outfilename = "array_a." + std::to_string(id);
        else {
            if      (strcmp(filenames[f], "varB") == 0) outfilename = "array_b";
            else if (strcmp(filenames[f], "varX") == 0) outfilename = "array_x_exact";
            else if (strcmp(filenames[f], "varX0")== 0) outfilename = "array_x";
            else  {
                printf("%s\n", filenames[f]);
                assert(false);
            }
        }
        fp = fopen(outfilename.c_str(), "wb+");
        printf("writing %s\n", outfilename.c_str());
        ret = fwrite(out_buf, sizeof(data_t), out_len, fp); assert(ret == out_len);
        fclose(fp);
    }


    delete in_buf;
    delete out_buf;
    return 0;
}