For all cases, case_name = "LASER" while cx, cy, cz, and px, py, pz are shown below specifically.

# Case: size of 128x128x128

## ARM-based platform
executable = "smg_All64.exe" and "smg_K64P32D16.exe".
cx = 128, cy = 128, cz = 128
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  1   | N0.5P64  | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  1   |  N1P64   | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  2   |  N2P64   | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  4   |  N4P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  8   |  N8P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
| 16   | N16P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |

This is for strong scalability tests.
## X86 platform
executable = "smg_All64.exe".
cx = 128, cy = 128, cz = 128
| Node | Rankfile | px | py | pz | other_param |
|------|:--------:|---:|----|----|-------------|
|  1   | N0.5P8  | 2  | 2  | 2  |CG 10 GMG 0 3 PGS PGS PGS LU |
|  1   |  N1P64   | 4  | 4  | 4  |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
|  2   |  N2P64   | 4  | 4  | 4  |CG 10 GMG 0 3 PGS PGS PGS LU|
|  4   |  N4P64   |  8 | 8  |  8 |CG 10 GMG 0 3 PGS PGS PGS LU|
|  8   |  N8P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |
| 16   | N16P512  |  8 | 8  |  8 |CG 10 GMG 0 4 PGS PGS PGS PGS LU |

This is for strong scalability tests.