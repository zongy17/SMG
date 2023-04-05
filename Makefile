CXX = mpicxx
CFLAGS = -std=c++17 -Wall -fopenmp -march=armv8.2-a+fp16

OPT = -O3 -g
#OPT = -Ofast -DNDEBUG
#OPT = -DTRC -O3 -g

CFLAGS += $(OPT)

SUPERLU_HOME = /storage/hpcauser/zongyi/HUAWEI/software/superlu/5.3.0
SUPERLU_INC = -I$(SUPERLU_HOME)/include
LSUPERLU = -L$(SUPERLU_HOME)/lib64 -lsuperlu -lblas

INC = $(SUPERLU_INC)
EXT_LD = $(LSUPERLU)

CL  = mpicxx
LFLAGS = -lm -fopenmp -lstdc++

all:    smg_All64.exe\
        smg_K64P32D16.exe

smg_All64.exe : main.cpp
	$(CL) $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_CALC_BIT=64 -DPC_DATA_BIT=64 $(LFLAGS) $(EXT_LD) -o $@
smg_K64P32D16.exe : main.cpp
	$(CL) $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_CALC_BIT=32 -DPC_DATA_BIT=16 $(LFLAGS) $(EXT_LD) -o $@

.PHONY: clean

clean : 
	rm smg_*.exe *.o
