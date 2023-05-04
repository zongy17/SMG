CXX = mpicxx
CFLAGS = -std=c++17 -Wall -fopenmp -march=armv8.2-a+fp16

OPT = -O3 -g
#OPT = -Ofast -DNDEBUG
#OPT = -DTRC -O3 -g

CFLAGS += $(OPT) -DCROSS_POLAR #-DCOMPRESS

SUPERLU_HOME = /storage/hpcauser/zongyi/HUAWEI/software/superlu/5.3.0
SUPERLU_INC = -I$(SUPERLU_HOME)/include
LSUPERLU = -L$(SUPERLU_HOME)/lib64 -lsuperlu -lblas

INC = $(SUPERLU_INC)
EXT_LD = $(LSUPERLU)

CL  = mpicxx
LFLAGS = -lm -fopenmp -lstdc++

all: 	smg_All32.exe\
	smg_K32P16.exe
	 #smg_PC16 #smg_All80

#smg_All80 : main_All80.o
#	$(CL) $^ $(LFLAGS) -o $@

#smg_All64.exe : main.cpp
#	$(CL) $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_BIT=64 $(LFLAGS) $(EXT_LD) -o $@
#smg_K64P32.exe : main.cpp
#	$(CL) $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_BIT=32 $(LFLAGS) $(EXT_LD) -o $@
smg_All32.exe : main.cpp
	$(CL) $^ $(CFLAGS) $(INC) -DKSP_BIT=32 -DPC_BIT=32 $(LFLAGS) $(EXT_LD) -o $@
smg_K32P16.exe: main.cpp
	$(CL) $^ $(CFLAGS) $(INC) -DKSP_BIT=32 -DPC_BIT=16 $(LFLAGS) $(EXT_LD) -o $@

#smg_All64.exe : main_All64.o 
#	$(CL) $^ $(LFLAGS) $(EXT_LD) -o $@
#smg_K64P32.exe : main_K64P32.o
#	$(CL) $^ $(LFLAGS) $(EXT_LD) -o $@
#smg_All32.exe : main_All32.o 
#	$(CL) $^ $(LFLAGS) $(EXT_LD) -o $@
#smg_K32P16.exe : main_K32P16.o
#	$(CL) $^ $(LFLAGS) $(EXT_LD) -o $@
#main_All64.o : main.cpp
#	$(CXX) -c $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_BIT=64 -o $@
#main_K64P32.o : main.cpp
#	$(CXX) -c $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_BIT=32 -o $@
#main_All32.o : main.cpp
#	$(CXX) -c $^ $(CFLAGS) $(INC) -DKSP_BIT=32 -DPC_BIT=32 -o $@
#main_K32P16.o : main.cpp
#	$(CXX) -c $^ $(CFLAGS) $(INC) -DKSP_BIT=32 -DPC_BIT=16 -o $@

.PHONY: clean

clean : 
	rm smg_All32.exe smg_K32P16.exe *.o
