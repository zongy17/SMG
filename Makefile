CXX = mpicxx
CFLAGS = -O3 -Wall -g -fopenmp -march=armv8.2-a+fp16 #-DPROFILE
#CFLAGS = -DTRC -O3 -Wall -g -fopenmp -march=armv8.2-a+fp16
#CFLAGS = -Ofast -fopenmp -march=armv8.2-a+fp16 -DNDEBUG #-DPROFILE
#CFLAGS = -Ofast -fopenmp -march=armv8.2-a+fp16 -DNDEBUG -DTRC

SUPERLU_HOME = /storage/hpcauser/zongyi/HUAWEI/software/superlu/5.3.0
SUPERLU_INC = -I$(SUPERLU_HOME)/include
LSUPERLU = -L$(SUPERLU_HOME)/lib64 -lsuperlu -lblas

INC = $(SUPERLU_INC)
EXT_LD = $(LSUPERLU)

CL  = mpicxx
LFLAGS = -lm -fopenmp -lstdc++

all: smg_All64 smg_All32 #smg_PC16 #smg_All80

#smg_All80 : main_All80.o
#	$(CL) $^ $(LFLAGS) -o $@

smg_All64 : main_All64.o 
	$(CL) $^ $(LFLAGS) $(EXT_LD) -o $@

smg_All32 : main_All32.o 
	$(CL) $^ $(LFLAGS) $(EXT_LD) -o $@

smg_PC16 : main_PC16.o
	$(CL) $^ $(LFLAGS) -o $@

#main_All80.o : main.cpp
#	$(CXX) -c $^ $(CFLAGS) -DKSP_BIT=80 -DPC_BIT=80 -o $@

main_All64.o : main.cpp
	$(CXX) -c $^ $(CFLAGS) $(INC) -DKSP_BIT=64 -DPC_BIT=64 -o $@

main_All32.o : main.cpp
	$(CXX) -c $^ $(CFLAGS) $(INC) -DKSP_BIT=32 -DPC_BIT=32 -o $@

main_PC16.o : main.cpp
	$(CXX) -c $^ $(CFLAGS) -DKSP_BIT=64 -DPC_BIT=16 -o $@

.PHONY: clean

clean : 
	rm smg_All64 smg_All32 smg_PC16 *.o #smg_All80
