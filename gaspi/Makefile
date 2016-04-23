#GPI2_DIR = $(HOME)/GPI2-1.1.1_intelMPI
#MPI_DIR= /apps/INTEL/impi/4.1.3.049/
#CC = mpiicc

MPI_DIR=/apps/all/MVAPICH2/2.1-iccifort-2015.3.187-GNU-5.1.0-2.25
GPI2_DIR=/home/chrisimm/GPI2-next_mvapich-2.1
#GPI2_DIR=/home/chrisimm/GPI2-1.1.1_read_notify_mvapich-2.1
CC=$(MPI_DIR)/bin/mpicc

#MPI_DIR=/apps/all/impi/5.0.3.048
#GPI2_DIR=$(HOME)/GPI2-next_impi-5.0.3

#GPI2_DIR = $(HOME)/GPI2-next_impi-4.1.1
#MPI_DIR=/apps/all/impi/4.1.1.036-iccifort-2013.5.192
#CC = $(MPI_DIR)/bin64/mpiicc

#GPI2_DIR = $(HOME)/GPI2_mvapich_thread_multiple
#MPI_DIR= /export/opt/mvapich2-2.2b-enable-mpi-thread-multiple
#CC = $(MPI_DIR)/bin/mpicc

#GPI2_DIR = $(HOME)/GPI2_OpenMPI_thread_multiple
#MPI_DIR=$(HOME)/openmpi-1.10.2-enableThreadsMultiple
#CC = $(MPI_DIR)/bin/mpicc


NAME += Pipelined_transpose_WN
NAME += Pipelined_transpose_RN

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
#CFLAGS += -O3 -debug -mmic
CFLAGS += -O3
#CFLAGS += -openmp
CFLAGS += -openmp
CFLAGS += -std=c99
#CFLAGS += -DDEBUG
CFLAGS += -DUSE_ALIGNMENT

###############################################################################

#INCLUDE_DIR += $(MPI_DIR)/include64 
INCLUDE_DIR += $(MPI_DIR)/include
INCLUDE_DIR += $(GPI2_DIR)/include 
INCLUDE_DIR += ..

#LIBRARY_DIR += $(MPI_DIR)/lib64
LIBRARY_DIR += $(MPI_DIR)/lib64
LIBRARY_DIR += $(GPI2_DIR)/lib64

#LIBRARY_DIR += $(MPI_DIR)/lib64/mic
#LIBRARY_DIR += $(GPI2_DIR)/lib64/mic

LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))

ifeq "$(CC)" "mpiicc"
  LDFLAGS += -openmp-link static
endif

CFLAGS += $(addprefix -I,$(INCLUDE_DIR))

LIB += ibverbs
#LIB += GPI2
LIB += GPI2-dbg
LIB += m
#LIB += mpi_mt
#LIB += mpi
LIB += mpich

OBJ += ../data_global
OBJ += ../data
OBJ += ../now
OBJ += ../threads
OBJ += queue



###############################################################################

BIN = $(addsuffix .exe, $(NAME))

default: $(BIN)

%.exe: %.o $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

###############################################################################

%.run: %.exe
	@$(CMD_RUN) $^ | grep ^#

.PHONY: run

run: $(addprefix $(CURDIR)/, $(addsuffix .run, $(NAME)))

###############################################################################

.PHONY: clean objclean

objclean:
	rm -f ../*.o *.o

clean: objclean
	rm -f $(BIN)
