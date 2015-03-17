GPI2_DIR = $(HOME)/GPI2-1.1.1_intel-5.0.1.035
MPI_DIR=/export/opt/intel2015/impi/5.0.1.035
CC = $(MPI_DIR)/bin64/mpiicc

#GPI2_DIR = $(HOME)/GPI2-1.1.1_intel-5.0.1.035.mic
#MPI_DIR=/export/opt/intel2015/impi/5.0.1.035/mic
#CC = $(MPI_DIR)/bin/mpiicc

GPI2_DIR = $(HOME)/GPI2-1.1.1_mvapich2-2.1a_enable-threads-multiple
MPI_DIR=$(HOME)/mvapich2-2.1a_enable-threads-multiple
CC = $(MPI_DIR)/bin/mpicc

NAME += Pipelined_transpose

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
#CFLAGS += -O3 -debug -mmic
CFLAGS += -O3 -debug
#CFLAGS += -openmp
CFLAGS += -fopenmp
CFLAGS += -std=c99
#CFLAGS += -DDEBUG
CFLAGS += -DUSE_ALIGNMENT

###############################################################################

INCLUDE_DIR += $(MPI_DIR)/include 
INCLUDE_DIR += $(GPI2_DIR)/include 
INCLUDE_DIR += ..

#LIBRARY_DIR += $(MPI_DIR)/lib64
LIBRARY_DIR += $(MPI_DIR)/lib
LIBRARY_DIR += $(GPI2_DIR)/lib64

#LIBRARY_DIR += $(MPI_DIR)/lib64/mic
#LIBRARY_DIR += $(GPI2_DIR)/lib64/mic

LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))

ifeq "$(CC)" "mpiicc"
  LDFLAGS += -openmp-link static
endif

CFLAGS += $(addprefix -I,$(INCLUDE_DIR))

LIB += ibverbs
LIB += GPI2
LIB += m
#LIB += mpi_mt
LIB += mpich

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
