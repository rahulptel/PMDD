# --- SYSTEM ---

SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
BASISDIR   = /opt
PLATFORM   = linux64


# --- DIRECTORIES ---

CCC = g++
NVCC_CANDIDATE := $(firstword $(wildcard /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc))
NVCC ?= $(if $(NVCC_CANDIDATE),$(NVCC_CANDIDATE),nvcc)
NVCC_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)

BOOSTDIR   = $(BASISDIR)/boost
ifeq ($(machine), cc)
	BOOSTDIR = $(BOOST_ROOT)
endif

GUROBIDIR = /opt/gurobi/$(PLATFORM)
GUROBIINCDIR = $(GUROBIDIR)/include
GUROBILIBDIR = $(GUROBIDIR)/lib


# --- FLAGS ---

CCOPT = -std=c++11 -m64 -fPIC -fno-strict-aliasing -fexceptions -Wno-deprecated-declarations -Wno-ignored-attributes

# --- OPTIMIZATION FLAGS ---

DEBUG_OPT = -DNDEBUG -O3
#DEBUG_OPT = -g3 -O0

#PROF = -pg
PROF =


# Flags
CFLAGS  = -I$(BOOSTDIR)/include -c $(PROF) $(DEBUG_OPT)
LDFLAGS = -lm -pthread

# Base compilation flags
CFLAGS  += $(CCOPT)

# Uncomment if using Gurobi
#CFLAGS += -I$(GUROBIINCDIR)
#LDFLAGS += -L$(GUROBILIBDIR) -lgurobi_c++ -lgurobi75 -lm 


# --- APPLICATION-SPECIFIC OPTIONS ---

# number of objective functions
NUM_OBJS=3
ENABLE_CUDA ?= 1
ENABLE_OPENMP ?= 1

CFLAGS += -DNOBJS=$(NUM_OBJS)

ifeq ($(ENABLE_OPENMP),1)
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

ifeq ($(ENABLE_CUDA),1)
CFLAGS += -DUSE_CUDA
CUDAFLAGS = -std=c++14 -O3 -DUSE_CUDA -DNOBJS=$(NUM_OBJS) -I$(BOOSTDIR)/include
LDFLAGS += -lcudart
endif

BIN_DIR = ../../resources/bin/pmdd
EXECUTABLE_NAME = multiobj_nobjs$(NUM_OBJS)
EXECUTABLE = $(BIN_DIR)/$(EXECUTABLE_NAME)

ifneq ($(filter clean,$(MAKECMDGOALS)),clean)
ifeq ($(ENABLE_CUDA),1)
ifeq ($(NVCC_MAJOR),)
$(error could not detect nvcc version from '$(NVCC)'. Please install/configure nvcc >= 12)
endif
ifeq ($(shell [ $(NVCC_MAJOR) -lt 12 ] && echo 1 || echo 0),1)
$(error nvcc >= 12 required for CUDA build; detected nvcc $(NVCC_MAJOR) at '$(NVCC)'. Use /usr/local/cuda/bin/nvcc)
endif
endif
endif


# ---- COMPILE  ----
SRC_DIRS  := . $(shell find bdd enum instances mdd util -type d)
OBJ_DIR   := obj/nobjs$(NUM_OBJS)
OBJ_DIRS  := $(OBJ_DIR) $(addprefix $(OBJ_DIR)/,$(filter-out .,$(SRC_DIRS)))

SOURCES_CPP := main.cpp $(shell find bdd enum instances mdd util -name '*.cpp')
SOURCES_CPP := $(filter-out enum/gpu/cuda_stubs.cpp, $(SOURCES_CPP))

OBJ_FILES   := $(addprefix $(OBJ_DIR)/, $(SOURCES_CPP:.cpp=.o))
ifeq ($(ENABLE_CUDA),1)
CUDA_SOURCES  := $(shell find bdd enum instances mdd util -name '*.cu')
CUDA_OBJ_FILES := $(addprefix $(OBJ_DIR)/, $(CUDA_SOURCES:.cu=.o))
OBJ_FILES     += $(CUDA_OBJ_FILES)
vpath %.cu $(SRC_DIRS)
else
STUB_SOURCES := enum/gpu/cuda_stubs.cpp
OBJ_FILES   += $(addprefix $(OBJ_DIR)/, $(STUB_SOURCES:.cpp=.o))
endif

vpath %.cpp $(SRC_DIRS)

# ---- TARGETS ----

.PHONY: all universal clean

all: universal
universal: $(EXECUTABLE)

$(EXECUTABLE): makedir $(OBJ_FILES) 
	$(CCC) $(OBJ_FILES) $(LDFLAGS) $(PROF) -o $@

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIRS)
	$(CCC) $(CFLAGS) $< -o $@

$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIRS)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

makedir: $(OBJ_DIRS) $(BIN_DIR)

$(OBJ_DIRS):
	@mkdir -p $@

$(BIN_DIR):
	@mkdir -p $@

clean:
	@rm -rf obj
	@rm -f $(BIN_DIR)/multiobj_nobjs* $(BIN_DIR)/multiobj
