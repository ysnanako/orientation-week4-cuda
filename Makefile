# if some source files in the same folder that you don't want to compile
# use this to select specific source file
# -----
SRC_DIR := ./source/
INC_DIR=./include/
DEG_DIR=./debug/

SRCS := $(wildcard $(SRC_DIR)*.cpp)       # this use wildcard function #查找source file中的所有.cpp檔案
# 2023.08.24 wlhuang
CU_SRCS :=$(wildcard $(SRC_DIR)*.cu)       # this use wildcard function #查找source file中的所有.cu檔案
# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-11.7
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS += -lcudart
CUDA_LINK_LIBS += -lcuda
# -----

OBJS := $(patsubst %.cpp, $(DEG_DIR)%.o, $(notdir $(SRCS)))    # this will replace *.cpp to *.o by patsubst function
CUDA_OBJS := $(patsubst %.cu, $(DEG_DIR)%.o, $(notdir $(CU_SRCS)))    # this will replace *.cu to *.o by patsubst function
#OBJS := $(patsubst %.c, %.o, $(OBJS))  # `:=` single expend
#OBJS := $(patsubst %.cc, %.o, $(OBJS))   # `=`  recursively expend

TARGET = ./CUDA_Cal

##########################################################

## CC COMPILER OPTIONS ##
# CC compiler options:

CXX      = g++
# CXXFLAG  = -O3 -Wall -g $(INCDIR) $(LIBDIR) $(LIBS)
# CXXFLAGS = -O3 -Wall -g -I $(LASPACK_ROOT)/include -I $(INC_DIR)

# CXXFLAG  = -O3 -Wall -g $(INCDIR) $(LIBDIR) $(LIBS)				# have -g
# CXXFLAGS = -O3 -Wall -g -I $(LASPACK_ROOT)/include -I $(INC_DIR)	# have -g
CXXFLAG  = -O3 -Wall $(INCDIR) $(LIBDIR) $(LIBS)					#no -g
CXXFLAGS = -O3 -Wall -I $(LASPACK_ROOT)/include -I $(INC_DIR)		#no -g

# CXXFLAGS = -O3 -Wall -g -I $(LASPACK_ROOT)/include -I $(LEMON_ROOT)/include $(DEBUG_TAG) 
# CXXFLAGS = -O3 -Wall -g -I $(LASPACK_ROOT)/include -I $(LEDA_ROOT)/incl $(DEBUG_TAG)

##########################################################

## NVCC COMPILER OPTIONS ##
# NVCC compiler options:

NVCC       = nvcc
NVCC_FLAGS = -I $(INC_DIR) #-I $(LASPACK_ROOT)/include
#NVCC_LIBS  = 


##########################################################

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(CUDA_OBJS)  $(CXXFLAG) -lpthread -lrt -std=c++17 -lstdc++fs $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# this use suffix rule $@ $< and pattern rule %
# $@ is target
# $< is first prerequest
# $? is all prerequest, exclude the file without modify
# $^ is all prerequest

# %.o : %.c
# 	$(CXX) $(CXXFLAGS) $(DEFLEFINC) $(FLUTE_INC) -c $< -o $@ -lpthread -lrt

$(DEG_DIR)%.o: $(SRC_DIR)%.cpp
	$(CXX) -std=c++17 -lstdc++fs $(CXXFLAGS) -c $< -o $@ -lpthread -lrt

$(DEG_DIR)%.o: $(SRC_DIR)%.cu
	$(NVCC) -c $< -o $@ $(NVCC_FLAGS) $(NVCC_LIBS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) 
#$(INC_DIR)%.cuh $(INC_DIR)%.h
# %.o: %.cc
# 	$(CXX) $(CXXFLAGS) $(DEFLEFINC) $(FLUTE_INC) -c $< -o $@ -lpthread -lrt
run: $(TARGET)
	time (nvprof --log-file "CUDA_Runtime.txt" ./$(TARGET))
	
clean:
	rm -f $(TARGET) $(OBJS) *.o core.*
