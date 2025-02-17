# Compilers
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++17 -Wall -Wextra
NVCCFLAGS = -std=c++17 

# Include directories (add CUDA include)
INCLUDES = -I./include -I/usr/local/include/eigen3 -I/usr/local/cuda/include

# Libraries
CXXLIB = -lcnpy -lz
# For release builds, linking with OpenBLAS might be needed:
ifeq ($(BUILD_TYPE), debug)
    CXXFLAGS += -O0 -DDEBUG -g 
	NVCCFLAGS += -O0 -DDEBUG -g 
else
    CXXFLAGS += -O3 -DRELEASE -DEIGEN_USE_BLAS -mavx -mfma
	NVCCFLAGS += -O3 -DRELEASE 
    CXXLIB += -lopenblas
endif


CUDA_LIB_PATH = /usr/local/cuda/lib64  # Change if using a different CUDA install

# CUDA runtime library (nvcc usually links this automatically, but include if needed)
NVCCLIB = -L$(CUDA_LIB_PATH) -lcudart

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Find all C++ sources (.cpp) and CUDA sources (.cu)
CPP_SOURCES := $(shell find $(SRCDIR) -type f -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -type f -name '*.cu')

# Convert sources to object files (preserving directory structure)
CPP_OBJECTS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPP_SOURCES))
CU_OBJECTS := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.cu.o, $(CU_SOURCES))

# Place main.o at the beginning if it exists
CPP_OBJECTS := $(if $(filter $(OBJDIR)/main.o, $(CPP_OBJECTS)), \
             $(OBJDIR)/main.o $(filter-out $(OBJDIR)/main.o, $(CPP_OBJECTS)), \
             $(CPP_OBJECTS))

# Final target: link both C++ and CUDA object files
TARGET = $(BINDIR)/gpt2

# Ensure necessary directories exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CXXFLAGS) $(CPP_OBJECTS) $(CU_OBJECTS) -o $@ $(CXXLIB) $(NVCCLIB)

# Rule to compile C++ (.cpp) files into .o files.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule to compile CUDA (.cu) files into .o files using nvcc.
$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Dependency tracking for C++ sources (if needed)
#$(OBJDIR)/%.d: $(SRCDIR)/%.cpp
#	@mkdir -p $(dir $@)
#	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $< -o $@

# Include dependency files (if they exist)
-include $(CPP_OBJECTS:.o=.d)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
