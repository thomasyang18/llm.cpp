CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
INCLUDES = -I./include -I/usr/local/include/eigen3

# Libraries
CXXLIB = -lcnpy -lz

SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Recursively find all .cpp files in $(SRCDIR)
SOURCES := $(shell find $(SRCDIR) -type f -name '*.cpp')

# Convert sources to corresponding object files in $(OBJDIR)
# This preserves the directory structure, e.g. src/foo/bar.cpp becomes obj/foo/bar.o
OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

# If main.o exists, force it to the front of the object list.
OBJECTS := $(if $(filter $(OBJDIR)/main.o,$(OBJECTS)), \
             $(OBJDIR)/main.o $(filter-out $(OBJDIR)/main.o,$(OBJECTS)), \
             $(OBJECTS))

# Main target
TARGET = $(BINDIR)/gpt2_weight_loader

# Default build type
BUILD_TYPE ?= release
ifeq ($(BUILD_TYPE), debug)
    CXXFLAGS += -O0 -DDEBUG -g 
else
    CXXFLAGS += -O3 -DRELEASE -DEIGEN_USE_BLAS -mavx -mfma
	CXXLIB += -lopenblas
endif

# Ensure top-level directories exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(CXXLIB)

# Rule to compile .cpp files into .o files.
# The command first ensures that the target directory exists.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Dependency tracking: generate .d files alongside .o files.
$(OBJDIR)/%.d: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $< -o $@

# Include all dependency files (if they exist)
-include $(OBJECTS:.o=.d)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
