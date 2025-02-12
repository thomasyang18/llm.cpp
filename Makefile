CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
INCLUDES = -I./include -I/usr/local/include/eigen3

# You might need to adjust these paths for your system
CNPY_LIB = -lcnpy
ZLIB = -lz

SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp) main.cpp
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

# Main target
TARGET = $(BINDIR)/gpt2_weight_loader

# Create directories
$(shell mkdir -p $(OBJDIR)/src $(BINDIR))

# Default build type
BUILD_TYPE ?= release

# Optimization level
OPTIMIZATION ?= 3

ifeq ($(BUILD_TYPE), debug)
    CXXFLAGS += -O0 -DDEBUG -g
else
    CXXFLAGS += -O$(OPTIMIZATION) -DRELEASE
endif

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(CNPY_LIB) $(ZLIB)

# Compile main.cpp
$(OBJDIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile source files
$(OBJDIR)/src/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJDIR)/* $(TARGET)

.PHONY: all clean
