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
SOURCES = $(notdir $(wildcard $(SRCDIR)/*.cpp))

# $(info $(SOURCES))

# Jesus christ, makefiels are so ass. 
# Manually force main.o to be up top jfc.
OBJECTS = $(OBJDIR)/main.o $(sort $(filter-out $(OBJDIR)/main.o,$(SOURCES:%.cpp=$(OBJDIR)/%.o)))
# Main target
TARGET = $(BINDIR)/gpt2_weight_loader

# Create directories
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Default build type
BUILD_TYPE ?= release

ifeq ($(BUILD_TYPE), debug)
    CXXFLAGS += -O0 -DDEBUG -g
else
    CXXFLAGS += -O3 -DRELEASE
endif

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(CNPY_LIB) $(ZLIB)


# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Dependency tracking
$(OBJDIR)/%.d: $(SRCDIR)/%.cpp
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $< > $@

-include $(OBJDIR)/*.d

clean:
	rm -rf $(OBJDIR)/* $(TARGET)

.PHONY: all clean
