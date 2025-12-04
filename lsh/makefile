CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC

# Get Python and pybind11 compile flags
PYTHON_INC := $(shell python3 -m pybind11 --includes)
PYTHON_EXT := $(shell python3-config --extension-suffix)

MODULES = \
    modules/UserInputHandling.cpp \
    modules/EuclidianHashing.cpp \
    modules/modulo_func.cpp \
    modules/randomness.cpp \
    modules/CosineHashing.cpp \
    modules/globals.cpp \
    modules/lsh_interface.cpp \
    bindings.cpp

all: lshlib$(PYTHON_EXT)

lshlib$(PYTHON_EXT): $(MODULES)
	$(CXX) $(CXXFLAGS) $(MODULES) $(PYTHON_INC) -o $@

clean:
	rm -f lshlib*.so
