CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC

# Get Python + pybind11 include flags
PYTHON_INC := $(shell python3 -m pybind11 --includes)
OUTPUT = lshlib.so

MODULES = \
    modules/UserInputHandling.cpp \
    modules/EuclidianHashing.cpp \
    modules/modulo_func.cpp \
    modules/randomness.cpp \
    modules/CosineHashing.cpp \
    modules/globals.cpp \
    modules/lsh_interface.cpp \
    bindings.cpp

all: $(OUTPUT)

$(OUTPUT): $(MODULES)
	$(CXX) $(CXXFLAGS) $(MODULES) $(PYTHON_INC) -o $(OUTPUT)

clean:
	rm -f lshlib*.so
