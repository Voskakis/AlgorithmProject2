#include <pybind11/pybind11.h>

#include "myheader.h"

namespace py = pybind11;

PYBIND11_MODULE(add_module, m) {
    py::class_<AddClass>(m, "AddClass").def(py::init<>()).def("add", &AddClass::add);
}