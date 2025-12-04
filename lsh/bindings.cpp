// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "modules/CosineHashing.h"
#include "modules/EuclidianHashing.h"
#include "modules/UserInputHandling.h"
#include "modules/lsh_interface.h"

namespace py = pybind11;

// We can also expose a few extra functions if you like
extern int k;
extern int L;
extern bool metric;

PYBIND11_MODULE(lshlib, m) {
    m.doc() = "LSH library (Option B binding around existing C++ code)";

    // High-level interface
    m.def("set_files", &lsh_set_files, py::arg("input_path"), py::arg("query_path"),
          py::arg("output_path"), "Open input, query, and output files.");
    m.def("set_parameters", &lsh_set_parameters, py::arg("k"), py::arg("L"), py::arg("metric"),
          "Set LSH parameters: k, L, metric (0 = Euclidean, 1 = Cosine).");
    m.def("build", &lsh_build, "Build hash tables and hash all input data.");
    m.def("run_all_queries", &lsh_run_all_queries,
          "Run LSH on all queries (results written to the output file).");
    m.def("close_files", &lsh_close_files, "Close all open files.");

    // (Optional) expose some lower-level/debug functions you already have:
    m.def("get_number_of_lines", &get_number_of_lines);
    m.def("get_number_of_queries", &get_number_of_queries);

    // Expose globals (read-only from Python by default)
    m.attr("k") = py::cast(&k);
    m.attr("L") = py::cast(&L);
    m.attr("metric") = py::cast(&metric);
}
