#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "cerebellum.hpp"

namespace py = pybind11;

PYBIND11_MODULE(garrido_brain_cpp, m) {
    m.doc() = "C++ port of garrido_brain_v00.py cerebellum model";

    m.def("angle_diff", &angle_diff, py::arg("a"), py::arg("b"),
          "Smallest signed difference between angle vectors a and b, in [-pi, pi]");

    py::class_<Cerebellum>(m, "Cerebellum")
        .def(py::init<int>(), py::arg("n_dof") = 2)
        .def("granuleLayer",  &Cerebellum::granuleLayer)
        .def("updatePF_PC",   &Cerebellum::updatePF_PC)
        .def("purkinjeCompute", &Cerebellum::purkinjeCompute)
        .def("getPC",         &Cerebellum::getPC)
        .def("getDCN",        &Cerebellum::getDCN)
        .def("updateMF_DCN",  &Cerebellum::updateMF_DCN)
        .def("getMF_DCN",     &Cerebellum::getMF_DCN)
        .def("getPC_DCN",     &Cerebellum::getPC_DCN)
        .def("getPF_PC",      &Cerebellum::getPF_PC)
        .def("updatePC_DCN",  &Cerebellum::updatePC_DCN)
        .def("DCNCompute",    &Cerebellum::DCNCompute)
        .def("dcnToTorque",   &Cerebellum::dcnToTorque)
        .def("getnPFs",        &Cerebellum::getnPFs)
        .def("compute",       &Cerebellum::compute,
             py::arg("qError"), py::arg("qdError"), py::arg("state"));
}
