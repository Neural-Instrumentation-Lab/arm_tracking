#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "neuron_params.hpp"
#include "lif_neuron.hpp"
#include "subcomplexes.hpp"
#include "arm_assets.hpp"
#include "cerebellum.hpp"

namespace py = pybind11;

PYBIND11_MODULE(garrido_brain, m) {
    m.doc() = "C++/pybind11 port of garrido_brain_v01.py + lif_neuron_v01.py";

    py::register_exception<BrainError>(m, "BrainError");

    // ---------------- NeuronParams ----------------
    py::class_<NeuronParams>(m, "NeuronParams")
        .def(py::init([](double Cm, double gL, double EL, double E_AMPA, double E_GABA,
                          double V_thr, double T_ref, double tau_AMPA, double tau_NMDA, double tau_GABA) {
                 NeuronParams p;
                 p.Cm = Cm; p.gL = gL; p.EL = EL; p.E_AMPA = E_AMPA; p.E_GABA = E_GABA;
                 p.V_thr = V_thr; p.T_ref = T_ref; p.tau_AMPA = tau_AMPA; p.tau_NMDA = tau_NMDA; p.tau_GABA = tau_GABA;
                 return p;
             }),
             py::arg("Cm") = 1.0e-9, py::arg("gL") = 0.1e-6, py::arg("EL") = -70e-3,
             py::arg("E_AMPA") = 0.0, py::arg("E_GABA") = -80e-3, py::arg("V_thr") = -50e-3,
             py::arg("T_ref") = 1e-3, py::arg("tau_AMPA") = 1.0e-3, py::arg("tau_NMDA") = 20e-3,
             py::arg("tau_GABA") = 5.0e-3)
        .def_readwrite("Cm", &NeuronParams::Cm)
        .def_readwrite("gL", &NeuronParams::gL)
        .def_readwrite("EL", &NeuronParams::EL)
        .def_readwrite("E_AMPA", &NeuronParams::E_AMPA)
        .def_readwrite("E_GABA", &NeuronParams::E_GABA)
        .def_readwrite("V_thr", &NeuronParams::V_thr)
        .def_readwrite("T_ref", &NeuronParams::T_ref)
        .def_readwrite("tau_AMPA", &NeuronParams::tau_AMPA)
        .def_readwrite("tau_NMDA", &NeuronParams::tau_NMDA)
        .def_readwrite("tau_GABA", &NeuronParams::tau_GABA);

    // ---------------- LIFNeuron (scalar) ----------------
    py::class_<LIFNeuron>(m, "LIFNeuron")
        .def(py::init<const NeuronParams&>(), py::arg("params") = NeuronParams())
        .def("step", &LIFNeuron::step, py::arg("dt") = 2.0e-3,
             py::arg("ampa_weights") = std::vector<double>{},
             py::arg("nmda_weights") = std::vector<double>{},
             py::arg("gaba_weights") = std::vector<double>{})
        .def_readwrite("V", &LIFNeuron::V)
        .def_readwrite("g_AMPA", &LIFNeuron::g_AMPA)
        .def_readwrite("g_NMDA", &LIFNeuron::g_NMDA)
        .def_readwrite("g_GABA", &LIFNeuron::g_GABA)
        .def_readwrite("refact_t", &LIFNeuron::refact_t)
        .def_readonly("spiked", &LIFNeuron::spiked)
        .def_readonly("spike_times", &LIFNeuron::spike_times)
        .def_readonly("t", &LIFNeuron::t);

    // ---------------- LIFPopulation ----------------
    py::class_<LIFPopulation>(m, "LIFPopulation")
        .def(py::init<std::size_t, const NeuronParams&>(), py::arg("n"), py::arg("params") = NeuronParams())
        .def("step", &LIFPopulation::step, py::arg("dt"),
             py::arg("ampa_input") = py::none(), py::arg("nmda_input") = py::none(), py::arg("gaba_input") = py::none())
        .def("__len__", &LIFPopulation::size)
        .def_readonly("n", &LIFPopulation::n)
        .def_readwrite("V", &LIFPopulation::V)
        .def_readwrite("g_AMPA", &LIFPopulation::g_AMPA)
        .def_readwrite("g_NMDA", &LIFPopulation::g_NMDA)
        .def_readwrite("g_GABA", &LIFPopulation::g_GABA)
        .def_readwrite("refractory_time_left", &LIFPopulation::refractory_time_left)
        .def_readonly("spike_counts", &LIFPopulation::spike_counts)
        .def_readonly("t", &LIFPopulation::t);

    // ---------------- CFsubcomplex ----------------
    py::class_<CFsubcomplex>(m, "CFsubcomplex")
        .def(py::init<int>(), py::arg("n_neurons") = 50)
        .def_static("burst_size", &CFsubcomplex::burst_size)
        .def("step", &CFsubcomplex::step, py::arg("dt"), py::arg("inp_I"))
        .def_readonly("n_neurons", &CFsubcomplex::n_neurons)
        .def_readonly("spikes_pending", &CFsubcomplex::spikes_pending)
        .def_readonly("last_spk_time", &CFsubcomplex::last_spk_time)
        .def_readonly("t", &CFsubcomplex::t);

    // ---------------- CFsubcomplexALT ----------------
    py::class_<CFsubcomplexALT>(m, "CFsubcomplexALT")
        .def(py::init<int>(), py::arg("n_neurons") = 50)
        .def("step", &CFsubcomplexALT::step, py::arg("dt"), py::arg("inp_I"))
        .def_readonly("n_neurons", &CFsubcomplexALT::n_neurons)
        .def_readonly("centers", &CFsubcomplexALT::centers)
        .def_readonly("widths", &CFsubcomplexALT::widths)
        .def_readonly("spikes_pending", &CFsubcomplexALT::spikes_pending);

    // ---------------- MFsubcomplex ----------------
    py::class_<MFsubcomplex>(m, "MFsubcomplex")
        .def(py::init<double, double, int>(), py::arg("min"), py::arg("max"), py::arg("n_neurons") = 10)
        .def("step", &MFsubcomplex::step, py::arg("inp"), py::arg("dt"))
        .def_readonly("centers", &MFsubcomplex::centers)
        .def_readonly("widths", &MFsubcomplex::widths)
        .def_readonly("spikes", &MFsubcomplex::spikes);

    // ---------------- PFSpikeHistory ----------------
    py::class_<PFSpikeHistory>(m, "PFSpikeHistory")
        .def(py::init<double>(), py::arg("prune_window") = PFSpikeHistory::HISTORY_PRUNE_WINDOW)
        .def("record", &PFSpikeHistory::record, py::arg("indices"), py::arg("t"))
        .def("prune", &PFSpikeHistory::prune, py::arg("current_t"))
        .def("flatten", [](const PFSpikeHistory& h) {
            std::vector<int64_t> idx;
            std::vector<double> times;
            h.flatten(idx, times);
            return py::make_tuple(idx, times);
        })
        .def("__len__", &PFSpikeHistory::size);

    // ---------------- free functions ----------------
    m.def("ltd_kernel", &ltd_kernel, py::arg("x"), py::arg("dk") = DK, py::arg("tau_ltd") = TAU_LTD);
    m.def("restrictAngle", py::overload_cast<double>(&restrictAngle), py::arg("angle"));
    m.def("restrictAngle", py::overload_cast<const std::vector<double>&>(&restrictAngle), py::arg("angle"));
    m.def("angle_diff", &angle_diff, py::arg("a"), py::arg("b"));

    // ---------------- cerebellum ----------------
    py::class_<cerebellum>(m, "cerebellum")
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::size_t>(),
             py::arg("qMins"), py::arg("qdMins"), py::arg("qMaxs"), py::arg("qdMaxs"), py::arg("n_dof") = 6)
        .def("compute", &cerebellum::compute,
             py::arg("q"), py::arg("qd"), py::arg("qdes"), py::arg("qddes"),
             py::arg("qErr"), py::arg("qdErr"), py::arg("dt") = 2e-3)
        .def_readonly("nJoints", &cerebellum::nJoints)
        .def_readonly("t", &cerebellum::t)
        .def_readonly("pfs", &cerebellum::pfs)
        .def_readonly("cf_output", &cerebellum::cf_output)
        .def_readonly("pc_out", &cerebellum::pc_out)
        .def_property_readonly("pf_pc_wts", [](const cerebellum& c) {
            // Expose the (large) flat weight matrix as a 2D numpy array view.
            const std::size_t rows = c.pf_pc_wts.size() / c.n_pc_total;
            return py::array_t<double>({rows, c.n_pc_total}, c.pf_pc_wts.data());
        });
}
