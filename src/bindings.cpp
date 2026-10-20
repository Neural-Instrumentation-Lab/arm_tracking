#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "neuron_params.hpp"
#include "lif_neuron.hpp"
#include "subcomplexes.hpp"
#include "arm_assets.hpp"
#include "cerebellum.hpp"

namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// copy of a 0/1 byte vector as a numpy bool array
static py::array_t<bool> to_bool_array(const std::vector<uint8_t>& v) {
    py::array_t<bool> out(v.size());
    auto* dst = out.mutable_data();
    for (std::size_t i = 0; i < v.size(); ++i) dst[i] = v[i] != 0;
    return out;
}

// LIFPopulation inputs: None, a scalar (broadcast to n), or an array of length n.
// `storage` keeps the converted data alive for the duration of the call.
static const double* population_input(const py::object& obj, std::size_t n, const char* name,
                                      std::vector<double>& storage) {
    if (obj.is_none()) return nullptr;
    DoubleArray arr = DoubleArray::ensure(obj);
    if (!arr) throw py::type_error(std::string(name) + " must be None, a number or an array");
    if (arr.size() == 1) {
        storage.assign(n, *arr.data());
    } else if (static_cast<std::size_t>(arr.size()) == n) {
        storage.assign(arr.data(), arr.data() + n);
    } else {
        throw py::value_error(std::string(name) + " must have length n=" + std::to_string(n));
    }
    return storage.data();
}

PYBIND11_MODULE(garrido_brain, m) {
    m.doc() = "C++/pybind11 port of garrido_brain_v02.py + lif_neuron_v02.py";

    py::register_exception<BrainError>(m, "BrainError");

    m.attr("LEGACY_DT") = LEGACY_DT;
    m.attr("PF_HISTORY_S") = PF_HISTORY_S;
    m.attr("TORQUE_FILTER_S") = TORQUE_FILTER_S;
    m.attr("CF_BURST_ISI_S") = CF_BURST_ISI_S;

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

    m.attr("GC_params") = GC_params;
    m.attr("PC_params") = PC_params;
    m.attr("DCN_params") = DCN_params;

    m.def("legacy_charge_gain", &legacy_charge_gain, py::arg("tau"), py::arg("dt0") = 2e-3);
    m.def("legacy_tonic_conductance", &legacy_tonic_conductance,
          py::arg("w"), py::arg("tau"), py::arg("dt0") = 2e-3);

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
        .def(py::init<std::size_t, const NeuronParams&, bool>(),
             py::arg("n"), py::arg("params") = NeuronParams(), py::arg("legacy_refractory") = false)
        .def("step", [](LIFPopulation& pop, double dt, const py::object& ampa_input,
                        const py::object& nmda_input, const py::object& gaba_input,
                        const py::object& ampa_tonic) {
                 std::vector<double> a, nm, g, tonic;
                 const double* pa = population_input(ampa_input, pop.n, "ampa_input", a);
                 const double* pn = population_input(nmda_input, pop.n, "nmda_input", nm);
                 const double* pg = population_input(gaba_input, pop.n, "gaba_input", g);
                 const double* pt = population_input(ampa_tonic, pop.n, "ampa_tonic", tonic);
                 return to_bool_array(pop.step(dt, pa, pn, pg, pt));
             },
             py::arg("dt"), py::arg("ampa_input") = py::none(), py::arg("nmda_input") = py::none(),
             py::arg("gaba_input") = py::none(), py::arg("ampa_tonic") = py::none())
        .def("__len__", &LIFPopulation::size)
        .def_readonly("n", &LIFPopulation::n)
        .def_readonly("legacy_refractory", &LIFPopulation::legacy_refractory)
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
        .def(py::init<int, std::optional<uint64_t>>(), py::arg("n_neurons") = 50, py::arg("seed") = py::none())
        .def("step", [](CFsubcomplexALT& c, double dt, double inp_I) {
                 return to_bool_array(c.step(dt, inp_I));
             }, py::arg("dt"), py::arg("inp_I"))
        .def_readonly("n_neurons", &CFsubcomplexALT::n_neurons)
        .def_readonly("centers", &CFsubcomplexALT::centers)
        .def_readonly("widths", &CFsubcomplexALT::widths)
        .def_readonly("spikes_pending", &CFsubcomplexALT::spikes_pending)
        .def_readonly("wait", &CFsubcomplexALT::wait)
        .def_readwrite("draw_override", &CFsubcomplexALT::draw_override);

    // ---------------- MFsubcomplex ----------------
    py::class_<MFsubcomplex>(m, "MFsubcomplex")
        .def(py::init<double, double, int>(), py::arg("min"), py::arg("max"), py::arg("n_neurons") = 10)
        .def("step", &MFsubcomplex::step, py::arg("inp"), py::arg("dt"))
        .def_readonly("centers", &MFsubcomplex::centers)
        .def_readonly("widths", &MFsubcomplex::widths)
        .def_readonly("spikes", &MFsubcomplex::spikes);

    // ---------------- PFSpikeHistory ----------------
    py::class_<PFSpikeHistory>(m, "PFSpikeHistory")
        .def(py::init<int64_t>(), py::arg("prune_window") = PFSpikeHistory::HISTORY_PRUNE_WINDOW)
        .def("record", &PFSpikeHistory::record, py::arg("indices"), py::arg("t"))
        .def("prune", &PFSpikeHistory::prune, py::arg("current_t"))
        .def("flatten", [](const PFSpikeHistory& h) {
            std::vector<int64_t> idx, times;
            h.flatten(idx, times);
            return py::make_tuple(py::array(py::cast(idx)), py::array(py::cast(times)));
        })
        .def_readonly("prune_window", &PFSpikeHistory::prune_window)
        .def("__len__", &PFSpikeHistory::size);

    // ---------------- free functions ----------------
    m.def("ltd_kernel", &ltd_kernel, py::arg("x"), py::arg("dk") = DK, py::arg("tau_ltd") = TAU_LTD);
    m.def("restrictAngle", py::overload_cast<double>(&restrictAngle), py::arg("angle"));
    m.def("restrictAngle", py::overload_cast<const std::vector<double>&>(&restrictAngle), py::arg("angle"));
    m.def("angle_diff", &angle_diff, py::arg("a"), py::arg("b"));

    // ---------------- cerebellum ----------------
    py::class_<cerebellum>(m, "cerebellum")
        .def(py::init<std::vector<double>, std::vector<double>, std::size_t, double, bool, bool,
                      std::optional<uint64_t>>(),
             py::arg("qMins"), py::arg("qMaxs"), py::arg("n_dof") = 6, py::arg("dt") = LEGACY_DT,
             py::arg("keep_legacy_drive") = true, py::arg("keep_legacy_refractory") = true,
             py::arg("seed") = py::none())
        .def("compute", [](cerebellum& c, const std::vector<double>& q, const std::vector<double>& qdes,
                           const std::vector<double>& qErr, std::optional<double> dt) {
                 auto corr = c.compute(q, qdes, qErr, dt);
                 return py::array(py::cast(corr));
             },
             py::arg("q"), py::arg("qdes"), py::arg("qErr"), py::arg("dt") = py::none())
        .def("set_cf_draw_override", &cerebellum::set_cf_draw_override, py::arg("value"),
             "Testing only: fix every climbing-fiber random draw to `value` (None restores the RNG).")
        .def_readonly_static("ALPHA", &cerebellum::ALPHA)
        .def_readonly_static("BETA", &cerebellum::BETA)
        .def_readonly_static("INIT_PF_PC_WT", &cerebellum::INIT_PF_PC_WT)
        .def_readonly_static("W_MIN", &cerebellum::W_MIN)
        .def_readonly_static("W_MAX", &cerebellum::W_MAX)
        .def_readonly_static("nMF_subgroups", &cerebellum::nMF_subgroups)
        .def_readonly_static("nMF_per_subgroup", &cerebellum::nMF_per_subgroup)
        .def_readonly_static("nMF", &cerebellum::nMF)
        .def_readonly_static("nGC", &cerebellum::nGC)
        .def_readonly_static("nCF", &cerebellum::nCF)
        .def_readonly_static("nPC", &cerebellum::nPC)
        .def_readonly_static("nDCN", &cerebellum::nDCN)
        .def_readonly_static("W_MF_GC", &cerebellum::W_MF_GC)
        .def_readonly_static("W_PC_DCN", &cerebellum::W_PC_DCN)
        .def_readonly_static("W_CF_DCN_AMPA", &cerebellum::W_CF_DCN_AMPA)
        .def_readonly_static("W_CF_DCN_NMDA", &cerebellum::W_CF_DCN_NMDA)
        .def_readonly("W_MF_DCN", &cerebellum::W_MF_DCN)
        .def_readwrite("TORQUE_ALPHA", &cerebellum::TORQUE_ALPHA)
        .def_readonly("nJoints", &cerebellum::nJoints)
        .def_readonly("dt", &cerebellum::dt)
        .def_readonly("timeStep", &cerebellum::dt)
        .def_readonly("t", &cerebellum::t)
        .def_readonly("keep_legacy_drive", &cerebellum::keep_legacy_drive)
        .def_readonly("keep_legacy_refractory", &cerebellum::keep_legacy_refractory)
        .def_readonly("qmins", &cerebellum::qmins)
        .def_readonly("qmaxs", &cerebellum::qmaxs)
        .def_readonly("gc_tonic_value", &cerebellum::gc_tonic_value)
        .def_readonly("mf_dcn_value", &cerebellum::mf_dcn_value)
        .def_readonly("gain_pc_ampa", &cerebellum::gain_pc_ampa)
        .def_readonly("gain_dcn_ampa", &cerebellum::gain_dcn_ampa)
        .def_readonly("gain_dcn_nmda", &cerebellum::gain_dcn_nmda)
        .def_readonly("gain_dcn_gaba", &cerebellum::gain_dcn_gaba)
        .def_readonly("nTaps", &cerebellum::nTaps)
        .def_readonly("kernelLookup", &cerebellum::kernelLookup)
        .def_readonly("gcNeurons", &cerebellum::gcNeurons)
        .def_readonly("pcNeurons", &cerebellum::pcNeurons)
        .def_readonly("dcnNeurons", &cerebellum::dcnNeurons)
        .def_readonly("pf_history", &cerebellum::pf_history)
        .def_property_readonly("pfs", [](const cerebellum& c) { return to_bool_array(c.pfs); })
        .def_property_readonly("cf_output", [](const cerebellum& c) { return to_bool_array(c.cf_output); })
        .def_property_readonly("pc_out", [](const cerebellum& c) { return to_bool_array(c.pc_out); })
        .def_property_readonly("dcnOut", [](const cerebellum& c) { return to_bool_array(c.dcnOut); })
        .def_property_readonly("prevDCN", [](const cerebellum& c) {
            DoubleArray out({c.nTaps, c.nJoints});
            std::copy(c.prevDCN.begin(), c.prevDCN.end(), out.mutable_data());
            return out;
        })
        .def_property_readonly("pf_pc_wts", [](py::object self) {
            // writable 2D view onto the weight matrix; keeps the brain alive while it exists
            auto& c = self.cast<cerebellum&>();
            const std::size_t rows = c.pf_pc_wts.size() / c.n_pc_total;
            return py::array_t<double>({rows, c.n_pc_total}, c.pf_pc_wts.data(), self);
        });
}
