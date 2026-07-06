/*
 * pin_ext.cpp
 *
 * pybind11 module that exposes a tight C++ substep integration loop using
 * pinocchio's ABA (Articulated Body Algorithm).  Python still owns the outer
 * per-timestep loop and the cerebellum; this module eliminates the Python
 * dispatch overhead for the n_substeps=100 inner loop inside forwardDynamics
 * and inverseDynamics.
 *
 * Exported function
 * -----------------
 * euler_substeps(model, data, q, v, tau, h, n_substeps)
 *   -> (q_new, v_new, a_last)
 *
 *   Runs `n_substeps` of semi-implicit Euler integration:
 *     a  = ABA(model, data, q, v, tau)
 *     v += a * h
 *     q  = integrate(model, q, v * h)
 *   and returns the final (q, v, a).  tau is held constant (ZOH) throughout.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <tuple>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace py = pybind11;
using Vec = Eigen::VectorXd;

/* -----------------------------------------------------------------------
 * euler_substeps
 * ----------------------------------------------------------------------- */
static std::tuple<Vec, Vec, Vec>
euler_substeps(
    pinocchio::Model& model,
    pinocchio::Data&  data,
    Vec  q,            // copied — modified in-place each substep
    Vec  v,            // copied
    const Vec& tau,    // held constant (ZOH)
    double h,          // substep dt
    int    n_substeps)
{
    Vec a(model.nv);
    for (int i = 0; i < n_substeps; ++i) {
        a = pinocchio::aba(model, data, q, v, tau);
        v.noalias() += a * h;
        // integrate q on the Lie group: q_next = q ⊕ (v_new * h)
        q = pinocchio::integrate(model, q, (v * h).eval());
    }
    return {std::move(q), std::move(v), std::move(a)};
}

/* -----------------------------------------------------------------------
 * Module definition
 * ----------------------------------------------------------------------- */
PYBIND11_MODULE(pin_ext, m) {
    m.doc() =
        "C++ acceleration for pinocchio substep integration loops.\n"
        "Import pinocchio before this module so its types are registered.";

    m.def(
        "euler_substeps", &euler_substeps,
        py::arg("model"), py::arg("data"),
        py::arg("q"), py::arg("v"), py::arg("tau"),
        py::arg("h"), py::arg("n_substeps"),
        R"(Run n_substeps of semi-implicit Euler integration using ABA.

Parameters
----------
model, data : pinocchio.Model, pinocchio.Data
q, v, tau   : joint config / velocity / torque (numpy arrays)
h           : substep dt (outer_dt / n_substeps)
n_substeps  : number of inner Euler steps

Returns
-------
(q_new, v_new, a_last) : numpy arrays after all substeps
)");
}
