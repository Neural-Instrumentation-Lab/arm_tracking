/*
 * Cerebellar motor learning module — C++ implementation exposed via pybind11.
 *
 * Mirrors the Python Cerebellum class in model_v01.py exactly:
 *   - same grid of Gaussian basis cells tiling motor-angle space
 *   - same online gradient-descent weight update
 *
 * Build:
 *   python setup.py build_ext --inplace
 *
 * Usage from Python:
 *   from cerebellum import Cerebellum
 *   C = Cerebellum()
 *   c = C.compute(m)   # m is a numpy array of length 2
 *   C.update(err)      # err is a numpy array of length 2
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

class Cerebellum {
public:
    Cerebellum() {
        const double pi = M_PI;
        const double spacing = 0.25 * pi;
        beta_     = 0.05;
        double sigma = std::sqrt(spacing);
        inv_2s2_  = 1.0 / (2.0 * sigma * sigma);

        // Replicate Python:
        //   c1_vals = np.arange(-0.25*pi, 1.25*pi, spacing)
        //   c2_vals = np.arange(0,        pi,       spacing)
        // np.arange(stop=X) excludes X — use -epsilon so floating-point
        // accumulation never sneaks in an extra step at the boundary.
        std::vector<double> c1_vals, c2_vals;
        for (double v = -0.25 * pi; v < 1.25 * pi - 1e-9; v += spacing)
            c1_vals.push_back(v);
        for (double v = 0.0; v < pi - 1e-9; v += spacing)
            c2_vals.push_back(v);

        n_cells_ = static_cast<int>(c1_vals.size() * c2_vals.size());

        // Flatten cell centres into ctrs_[i*2 + 0/1].
        // Outer loop c2, inner loop c1 — matches the Python list comprehension:
        //   [Cell(c1, c2, sigma) for c2 in c2_vals for c1 in c1_vals]
        ctrs_.resize(n_cells_ * 2);
        int idx = 0;
        for (double c2 : c2_vals)
            for (double c1 : c1_vals) {
                ctrs_[idx * 2 + 0] = c1;
                ctrs_[idx * 2 + 1] = c2;
                ++idx;
            }

        wts_.assign(n_cells_ * 2, 0.0);   // shape (n_cells, 2), zero-initialised
        p_.resize(n_cells_);
    }

    /*
     * compute(m) — evaluate Gaussian activations for motor state m, then
     * return the weighted sum as a length-2 numpy array.
     *
     * Mirrors:
     *   for i, cell in enumerate(self.cells): self.p[i] = cell.eval(m)
     *   return np.sum(self.wts * self.p[:, np.newaxis], axis=0)
     *
     * Uses request() rather than unchecked<1>() so the array can be any
     * contiguous shape (e.g. (2,) or (1,2)) as long as it holds 2 doubles.
     */
    py::array_t<double> compute(py::array_t<double, py::array::c_style | py::array::forcecast> m_arr) {
        double* m = static_cast<double*>(m_arr.request().ptr);

        for (int i = 0; i < n_cells_; ++i) {
            double d0 = m[0] - ctrs_[i * 2 + 0];
            double d1 = m[1] - ctrs_[i * 2 + 1];
            p_[i] = std::exp(-(d0 * d0 + d1 * d1) * inv_2s2_);
        }

        double c0 = 0.0, c1 = 0.0;
        for (int i = 0; i < n_cells_; ++i) {
            c0 += wts_[i * 2 + 0] * p_[i];
            c1 += wts_[i * 2 + 1] * p_[i];
        }

        // Build a fresh numpy array to return — Python owns the memory.
        auto result = py::array_t<double>(2);
        auto buf    = result.mutable_unchecked<1>();
        buf(0) = c0;
        buf(1) = c1;
        return result;
    }

    /*
     * update(err) — gradient-descent weight update.
     *
     * Mirrors:
     *   self.wts -= self.beta * self.p[:, np.newaxis] * err
     *
     * Same forcecast as compute(): accepts (2,) or (1,2) shaped arrays.
     */
    void update(py::array_t<double, py::array::c_style | py::array::forcecast> err_arr) {
        double* err = static_cast<double*>(err_arr.request().ptr);
        for (int i = 0; i < n_cells_; ++i) {
            wts_[i * 2 + 0] -= beta_ * p_[i] * err[0];
            wts_[i * 2 + 1] -= beta_ * p_[i] * err[1];
        }
    }

private:
    int    n_cells_;
    double beta_;
    double inv_2s2_;
    std::vector<double> ctrs_;   // (n_cells, 2) flattened
    std::vector<double> wts_;    // (n_cells, 2) flattened
    std::vector<double> p_;      // (n_cells,) — last activation, kept for update()
};


PYBIND11_MODULE(cerebellum, m) {
    m.doc() = "Cerebellar motor learning — C++ implementation via pybind11";

    py::class_<Cerebellum>(m, "Cerebellum")
        .def(py::init<>(),
             "Construct the cerebellar module (replicates model_v01 Cerebellum())")
        .def("compute", &Cerebellum::compute,
             py::arg("m"),
             "Return the correction vector c (length-2 array) for motor state m")
        .def("update", &Cerebellum::update,
             py::arg("err"),
             "Update weights given endpoint error err (length-2 array)");
}
