#pragma once
// Port of the small piece of arm_assets_v02.py that garrido_brain_v01.py
// touches (angle_diff) plus restrictAngle, which garrido_brain_v01.py
// defines itself.
//
// NOTE: `angle_diff` is imported by garrido_brain_v01.py but is never
// actually called anywhere in that file -- it's dead code as far as the
// cerebellum port is concerned. It's included here only for parity with
// the original module's public surface, using the pymod-based
// implementation you provided (matches arm_assets_v01.py:angle_diff).
// Since arm_assets_v02.py itself wasn't provided, double check this
// against your real source before relying on it.

#include <cmath>
#include <vector>
#include <cstddef>

// Python's % always returns a result with the same sign as m (m>0 here).
inline double pymod(double x, double m) {
    double r = std::fmod(x, m);
    if (r < 0) r += m;
    return r;
}

// Wrap an angle (radians) into [-pi, pi).
inline double restrictAngle(double angle) {
    return pymod(angle + M_PI, 2.0 * M_PI) - M_PI;
}

inline std::vector<double> restrictAngle(const std::vector<double>& angles) {
    std::vector<double> out(angles.size());
    for (std::size_t i = 0; i < angles.size(); ++i) out[i] = restrictAngle(angles[i]);
    return out;
}

// Shortest signed difference a-b, wrapped into [-pi, pi).
inline double angle_diff(double a, double b) {
    return restrictAngle(a - b);
}
