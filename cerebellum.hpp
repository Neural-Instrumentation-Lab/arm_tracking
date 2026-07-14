#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

// ---- angle_diff helpers (matches arm_assets_v01.py:angle_diff) ----
inline double pymod(double x, double m) {
    // Python's % always returns a result with the same sign as m (m>0 here).
    double r = std::fmod(x, m);
    if (r < 0) r += m;
    return r;
}

inline double angle_diff_scalar(double a, double b) {
    return pymod(a - b + M_PI, 2.0 * M_PI) - M_PI;
}

inline Eigen::VectorXd angle_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    Eigen::VectorXd out(a.size());
    for (int i = 0; i < a.size(); ++i) out[i] = angle_diff_scalar(a[i], b[i]);
    return out;
}

// ---- cerebellum model (matches garrido_brain_v00.py:cerebellum) ----
class Cerebellum {
public:
    explicit Cerebellum(int n_dof = 2)
        : currPF(0), nPFs(500), pfIdx(0), nMuscles(n_dof * 2), active_pf_pc(true), active_pc_dcn(true), active_mf_dcn(true){
        purAct          = Eigen::VectorXd::Zero(nMuscles);
        pf_pc_weights   = Eigen::MatrixXd::Zero(nPFs, nMuscles);
        mf_dcn_weights  = Eigen::VectorXd::Zero(nMuscles);
        pc_dcn_weights  = Eigen::VectorXd::Zero(nMuscles);
        dcnAct          = Eigen::VectorXd::Zero(nMuscles);
    }

    void granuleLayer(int state) {
        pfIdx  = ((state - 1) % nPFs + nPFs) % nPFs;
        currPF = (state % nPFs + nPFs) % nPFs;
    }

    void updatePF_PC(const Eigen::VectorXd& error) {
        // row view of the current PF's weights
        Eigen::VectorXd row = pf_pc_weights.row(pfIdx);
        Eigen::ArrayXd errArr = error.array();
        Eigen::ArrayXd update = (LTP_max / (errArr + 1.0).pow(alpha)) - LTD_max * errArr;
        row = row.array() + update;
        row = row.array().max(0.0).min(1.0);
        pf_pc_weights.row(pfIdx) = row;
    }

    void purkinjeCompute() {
        purAct = pf_pc_weights.row(currPF).transpose();
        purAct = purAct.array().max(0.0).min(1.0);
    }

    Eigen::VectorXd getPC() const  { return purAct; }
    Eigen::VectorXd getDCN() const { return dcnAct; }
    int getnPFs() const { return nPFs; }
    void setActiveSites(const bool pf_pc, const bool mf_dcn, const bool pc_dcn) {active_mf_dcn = mf_dcn; active_pf_pc = pf_pc; active_pc_dcn = pc_dcn;}

    void updateMF_DCN() {
        Eigen::ArrayXd pa = purAct.array();
        Eigen::ArrayXd update = (LTP_max_dcn / (pa + 1.0).pow(alpha)) - LTD_max_dcn * pa;
        mf_dcn_weights = (mf_dcn_weights.array() + update).max(0.0);
    }

    Eigen::VectorXd getMF_DCN() const { return mf_dcn_weights; }
    Eigen::VectorXd getPC_DCN() const { return pc_dcn_weights; }
    Eigen::MatrixXd getPF_PC() const  { return pf_pc_weights; }

    void updatePC_DCN() {
        Eigen::ArrayXd dcn_clipped = dcnAct.array().max(0.0).min(1.0);
        Eigen::ArrayXd pa = purAct.array();
        Eigen::ArrayXd update =
            (LTP_max_dcn * pa.pow(alpha)) * (1.0 - (dcn_clipped + 1.0).pow(alpha).inverse())
            - LTD_max_dcn * (1.0 - pa);
        pc_dcn_weights = (pc_dcn_weights.array() + update).max(0.0);
    }

    void DCNCompute() {
        dcnAct = (mf_dcn_weights.array() - purAct.array() * pc_dcn_weights.array()).max(0.0);
    }

    Eigen::VectorXd dcnToTorque() const {
        // corr[1::2] *= -1 ; then sum consecutive pairs
        int nOut = nMuscles / 2;
        Eigen::VectorXd out(nOut);
        for (int i = 0; i < nOut; ++i) {
            out[i] = dcnAct[2 * i] - dcnAct[2 * i + 1];
        }
        return out;
    }

    void loadWts(const Eigen::MatrixXd& init_pf_pc, const Eigen::VectorXd& init_mf_dcn, const Eigen::VectorXd& init_pc_dcn){
        pf_pc_weights  = init_pf_pc;
        mf_dcn_weights = init_mf_dcn;
        pc_dcn_weights = init_pc_dcn;
    }

    Eigen::VectorXd compute(const Eigen::VectorXd& qError, const Eigen::VectorXd& qdError, int state) {
        if (state == 0) {
            currPF = 0;
            purkinjeCompute();
            DCNCompute();
            return dcnToTorque();
        }
        granuleLayer(state);

        // NOTE: mirrors the original Python, which hardcodes 3-DOF gain
        // vectors (posCon/velCon). This assumes qError/qdError have length 3
        // (nMuscles == 6). If you generalize n_dof, generalize these too.
        static const Eigen::Vector3d posCon(1.0, 12.0, 6.0);
        static const Eigen::Vector3d velCon(2.0, 10.0, 5.0);
        if (qError.size() != 3 || qdError.size() != 3) {
            throw std::runtime_error("compute(): qError/qdError must have length 3 (posCon/velCon are hardcoded 3-DOF)");
        }

        Eigen::VectorXd error3 = posCon.cwiseProduct(qError) + velCon.cwiseProduct(qdError);
        Eigen::VectorXd error(nMuscles);
        for (int i = 0; i < error3.size(); ++i) {
            double agonist    = std::max(error3[i], 0.0);
            double antagonist = std::max(-error3[i], 0.0);
            error[2 * i]     = agonist;
            error[2 * i + 1] = antagonist;
        }

        if (active_pf_pc)
            updatePF_PC(error);
        purkinjeCompute();
        if (active_mf_dcn)
            updateMF_DCN();
        if (active_pc_dcn)
            updatePC_DCN();
        DCNCompute();
        return dcnToTorque();
    }

    int currPF;
    int nPFs;
    int pfIdx;
    int nMuscles;
    bool active_pf_pc;
    bool active_pc_dcn;
    bool active_mf_dcn;

private:
    Eigen::VectorXd purAct;
    Eigen::MatrixXd pf_pc_weights;
    Eigen::VectorXd mf_dcn_weights;
    Eigen::VectorXd pc_dcn_weights;
    Eigen::VectorXd dcnAct;

    const double LTP_max     = 0.01;
    const double LTD_max     = 0.02;
    const double LTP_max_dcn = 1e-3;
    const double LTD_max_dcn = 1e-4;
    const double alpha       = 1000.0;
};
