#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <vector>

#include "humanoid_cost_matching_cpp/offline/OfflineQEvaluator.h"

namespace py = pybind11;
using ocs2::scalar_t;
using ocs2::vector_array_t;
using ocs2::vector_t;

namespace ocs2::humanoid_cost_matching {

static void require_shape(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error("[pybind_offline_q] " + msg);
}

// Convert a contiguous 1D row pointer (double*) to vector_t
static inline vector_t toVec(const double* ptr, int dim) {
  vector_t v(dim);
  Eigen::Map<const Eigen::VectorXd> m(ptr, dim);
  v = m;
  return v;
}

class OfflineQEvaluatorPy {
 public:
  OfflineQEvaluatorPy(std::string taskFile, std::string urdfFile, std::string referenceFile)
      : eval_(std::move(taskFile), std::move(urdfFile), std::move(referenceFile)) {}

  py::array_t<double> evaluate_q_batch(py::array_t<double, py::array::c_style | py::array::forcecast> t0,            // (B,)
                                       py::array_t<double, py::array::c_style | py::array::forcecast> x0,            // (B,nx)
                                       py::array_t<long long, py::array::c_style | py::array::forcecast> initMode,   // (B,)
                                       py::array_t<double, py::array::c_style | py::array::forcecast> xMeasSeq,      // (B,N+1,nx)
                                       py::array_t<double, py::array::c_style | py::array::forcecast> uSeq,          // (B,N,nu)
                                       py::array_t<double, py::array::c_style | py::array::forcecast> target_time,   // (B,3)
                                       py::array_t<double, py::array::c_style | py::array::forcecast> target_state,  // (B,3,nx)
                                       py::array_t<double, py::array::c_style | py::array::forcecast> target_input,  // (B,3,nu)
                                       double dt) {
    // --- shapes ---
    auto bt0 = t0.request();
    auto bx0 = x0.request();
    auto bmode = initMode.request();
    auto bxm = xMeasSeq.request();
    auto bu = uSeq.request();
    auto btt = target_time.request();
    auto btx = target_state.request();
    auto btu = target_input.request();

    require_shape(bt0.ndim == 1, "t0 must be 1D (B,).");
    require_shape(bmode.ndim == 1, "initMode must be 1D (B,).");
    require_shape(bx0.ndim == 2, "x0 must be 2D (B,nx).");
    require_shape(bxm.ndim == 3, "xMeasSeq must be 3D (B,N+1,nx).");
    require_shape(bu.ndim == 3, "uSeq must be 3D (B,N,nu).");
    require_shape(btt.ndim == 2 && btt.shape[1] == 3, "target_time must be (B,3).");
    require_shape(btx.ndim == 3 && btx.shape[1] == 3, "target_state must be (B,3,nx).");
    require_shape(btu.ndim == 3 && btu.shape[1] == 3, "target_input must be (B,3,nu).");

    const int B = static_cast<int>(bt0.shape[0]);
    const int nx = static_cast<int>(bx0.shape[1]);
    const int N = static_cast<int>(bu.shape[1]);
    const int nu = static_cast<int>(bu.shape[2]);

    require_shape(static_cast<int>(bmode.shape[0]) == B, "initMode length mismatch.");
    require_shape(static_cast<int>(bx0.shape[0]) == B, "x0 batch size mismatch.");
    require_shape(static_cast<int>(bxm.shape[0]) == B, "xMeasSeq batch size mismatch.");
    require_shape(static_cast<int>(bu.shape[0]) == B, "uSeq batch size mismatch.");
    require_shape(static_cast<int>(btt.shape[0]) == B, "target_time batch size mismatch.");
    require_shape(static_cast<int>(btx.shape[0]) == B, "target_state batch size mismatch.");
    require_shape(static_cast<int>(btu.shape[0]) == B, "target_input batch size mismatch.");

    require_shape(static_cast<int>(btx.shape[2]) == nx, "target_state nx mismatch.");
    require_shape(static_cast<int>(btu.shape[2]) == nu, "target_input nu mismatch.");

    if (dt <= 0.0) throw std::runtime_error("[pybind_offline_q] dt must be > 0.");

    // output (B,)
    py::array_t<double> out(B);
    auto bout = out.mutable_unchecked<1>();

    const double* pt0 = static_cast<const double*>(bt0.ptr);
    const double* px0 = static_cast<const double*>(bx0.ptr);
    const long long* pmode = static_cast<const long long*>(bmode.ptr);
    const double* pxm = static_cast<const double*>(bxm.ptr);
    const double* pu = static_cast<const double*>(bu.ptr);
    const double* ptt = static_cast<const double*>(btt.ptr);
    const double* ptx = static_cast<const double*>(btx.ptr);
    const double* ptu = static_cast<const double*>(btu.ptr);

    // heavy compute: release GIL
    py::gil_scoped_release release;

    for (int b = 0; b < B; ++b) {
      OfflineSample sample;
      sample.t0 = static_cast<scalar_t>(pt0[b]);
      sample.dt = static_cast<scalar_t>(dt);
      sample.horizon = static_cast<scalar_t>(N * dt);
      sample.initMode = static_cast<size_t>(pmode[b]);

      // x0: row-major contiguous assumed by c_style
      // px0 layout: (B,nx)
      sample.x0 = toVec(px0 + b * nx, nx);

      // xMeasSeq: (B, N+1, nx) contiguous
      sample.xMeasSeq.resize(N + 1);
      for (int k = 0; k < N + 1; ++k) {
        const double* xk_meas = pxm + (b * (N + 1) * nx) + (k * nx);
        sample.xMeasSeq[k] = toVec(xk_meas, nx);
      }

      // uSeq: (B,N,nu) contiguous
      sample.uSeq.resize(N);
      for (int i = 0; i < N; ++i) {
        const double* ui = pu + (b * N * nu) + (i * nu);  // ToDo: check the indexing carefully
        sample.uSeq[i] = toVec(ui, nu);
      }

      // target 3 knots
      sample.target.timeTrajectory.resize(3);
      sample.target.stateTrajectory.resize(3);
      sample.target.inputTrajectory.resize(3);

      // target_time: (B,3)
      for (int k = 0; k < 3; ++k) {
        sample.target.timeTrajectory[k] = static_cast<scalar_t>(ptt[b * 3 + k]);
      }

      // target_state: (B,3,nx)
      for (int k = 0; k < 3; ++k) {
        const double* xk = ptx + (b * 3 * nx) + (k * nx);
        sample.target.stateTrajectory[k] = toVec(xk, nx);
      }

      // target_input: (B,3,nu)
      for (int k = 0; k < 3; ++k) {
        const double* uk = ptu + (b * 3 * nu) + (k * nu);
        sample.target.inputTrajectory[k] = toVec(uk, nu);
      }

      const scalar_t Q = eval_.evaluateQ(sample);
      bout(b) = static_cast<double>(Q);
    }

    return out;
  }

  py::tuple evaluate_q_and_grad_batch(py::array_t<double, py::array::c_style | py::array::forcecast> t0,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> x0,
                                      py::array_t<long long, py::array::c_style | py::array::forcecast> initMode,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> xMeasSeq,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> uSeq,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> tt,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> tx,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> tu,
                                      double dt,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> thetaBatch) {
    // shapes
    const auto B = t0.shape(0);
    const auto nx = x0.shape(1);
    const auto N = uSeq.shape(1);
    const auto nu = uSeq.shape(2);
    const auto p = thetaBatch.ndim() == 1 ? thetaBatch.shape(0) : thetaBatch.shape(1);

    // outputs
    py::array_t<double> Qs({B});
    py::array_t<double> grads({B, p});

    auto t0a = t0.unchecked<1>();
    auto x0a = x0.unchecked<2>();
    auto ma = initMode.unchecked<1>();
    auto xMeasSa = xMeasSeq.unchecked<3>();
    auto uSa = uSeq.unchecked<3>();
    auto tta = tt.unchecked<2>();
    auto txa = tx.unchecked<3>();
    auto tua = tu.unchecked<3>();

    auto Qout = Qs.mutable_unchecked<1>();
    auto Gout = grads.mutable_unchecked<2>();

    {
      py::gil_scoped_release release;

      for (ssize_t b = 0; b < B; ++b) {
        OfflineSample s;
        s.t0 = t0a(b);
        s.dt = dt;
        s.horizon = static_cast<double>(N) * dt;
        s.initMode = static_cast<int>(ma(b));

        s.x0 = ocs2::vector_t(nx);
        for (ssize_t i = 0; i < nx; ++i) s.x0(i) = x0a(b, i);

        s.xMeasSeq.resize(N + 1);
        for (ssize_t k = 0; k < N + 1; ++k) {
          s.xMeasSeq[k] = ocs2::vector_t(nx);
          for (ssize_t i = 0; i < nx; ++i) {
            s.xMeasSeq[k](i) = xMeasSa(b, k, i);
          }
        }

        s.uSeq.resize(N);
        for (ssize_t k = 0; k < N; ++k) {
          s.uSeq[k] = ocs2::vector_t(nu);
          for (ssize_t j = 0; j < nu; ++j) s.uSeq[k](j) = uSa(b, k, j);
        }

        // target trajectories 3 knots
        s.target.timeTrajectory.resize(3);
        s.target.stateTrajectory.resize(3);
        s.target.inputTrajectory.resize(3);

        for (int k = 0; k < 3; ++k) {
          s.target.timeTrajectory[k] = tta(b, k);

          s.target.stateTrajectory[k] = ocs2::vector_t(nx);
          for (ssize_t i = 0; i < nx; ++i) s.target.stateTrajectory[k](i) = txa(b, k, i);

          s.target.inputTrajectory[k] = ocs2::vector_t(nu);
          for (ssize_t j = 0; j < nu; ++j) s.target.inputTrajectory[k](j) = tua(b, k, j);
        }

        // theta
        ocs2::vector_t theta(p);
        if (thetaBatch.ndim() == 1) {
          auto th = thetaBatch.unchecked<1>();
          for (ssize_t i = 0; i < p; ++i) theta(i) = th(i);
        } else {
          auto th = thetaBatch.unchecked<2>();
          for (ssize_t i = 0; i < p; ++i) theta(i) = th(b, i);
        }

        auto out = eval_.evaluateQAndGradFixedGrid(s, theta);

        Qout(b) = out.Q;
        for (ssize_t i = 0; i < p; ++i) Gout(b, i) = out.grad(i);
      }
    }

    return py::make_tuple(Qs, grads);
  }

 private:
  OfflineQEvaluator eval_;
};

}  // namespace ocs2::humanoid_cost_matching

PYBIND11_MODULE(humanoid_cost_matching_offline_eval_py, m) {
  using namespace ocs2::humanoid_cost_matching;

  m.doc() = "Offline Q evaluator (batch) for humanoid_cost_matching_cpp";

  py::class_<OfflineQEvaluatorPy>(m, "OfflineQEvaluator")
      .def(py::init<std::string, std::string, std::string>(), py::arg("taskFile"), py::arg("urdfFile"), py::arg("referenceFile"))
      .def("evaluate_q_batch", &OfflineQEvaluatorPy::evaluate_q_batch, py::arg("t0"), py::arg("x0"), py::arg("initMode"),
           py::arg("xMeasSeq"), py::arg("uSeq"), py::arg("target_time"), py::arg("target_state"), py::arg("target_input"), py::arg("dt"),
           "Compute Q for a batch.\n"
           "Shapes:\n"
           "  t0: (B,)\n"
           "  x0: (B,nx)\n"
           "  initMode: (B,)\n"
           "  xMeasSeq: (B,N+1,nx)\n"
           "  uSeq: (B,N,nu)\n"
           "  target_time: (B,3)\n"
           "  target_state: (B,3,nx)\n"
           "  target_input: (B,3,nu)\n"
           "Returns:\n"
           "  Qs: (B,)\n")
      .def("evaluate_q_and_grad_batch", &OfflineQEvaluatorPy::evaluate_q_and_grad_batch, py::arg("t0"), py::arg("x0"), py::arg("initMode"),
           py::arg("xMeasSeq"), py::arg("uSeq"), py::arg("tt"), py::arg("tx"), py::arg("tu"), py::arg("dt"), py::arg("thetaBatch"));
}
