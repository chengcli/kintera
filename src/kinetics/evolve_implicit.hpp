#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

//! Single-step implicit Euler solver for chemical kinetics.
//!
//! Solves  (I/dt - S*J) * delta = S*rate  where S is the stoichiometric
//! matrix, J is the reaction-space Jacobian, and rate is the vector of
//! reaction rates. Supports batched layers via leading dimensions.
//!
//! Defined out-of-line in evolve_implicit_dispatch.cpp (not inline): it calls a
//! DECLARE_DISPATCH stub, whose DispatchStubImpl::get_call_ptr arity is gated
//! by torch's AVX feature macros. Keeping the definition in the kintera library
//! (built with those macros via Caffe2) ensures the get_call_ptr reference is
//! never compiled into a downstream consumer (e.g. the pybind module) that was
//! built without them.
//!
//! @param rate     reaction rates, shape (..., nreaction)
//! @param stoich   stoichiometric matrix, shape (nspecies, nreaction)
//! @param jacobian reaction-space Jacobian, shape (..., nreaction, nspecies)
//! @param dt       time step
//! @return         concentration change delta, shape (..., nspecies)
torch::Tensor evolve_implicit(torch::Tensor rate, torch::Tensor stoich,
                              torch::Tensor jacobian, double dt);

//! Two-stage Rosenbrock (Ros2) solver for stiff chemical kinetics.
//!
//! This is the same method used by VULCAN.  It factorises a single
//! system matrix  W = I/(γ·dt) - S·J  (evaluated once per step)
//! and performs two back-solves to obtain a second-order solution
//! together with an embedded first-order error estimate.
//!
//! The caller must evaluate rate2 at C + (1/γ)*k1, where k1 is the
//! first-stage solution returned by a prior call or computed here.
//!
//! @param rate1    reaction rates at C,          shape (..., nreaction)
//! @param rate2    reaction rates at C + (1/γ)*k1, shape (..., nreaction)
//! @param stoich   stoichiometric matrix,        shape (nspecies, nreaction)
//! @param jacobian reaction-space Jacobian at C, shape (..., nreaction,
//! nspecies)
//! @param dt       time step
//! @return tuple(delta, error)
//!         delta – 2nd-order concentration change, shape (..., nspecies)
//!         error – embedded error estimate,        shape (..., nspecies)
inline std::tuple<torch::Tensor, torch::Tensor> evolve_ros2(
    torch::Tensor rate1, torch::Tensor rate2, torch::Tensor stoich,
    torch::Tensor jacobian, double dt) {
  // Ros2 tableau  (T-form / gamma-form, L-stable, Verwer et al. 1999)
  //
  //   W = I/(γ·h) - J
  //   W · k1 = f(yn)
  //   W · k2 = f(yn + k1/γ) - 2/(γ·h) · k1
  //   yn+1   = yn + 3/(2γ)·k1 + 1/(2γ)·k2
  //
  // k1, k2 have units of [concentration], NOT [concentration/time].
  constexpr double gamma = 1.7071067811865476;  // 1 + 1/sqrt(2)
  constexpr double c21 = -2.0 / gamma;          // ≈ -1.1716
  constexpr double m1 = 1.5 / gamma;            // ≈ 0.8787
  constexpr double m2 = 0.5 / gamma;            // ≈ 0.2929
  constexpr double e1 = m1 - 1.0;               // ≈ -0.1213
  constexpr double e2 = m2;                     // ≈  0.2929

  auto nspecies = stoich.size(0);
  auto eye = torch::eye(nspecies, rate1.options());
  auto SJ = stoich.matmul(jacobian);

  auto W = eye / (gamma * dt) - SJ;

  // Stage 1:  W * k1 = S * rate1
  auto f1 = stoich.matmul(rate1.unsqueeze(-1)).squeeze(-1);
  auto k1 = std::get<0>(torch::linalg_lstsq(W, f1.unsqueeze(-1))).squeeze(-1);

  // Stage 2:  W * k2 = S * rate2 + (c21/dt) * k1
  auto f2 = stoich.matmul(rate2.unsqueeze(-1)).squeeze(-1);
  auto rhs2 = f2 + (c21 / dt) * k1;
  auto k2 = std::get<0>(torch::linalg_lstsq(W, rhs2.unsqueeze(-1))).squeeze(-1);

  // 2nd-order solution  (no dt multiplier — k1,k2 are already Δy)
  auto delta = m1 * k1 + m2 * k2;

  // Embedded error estimate
  auto error = e1 * k1 + e2 * k2;

  return {delta, error};
}

//! First stage of Ros2 — returns k1 so the caller can evaluate
//! rate2 at C + (1/gamma)*k1 before calling evolve_ros2.
inline torch::Tensor ros2_k1(torch::Tensor rate1, torch::Tensor stoich,
                             torch::Tensor jacobian, double dt) {
  constexpr double gamma = 1.7071067811865476;
  auto nspecies = stoich.size(0);
  auto eye = torch::eye(nspecies, rate1.options());
  auto SJ = stoich.matmul(jacobian);
  auto W = eye / (gamma * dt) - SJ;
  auto f1 = stoich.matmul(rate1.unsqueeze(-1)).squeeze(-1);
  return std::get<0>(torch::linalg_lstsq(W, f1.unsqueeze(-1))).squeeze(-1);
}

}  // namespace kintera
