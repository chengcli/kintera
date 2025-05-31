// C/C++
#include <algorithm>
#include <numeric>

// Eigen
#include <Eigen/Dense>

namespace kintera {

// x -> y at constant volume (mole concentration)
template <typename T>
inline DISPATCH_MACRO void satfunc1v(T* out1, T* out2, T const* s,
                                     T const* conc, T const* logs_ddT, int j,
                                     int ix, int iy) {
  auto const& x = conc[ix];
  auto const& y = conc[iy];
  T rate = x - s;

  if (rate > 0. || (rate < 0. && y > -rate)) {
    (*out1) = rate;
    (*out2) = -s * logs_ddT;
  } else {
    (*out1) = -y;
    (*out2) = -s * logs_ddT;
  }
}

template <typename T>
inline DISPATCH_MACRO void set_jac1v(T* m_jac, T const* s, T const* conc, int j,
                                     int ix, int iy) {
  auto const& x = conc[ix];
  auto const& y = conc[iy];
  T rate = x - s[j];
  if (rate > 0. || (rate < 0. && y > -rate)) {
    M_JAC(j, ix) = 1.;
  } else {
    M_JAC(j, iy) = -1.;
  }
}

template <typename T>
void equilibrate_uv_rate_impl(T* rate, T const* conc, T const* int_eng,
                              T const* svp_RT, T const* logsvp_ddT, T* m_jac,
                              T* m_stoich, int nreact, int nspecies) {
  size_t nfast = m_jxy.size() + m_jxxy.size() + m_jyy.size();

  //! initialize rate jacobian matrix
  for (int i = 0; i < nreact; i++)
    for (int j = 0; j < nspecies; j++) M_JAC(i, j) = 0.;

  for (int j = 0; j < nreact; ++j) {
    m_b_ddT[j] = 0.;
    for (int i = 0; i < nspecies; ++i) {
      M_STOICH(i, j) = 0.;
    }
  }

  // nucleation: x <=> y
  for (int j = 0; j < nreact; ++j) {
    // inactive reactions
    if (svp_RT[j] < 0.0) continue;

    int ix = species_index[0];
    int iy = species_index[1];
    T b_ddT;

    satfunc1v(rate + j, &b_ddT, svp_RT, conc, logsvp_ddT, j, ix, iy);
    set_jac1v(m_jac, svp_RT, m_conc, j, ix, iy);

    // set up temperature gradient
    for (int i = 0; i < nspecies; ++i) {
      M_STOICH(i, j) = STOICH(i, j);
      // RATE_DDT(j, i) = m_b_ddT[j] * m_intEng[i];
      M_JAC(j, i) -= b_ddT * int_eng[i];
    }
  }

  /*std::cout << "u = " << m_intEng << std::endl;
  std::cout << "cc = " << m_conc.dot(m_cv) << std::endl;
  std::cout << "cv = " << m_cv << std::endl;*/
  m_jac -= rate_ddT / m_conc.dot(m_cv);

  // solve the optimal net rates
  Eigen::MatrixXd A = m_jac * m_stoich;
  Eigen::VectorXd r = A.colPivHouseholderQr().solve(rate);

  /*std::cout << m_jac << std::endl;
  std::cout << A << std::endl;
  std::cout << A.transpose() * A << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "r = " << -r << std::endl;*/

  // scale rate down if some species becomes negative
  // Eigen::VectorXd rates = - stoich * r;

  for (size_t j = 0; j < nfast; ++j) {
    m_ropf[j] = std::max(0., -r(j));
    m_ropr[j] = std::max(0., r(j));
    rate[j] = m_ropf[j] - m_ropr[j];
  }
}

}  // namespace kintera
