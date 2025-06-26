#pragma once

#include "MultiRate.h"
#include "ReactionRate.h"
#include "cantera/base/Units.h"
#include "cantera/base/ct_defs.h"
#include "cantera/kinetics/ReactionData.h"

namespace kintera {

int locate(double const* xx, double x, int n);
void interpn(double* val, double const* coor, double const* data,
             double const* axis, size_t const* len, int ndim, int nval);

class ThermoPhase;
class Kinetics;

//! Data container holding shared data specific to photolysis reactions
/**
 * The data container `PhotolysisData` holds photolysis cross-section data
 * @ingroup reactionGroup
 */
struct PhotolysisData : public ReactionData {
  bool check() const;

  bool update(const ThermoPhase& thermo, const Kinetics& kin) override;
  using ReactionData::update;

  //! \brief wavelength grid
  //!
  //! The wavelength grid is a vector of size nwave.
  //! Default units are nanometers.
  torch::Tensor wavelength;

  //! \brief actinic flux
  //!
  //! The actinic flux is a vector of size nwave.
  //! Default units are photons cm^-2 s^-1 nm^-1.
  torch::Tensor actinicFlux;
};

class PhotolysisBase : public ReactionRate {
 public:
  //! Default constructor
  PhotolysisBase() {}

  //! Constructor.
  /*!
   * @param temp Temperature grid
   * @param wavelength Wavelength grid
   * @param branches Branch strings of the photolysis products
   * @param xsection Cross-section data
   */
  PhotolysisBase(vector<double> const& temp, vector<double> const& wavelength,
                 vector<std::string> const& branches,
                 vector<double> const& xsection);

  //! Constructor based on AnyValue content
  explicit PhotolysisBase(AnyMap const& node, UnitStack const& rate_units = {});

  void setParameters(AnyMap const& node, UnitStack const& rate_units) override;

  //! Set the rate parameters for each branch
  //! @param rate Rate coefficient data
  //! @param branch_map Map of branch names to branch indices
  void setRateParameters(const AnyValue& rate,
                         map<string, int> const& branch_map);

  void getParameters(AnyMap& node) const override;

  void getRateParameters(AnyMap& node) const;

  void check(string const& equation) override;

  void validate(const string& equation, const Kinetics& kin) override;

  vector<double> getCrossSection(double temp, double wavelength) const;

 protected:
  //! composition of photolysis branch products
  vector<Composition> m_branch;

  //! number of temperature grid points
  size_t m_ntemp;

  //! number of wavelength grid points
  size_t m_nwave;

  //! temperature grid followed by wavelength grid
  vector<double> m_temp_wave_grid;

  //! \brief photolysis cross-section data
  //!
  //! The cross-section data is a three dimensional table of size (ntemp, nwave,
  //! nbranch). The first dimension is the number of temperature grid points,
  //! the second dimension is the number of wavelength grid points, and the
  //! third dimension is the number of branches of the photolysis reaction.
  //! Default units are SI units such as m, m^2, and m^2/m.
  vector<double> m_crossSection;
};

//! Photolysis reaction rate type depends on temperature and the actinic flux
/*!
 * A reaction rate coefficient of the following form.
 *
 * \f[
 *    k(T) = \int_{\lambda_1}^{\lambda_2} \sigma(\lambda) \phi(\lambda) d\lambda
 * \f]
 *
 * where \f$ \sigma(\lambda) \f$ is the cross-section and \f$ \phi(\lambda) \f$
 * is the actinic flux. \f$ \lambda_1 \f$ and \f$ \lambda_2 \f$ are the lower
 * and upper bounds of the wavelength grid.
 */
class PhotolysisRate : public PhotolysisBase {
 public:
  using PhotolysisBase::PhotolysisBase;  // inherit constructor

  unique_ptr<MultiRateBase> newMultiRate() const override {
    return make_unique<MultiRate<PhotolysisRate, PhotolysisData>>();
  }

  const string type() const override { return "Photolysis"; }

  Composition const& photoProducts() const override { return m_net_products; }

  double evalFromStruct(PhotolysisData const& data);

 protected:
  //! net stoichiometric coefficients of products
  Composition m_net_products;
};

}  // namespace kintera

#endif  // CT_PHOTOLYSIS_H
