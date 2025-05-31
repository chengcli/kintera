#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*user_func1)(double temp);

/*!
 * \brief Calculate thermodynamic properties based on temperature and
 * concentrations
 *
 * \param[in] temp temperature in Kelvin.
 * \param[in] conc concentrations of species in mol/m^3.
 * \param[in] nspecies number of species.
 * \param[in] offset thermodynamic property offsets for each species, used in
 * the calculation.
 * \param[in] first_derivative first derivative of thermodynamic properties with
 * respect to temperature.
 * \param[in] extra user-defined functions for additional temperature dependent
 * dependencies.
 */
double thermo_prop(double temp, double const *conc, int nspecies,
                   double const *offset, double const *first_derivative,
                   user_func1 const *extra);

/*
 * \brief Convert mole fractions to mole concentrations
 *
 * This function converts mole fractions of species to their
 * corresponding concentrations in mol/m^3.
 * Gas species are assumed to be the first `ngas` species in the
 * `xfrac` array, and the remaining species are considered
 * condensed.
 *
 * \param[out] conc output array of concentrations in mol/m^3.
 * \param[in] p_ov_RT pressure divided by the product of gas constant and
 * temperature (P/RT).
 * \param[in] xfrac input array of mole fractions.
 * \param[in] nspecies number of species.
 * \param[in] ngas number of gas species.
 */
void frac2conc(double *conc, double p_ov_RT, double const *xfrac, int nspecies,
               int ngas);

/*
 * \brief Convert mole concentrations to mole fractions
 *
 * This function converts mole concentrations of species to their
 * corresponding mole fractions.
 * Gas species are assumed to be the first `ngas` species in the
 * `conc` array, and the remaining species are considered
 * condensed.
 *
 * \param[out] xfrac output array of mole fractions.
 * \param[in] conc input array of concentrations in mol/m^3.
 * \param[in] nspecies number of species.
 * \param[in] ngas number of gas species.
 */
void conc2frac(double *xfrac, double const *conc, int nspecies, int ngas);

/*!
 * \brief Calculate thermodynamic equilibrium at gven temperature and pressure
 *
 * This function finds the thermodynamic equilibrium for an array
 * of species.
 *
 * \param[in,out] xfrac array of species mole fractions, modified in place.
 * \param[in] temp equilibrium temperature in Kelvin.
 * \param[in] pres equilibrium pressure in Pascals.
 * \param[in] nspecies number of species in the system.
 * \param[in] ngas number of gas species in the system.
 * \param[in] logsvp_func user-defined function for logarithm of saturation
 * vapor pressure.
 * \param[in] logsvp_func_ddT user-defined function for derivative of logsvp
 * with respect to temperature.
 * \param[in] logsvp_eps tolerance for convergence in logarithm of saturation
 * vapor pressure.
 * \param[in,out] max_iter maximum number of iterations allowed for convergence.
 */
int equilibrate_tp(double *xfrac, double temp, double pres,
                   double const *stoich, int nspecies, int nreaction, int ngas,
                   user_func1 const *logsvp_func, double logsvp_eps,
                   int *max_iter);

/*!
 * \brief Calculate thermodynamic equilibrium at fixed volume and internal
 * energy
 *
 * Given an initial guess of temperature and concentrations, this function
 * adjusts the temperature and concentrations to satisfy the saturation
 * condition.
 *
 * \param[in,out] temp in: initial temperature, out: adjusted temperature.
 * \param[in,out] conc in: initial concentrations for each species, out:
 * adjusted concentrations.
 * \param[in] h0 initial enthalpy.
 * \param[in] stoich reaction stoichiometric matrix, nspecies x nreaction.
 * \param[in] nspecies number of species in the system.
 * \param[in] nreaction number of reactions in the system.
 * \param[in] enthalpy_offset offset for enthalpy calculations.
 * \param[in] cp_const const component of heat capacity.
 * \param[in] logsvp_func user-defined functions for logarithm of saturation
 * vapor pressure.
 * \param[in] logsvp_func_ddT user-defined functions for derivative of logsvp
 *            with respect to temperature.
 * \param[in] enthalpy_extra user-defined functions for enthalpy calculation
 *            in addition to the linear term.
 * \param[in] enthalpy_extra_ddT user-defined functions for enthalpy derivative
 *            with respect to temperature in addition to the constant term.
 * \param[in] lnsvp_eps tolerance for convergence in logarithm of saturation
 * vapor pressure.
 * \param[in,out] max_iter maximum number of iterations allowed for convergence.
 */
int equilibrate_uv(double *temp, double *conc, double h0, double const *stoich,
                   int nspecies, int nreaction, double const *enthalpy_offset,
                   double const *cp_const, user_func1 const *logsvp_func,
                   user_func1 const *logsvp_func_ddT,
                   user_func1 const *enthalpy_extra,
                   user_func1 const *enthalpy_extra_ddT, double lnsvp_eps,
                   int *max_iter);

#ifdef __cplusplus
} /* extern "C" */
#endif
