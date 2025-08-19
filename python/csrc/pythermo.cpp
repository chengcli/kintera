// torch
#include <torch/extension.h>

// pybind11
#include <pybind11/stl.h>

// kintera
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module &m) {
  auto pyNucleationOptions =
      py::class_<kintera::NucleationOptions>(m, "NucleationOptions");

  pyNucleationOptions
      .def(py::init<>(), R"doc(
Returns:
  NucleationOptions: class object

Examples:
  .. code-block:: python

    >> from kintera import NucleationOptions
    >> nucleation = NucleationOptions().minT([200.0]).maxT([300.0]).reaction(["H2O + CO2 -> H2CO3"])
    )doc")

      .def("__repr__",
           [](const kintera::NucleationOptions &self) {
             std::stringstream ss;
             self.report(ss);
             return fmt::format("NucleationOptions({})", ss.str());
           })

      .ADD_OPTION(std::vector<double>, kintera::NucleationOptions, minT, R"doc(
Set or get the minimum temperature for the nucleation reaction.

Args:
  value (list[float]): list of Minimum temperature in Kelvin.

Returns:
  Nucleation | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import NucleationOptions
    >> nucleation = NucleationOptions().minT([200.0])
    >> print(nucleation.minT())
    [200.0]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::NucleationOptions, maxT, R"doc(
Set or get the maximum temperature for the nucleation reaction.

Args:
  value (list[float]): list of Maximum temperature in Kelvin.

Returns:
  Nucleation | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import NucleationOptions
    >> nucleation = NucleationOptionis().maxT(300.0)
    >> print(nucleation.maxT())
    [300.0]
    )doc")

      .ADD_OPTION(std::vector<std::string>, kintera::NucleationOptions, logsvp,
                  R"doc(
Set or get the log of saturation vapor function for the nucleation reaction.

Args:
  value (list[str]): list of log saturation vapor pressure functions.

Returns:
  Nucleation | list[str]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import NucleationOptions
    >> nucleation = NucleationOptions().logsvp(["h2o_ideal"])
    >> print(nucleation.logsvp())
    ["h2o_ideal"]
    )doc")

      .ADD_OPTION(std::vector<kintera::Reaction>, kintera::NucleationOptions,
                  reactions,
                  R"doc(
Set or get the reaction associated with the nucleation.

Args:
  value (list[Reaction]): list of Reaction object representing the nucleation reactions.

Returns:
  Nucleation | list[Reaction]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import NucleationOptions, Reaction
    >> reaction = Reaction("H2O + CO2 -> H2CO3")
    >> nucleation = NucleationOptions().reactions([reaction])
    >> print(nucleation.reactions())
    [Reaction(H2O + CO2 -> H2CO3)]
    )doc");

  auto pyThermoOptions =
      py::class_<kintera::ThermoOptions, kintera::SpeciesThermo>(
          m, "ThermoOptions");

  pyThermoOptions
      .def(py::init<>(), R"doc(
Returns:
  ThermoOptions: class object

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().Tref(300.0).Pref(1.e5).cref_R(2.5)
    )doc")

      .def("__repr__",
           [](const kintera::ThermoOptions &self) {
             std::stringstream ss;
             self.report(ss);
             return fmt::format("ThermoOptions({})", ss.str());
           })

      .def("reactions", &kintera::ThermoOptions::reactions, R"doc(
Returns:
  list[Reaction]: list of reactions

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().reactions([Reaction("H2 + O2 -> H2O")])
    >> print(op.reactions())
    [Reaction(H2 + O2 -> H2O)]
    )doc")

      .def("from_yaml",
           py::overload_cast<std::string const &>(
               &kintera::ThermoOptions::from_yaml),
           R"doc(
Create a `ThermoOptions` object from a YAML file.

Args:
  filename (str): Path to the YAML file.

Returns:
  ThermoOptions: class object

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions.from_yaml("thermo_options.yaml")
    )doc")

      .ADD_OPTION(double, kintera::ThermoOptions, Tref, R"doc(
Set or get the reference temperature (default: 300.0).

Args:
  value (float): Reference temperature value.

Returns:
  ThermoOptions | float: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().Tref(300.0)
    >> print(op.Tref())
    300.0
    )doc")

      .ADD_OPTION(double, kintera::ThermoOptions, Pref, R"doc(
Set or get the reference pressure (default: 1.e5).

Args:
  value (float): Reference pressure value.

Returns:
  ThermoOptions | float: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().Pref(1.e5)
    >> print(op.Pref())
    100000.0
    )doc")

      .ADD_OPTION(kintera::NucleationOptions, kintera::ThermoOptions,
                  nucleation, R"doc(
Set or get the nucleation reactions options

Args:
  value (NucleationOptions): nucleation reaction options

Returns:
  SpeciesThermo | NucleationOptions: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo, NucleationOptions
    >> op = SpeciesThermo().react([Nucleation("R1", 200.0, 300.0), Nucleation("R2", 250.0, 350.0)])
    >> print(op.react())
    [Nucleation(R1; minT = 200.00; maxT = 300.00), Nucleation(R2; minT = 250.00; maxT = 350.00)]
    )doc")

      .ADD_OPTION(int, kintera::ThermoOptions, max_iter, R"doc(
Set or get the maximum number of iterations for convergence (default: 10).

Args:
  value (int): Maximum number of iterations.

Returns:
  ThermoOptions | int: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().max_iter(10)
    >> print(op.max_iter())
    10
    )doc")

      .ADD_OPTION(double, kintera::ThermoOptions, ftol, R"doc(
Set or get the convergence tolerance for the free energy (default: 1.e-6).

Args:
  value (float): Convergence tolerance for the free energy.

Returns:
  ThermoOptions | float: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().ftol(1.e-6)
    >> print(op.ftol())
    1e-06
    )doc");

  ADD_KINTERA_MODULE(ThermoY, ThermoOptions, R"doc(
Perform saturation adjustment

Args:
  rho (torch.Tensor): Density tensor [kg/m^3].
  intEng (torch.Tensor): Internal energy tensor [J/m^3].
  yfrac (torch.Tensor): Mass fraction tensor.

Returns:
  torch.Tensor: changed in mass fraction

Examples:
  .. code-block:: python

    >> from kintera import ThermoY, ThermoOptions
    >> options = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    >> thermo_y = ThermoY(options)
    >> rho = torch.tensor([1.0, 2.0, 3.0])
    >> intEng = torch.tensor([1000.0, 2000.0, 3000.0])
    >> yfrac = torch.tensor([0.1, 0.2, 0.3])
    >> dyfrac = thermo_y.forward(rho, intEng, yfrac)
    )doc",
                     py::arg("rho"), py::arg("intEng"), py::arg("yfrac"),
                     py::arg("diag") = py::none())

      .def("compute", &kintera::ThermoYImpl::compute, R"doc(
Compute the transformation.

Args:
  ab (str): Transformation string, choose from
            ["C->Y", "Y->X", "DY->C", "DPY->U", "DUY->P", "DPY->T", "DTY->P"].
  args (list): List of arguments for the transformation.
  out (torch.Tensor, optional): Output tensor to store the result.

Returns:
  torch.Tensor: Resulting tensor after the transformation.

Examples:
  .. code-block:: python

    >> from kintera import ThermoY, ThermoOptions
    >> options = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    >> thermo_y = ThermoY(options)
    >> result = thermo_y.compute("C->Y", [torch.tensor([1.0, 2.0, 3.0])])
    )doc",
           py::arg("ab"), py::arg("args"));

  ADD_KINTERA_MODULE(ThermoX, ThermoOptions, R"doc(
Perform equilibrium condensation

Args:
  temp (torch.Tensor): Temperature tensor [K].
  pres (torch.Tensor): Pressure tensor [Pa].
  xfrac (torch.Tensor): mole fraction tensor.

Returns:
  torch.Tensor: changes in mole fraction

Examples:
  .. code-block:: python

    >> from kintera import ThermoX, ThermoOptions
    >> options = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    >> thermo_x = ThermoX(options)
    >> temp = torch.tensor([300.0, 310.0, 320.0])
    >> pres = torch.tensor([1.e5, 1.e6, 1.e7])
    >> xfrac = torch.tensor([0.1, 0.2, 0.3])
    >> dxfrac = thermo_x.forward(temp, pres, xfrac)
    )doc",
                     py::arg("temp"), py::arg("pres"), py::arg("xfrac"),
                     py::arg("diag") = py::none())

      .def("compute", &kintera::ThermoXImpl::compute, R"doc(
Compute the transformation.

Args:
  ab (str): Transformation string, choose from ["X->Y", "TPX->D"].
  args (list): List of arguments for the transformation.
  out (torch.Tensor, optional): Output tensor to store the result.

Returns:
  torch.Tensor: Resulting tensor after the transformation.

Examples:
  .. code-block:: python

    >> from kintera import ThermoX, ThermoOptions
    >> options = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    >> thermo_x = ThermoX(options)
    >> result = thermo_x.compute("X->Y", [torch.tensor([0.1, 0.2, 0.3])])
    )doc",
           py::arg("ab"), py::arg("args"))

      .def("effective_cp", &kintera::ThermoXImpl::effective_cp, R"doc(
Compute the effective specific heat capacity at constant pressure.

Args:
  temp (torch.Tensor): Temperature tensor [K].
  pres (torch.Tensor): Pressure tensor [Pa].
  xfrac (torch.Tensor): Mole fraction tensor.
  gain (torch.Tensor): gain tensor.
  conc (torch.Tensor, optional): Concentration tensor.

Returns:
  torch.Tensor: Effective molar heat capacity tensor [J/(mol*K)].
  )doc",
           py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("gain"),
           py::arg("conc") = py::none())

      .def("extrapolate_ad",
           py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor,
                             double>(&kintera::ThermoXImpl::extrapolate_ad),
           R"doc(
Extrapolate the temperature and pressure along an adiabatic path.

Args:
  temp (torch.Tensor): Temperature tensor [K].
  pres (torch.Tensor): Pressure tensor [Pa].
  xfrac (torch.Tensor): Mole fraction tensor.
  dlnp (float): delta ln pressure

Returns:
  None

Examples:
  .. code-block:: python

    >> from kintera import ThermoX, ThermoOptions
    >> options = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    >> thermo = ThermoX(options)
    >> temp = torch.tensor([300.0, 310.0, 320.0])
    >> pres = torch.tensor([1.e5, 1.e6, 1.e7])
    >> xfrac = torch.tensor([0.1, 0.2, 0.3])
    >> thermo_x.extrapolate_ad(temp, pres, xfrac, thermo, -0.01)
    )doc",
           py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("dlnp"))

      .def(
          "extrapolate_ad",
          py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, double,
                            double>(&kintera::ThermoXImpl::extrapolate_ad),
          R"doc(
Extrapolate the temperature and pressure along an adiabatic path.

Args:
  temp (torch.Tensor): Temperature tensor [K].
  pres (torch.Tensor): Pressure tensor [Pa].
  xfrac (torch.Tensor): Mole fraction tensor.
  grav (float): gravitational acceleration [m/s^2].
  dz (float): delta height [m].

Returns:
  None

Examples:
  .. code-block:: python

    >> from kintera import ThermoX, ThermoOptions
    >> options = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    >> thermo = ThermoX(options)
    >> temp = torch.tensor([300.0, 310.0, 320.0])
    >> pres = torch.tensor([1.e5, 1.e6, 1.e7])
    >> xfrac = torch.tensor([0.1, 0.2, 0.3])
    >> thermo_x.extrapolate_ad(temp, pres, xfrac, thermo, -0.01)
    )doc",
          py::arg("temp"), py::arg("pres"), py::arg("xfrac"), py::arg("grav"),
          py::arg("dz"));

  m.def("relative_humidity", &kintera::relative_humidity, R"doc(
Calculate the relative humidity.

Args:
  temp (torch.Tensor): Temperature tensor [K].
  conc (torch.Tensor): Concentration tensor [mol/m^3].
  stoich (torch.Tensor): Stoichiometric coefficients tensor.
  op (ThermoOptions): Thermodynamic options.

Returns:
  torch.Tensor: Relative humidity tensor.

Examples:
  .. code-block:: python

    >> from kintera import relative_humidity, ThermoOptions
    >> ...
    >> temp = torch.tensor([300.0, 310.0, 320.0])
    >> conc = torch.tensor([1.e-3, 2.e-3, 3.e-3])
    >> stoich = thermo.get_buffer("stoich")
    >> rh = relative_humidity(temp, conc, stoich, op.nucleation())
    )doc",
        py::arg("temp"), py::arg("conc"), py::arg("stoich"), py::arg("op"));
}
