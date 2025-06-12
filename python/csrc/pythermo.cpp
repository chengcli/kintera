// torch
#include <torch/extension.h>

// kintera
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module &m) {
  py::class_<kintera::Nucleation>(m, "Nucleation")
      .def(py::init<>(), R"doc(
Returns:
  Nucleation: class object

Examples:
  .. code-block:: python

    >> from kintera import Nucleation
    >> nucleation = Nucleation().minT(200.0).maxT(300.0).reaction("H2O + CO2 -> H2CO3")
    )doc")

      .def("__repr__",
           [](const kintera::Nucleation &self) {
             return fmt::format("Nucleation({})", self);
           })

      .ADD_OPTION(double, kintera::Nucleation, minT, R"doc(
Set or get the minimum temperature for the nucleation reaction.

Args:
  value (float): Minimum temperature in Kelvin.

Returns:
  Nucleation | float: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import Nucleation
    >> nucleation = Nucleation().minT(200.0)
    >> print(nucleation.minT())
    200.0
    )doc")

      .ADD_OPTION(double, kintera::Nucleation, maxT, R"doc(
Set or get the maximum temperature for the nucleation reaction.

Args:
  value (float): Maximum temperature in Kelvin.

Returns:
  Nucleation | float: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import Nucleation
    >> nucleation = Nucleation().maxT(300.0)
    >> print(nucleation.maxT())
    300.0
    )doc")

      .ADD_OPTION(kintera::Reaction, kintera::Nucleation, reaction, R"doc(
Set or get the reaction associated with the nucleation.

Args:
  value (Reaction): Reaction object representing the nucleation reaction.

Returns:
  Nucleation | Reaction: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import Nucleation, Reaction
    >> reaction = Reaction("H2O + CO2 -> H2CO3")
    >> nucleation = Nucleation().reaction(reaction)
    >> print(nucleation.reaction())
    Reaction(H2O + CO2 -> H2CO3)
    )doc");

  auto pyThermoOptions = py::class_<kintera::ThermoOptions>(m, "ThermoOptions");

  pyThermoOptions
      .def(py::init<>(), R"doc(
Returns:
  ThermoOptions: class object

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().Rd(287.0).Tref(300.0).Pref(1.e5).cref_R(2.5)
    )doc")

      .def("__repr__",
           [](const kintera::ThermoOptions &self) {
             return fmt::format("ThermoOptions({})", self);
           })

      .def("from_yaml", &kintera::ThermoOptions::from_yaml, R"doc(
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

      .ADD_OPTION(double, kintera::ThermoOptions, Rd, R"doc(
Set or get the specific gas constant (default: 287.0).

Args:
  value (float): Specific gas constant value.

Returns:
  ThermoOptions | float: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().Rd(287.0)
    >> print(op.Rd())
    287.0
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

      .ADD_OPTION(std::vector<int>, kintera::ThermoOptions, vapor_ids, R"doc(
Set or get the vapor species IDs.

Args:
  value (list[int]): List of vapor species IDs.

Returns:
  ThermoOptions | list[int]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().vapor_ids([1, 2, 3])
    >> print(op.vapor_ids())
    [1, 2, 3]
    )doc")

      .ADD_OPTION(std::vector<int>, kintera::ThermoOptions, cloud_ids, R"doc(
Set or get the cloud species IDs.

Args:
  value (list[int]): List of cloud species IDs.

Returns:
  ThermoOptions | list[int]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().cloud_ids([4, 5])
    >> print(op.cloud_ids())
    [4, 5]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::ThermoOptions, mu_ratio, R"doc(
Set or get the molecular weight ratios.

Args:
  value (list[float]): List of molecular weight ratios.

Returns:
  ThermoOptions | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().mu_ratio([1.0, 1.2, 1.5])
    >> print(op.mu_ratio())
    [1.0, 1.2, 1.5]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::ThermoOptions, cref_R, R"doc(
Set or get the specific heat ratio for the reference state.

Args:
  value (list[float]): List of specific heat ratios for the reference state.

Returns:
  ThermoOptions | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().cref_R([2.5, 2.7, 2.9])
    >> print(op.cref_R())
    [2.5, 2.7, 2.9]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::ThermoOptions, uref_R, R"doc(
Set or get the internal energy for the reference state.

Args:
  value (list[float]): List of internal energies for the reference state.

Returns:
  ThermoOptions | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().uref_R([0.0, 1.0, 2.0])
    >> print(op.uref_R())
    [0.0, 1.0, 2.0]
    )doc")

      .ADD_OPTION(std::vector<kintera::Nucleation>, kintera::ThermoOptions,
                  react, R"doc(
Set or get the nucleation reactions.

Args:
  value (list[Nucleation]): List of nucleation reactions.

Returns:
  ThermoOptions | list[Nucleation]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions, Nucleation
    >> op = ThermoOptions().react([Nucleation("R1", 200.0, 300.0), Nucleation("R2", 250.0, 350.0)])
    >> print(op.react())
    [Nucleation(R1; minT = 200.00; maxT = 300.00), Nucleation(R2; minT = 250.00; maxT = 350.00)]
    )doc")

      .ADD_OPTION(std::vector<std::string>, kintera::ThermoOptions, species,
                  R"doc(
Set or get the species names.

Args:
  value (list[str]): List of species names.

Returns:
  ThermoOptions | list[str]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import ThermoOptions
    >> op = ThermoOptions().species(["H2O", "CO2", "O2"])
    >> print(op.species())
    ['H2O', 'CO2', 'O2']
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
                     py::arg("rho"), py::arg("intEng"), py::arg("yfrac"))

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
                     py::arg("temp"), py::arg("pres"), py::arg("xfrac"))

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
           py::arg("ab"), py::arg("args"));
}
