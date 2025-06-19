// torch
#include <torch/extension.h>

// kintera
#include <kintera/kintera_formatter.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/utils/find_resource.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

namespace kintera {

extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;

}  // namespace kintera

void bind_thermo(py::module &m);

PYBIND11_MODULE(kintera, m) {
  m.attr("__name__") = "kintera";
  m.doc() = R"(Atmospheric Thermodynamics and Chemistry Library)";

  bind_thermo(m);

  m.def(
      "species_names",
      []() -> const std::vector<std::string> & {
        return kintera::species_names;
      },
      R"doc(Retrieves the list of species names)doc");

  m.def(
      "species_weights",
      []() -> const std::vector<double> & { return kintera::species_weights; },
      R"doc(Retrieves the list of species molecular weights [kg/mol])doc");

  m.def(
      "set_search_paths",
      [](const std::string path) {
        strcpy(kintera::search_paths, path.c_str());
        return kintera::deserialize_search_paths(kintera::search_paths);
      },
      R"doc(
Set the search paths for resource files.

Args:
  path (str): The search paths

Return:
  str: The search paths

Example:
  .. code-block:: python

    >>> import kintera

    # set the search paths
    >>> kintera.set_search_paths("/path/to/resource/files")
      )doc",
      py::arg("path"));

  m.def(
      "get_search_paths",
      []() { return kintera::deserialize_search_paths(kintera::search_paths); },
      R"doc(
Get the search paths for resource files.

Return:
  str: The search paths

Example:
  .. code-block:: python

    >>> import kintera

    # get the search paths
    >>> kintera.get_search_paths()
      )doc");

  m.def(
      "add_resource_directory",
      [](const std::string path, bool prepend) {
        kintera::add_resource_directory(path, prepend);
        return kintera::deserialize_search_paths(kintera::search_paths);
      },
      R"doc(
Add a resource directory to the search paths.

Args:
  path (str): The resource directory to add.
  prepend (bool): If true, prepend the directory to the search paths. If false, append it.

Returns:
  str: The updated search paths.

Example:
  .. code-block:: python

    >>> import kintera

    # add a resource directory
    >>> kintera.add_resource_directory("/path/to/resource/files")
      )doc",
      py::arg("path"), py::arg("prepend") = true);

  m.def("find_resource", &kintera::find_resource, R"doc(
Find a resource file from the search paths.

Args:
  filename (str): The name of the resource file.

Returns:
  str: The full path to the resource file.

Example:
  .. code-block:: python

    >>> import kintera

    # find a resource file
    >>> path = kintera.find_resource("example.txt")
    >>> print(path)  # /path/to/resource/files/example.txt
      )doc",
        py::arg("filename"));

  auto pySpeciesThermo = py::class_<kintera::SpeciesThermo>(m, "SpeciesThermo");

  pySpeciesThermo
      .def(py::init<>(), R"doc(
Returns:
  SpeciesThermo: class object

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo()
      )doc")

      .def("__repr__",
           [](const kintera::SpeciesThermo &self) {
             return fmt::format("SpeciesThermo({})", self);
           })

      .def("species", &kintera::SpeciesThermo::species, R"doc(
Returns:
  list[str]: list of species names

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo()
    >> op.species()
    ['H2', 'O2', 'N2', 'Ar']
      )doc")

      .ADD_OPTION(std::vector<int>, kintera::SpeciesThermo, vapor_ids, R"doc(
Set or get the vapor species IDs.

Args:
  value (list[int]): List of vapor species IDs.

Returns:
  SpeciesThermo | list[int]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo().vapor_ids([1, 2, 3])
    >> print(op.vapor_ids())
    [1, 2, 3]
    )doc")

      .ADD_OPTION(std::vector<int>, kintera::SpeciesThermo, cloud_ids, R"doc(
Set or get the cloud species IDs.

Args:
  value (list[int]): List of cloud species IDs.

Returns:
  SpeciesThermo | list[int]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo().cloud_ids([1, 2, 3])
    >> print(op.cloud_ids())
    [1, 2, 3]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermo, cref_R, R"doc(
Set or get the specific heat ratio for the reference state.

Args:
  value (list[float]): List of specific heat ratios for the reference state.

Returns:
  SpeciesThermo | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo().cref_R([2.5, 2.7, 2.9])
    >> print(op.cref_R())
    [2.5, 2.7, 2.9]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermo, uref_R, R"doc(
Set or get the internal energy for the reference state.

Args:
  value (list[float]): List of internal energies for the reference state.

Returns:
  SpeciesThermo | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo().uref_R([0.0, 1.0, 2.0])
    >> print(op.uref_R())
    [0.0, 1.0, 2.0]
    )doc")

      .ADD_OPTION(std::vector<double>, kintera::SpeciesThermo, sref_R, R"doc(
Set or get the internal energy for the reference state.

Args:
  value (list[float]): List of internal energies for the reference state.

Returns:
  SpeciesThermo | list[float]: class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import SpeciesThermo
    >> op = SpeciesThermo().sref_R([0.0, 1.0, 2.0])
    >> print(op.sref_R())
    [0.0, 1.0, 2.0]
    )doc");

  auto pyReaction = py::class_<kintera::Reaction>(m, "Reaction");

  pyReaction
      .def(py::init<>(), R"doc(
Returns:
  Reaction: class object

Examples:
  .. code-block:: python

    >> from kintera import Reaction
    >> op = Reaction("H2 + O2 => H2O2")
      )doc")

      .def("__repr__",
           [](const kintera::Reaction &self) {
             return fmt::format("Reaction({})", self);
           })

      .def("equation", &kintera::Reaction::equation, R"doc(
Returns:
  str: The chemical equation of the reaction.

Examples:
  .. code-block:: python

    >> from kintera import Reaction
    >> op = Reaction("H2 + O2 => H2O2")
    >> print(op.equation())
    H2 + O2 => H2O2
    )doc")

      .ADD_OPTION(kintera::Composition, kintera::Reaction, reactants, R"doc(
Set or get the reactants of the reaction.

Args:
  value (map(str,float)): The reactants of the reaction.

Returns:
  Reaction | map(str, float): class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import Reaction
    >> op = Reaction("H2 + O2 => H2O2")
    >> print(op.reactants())
    {'H2': 1.0, 'O2': 1.0}
    )doc")

      .ADD_OPTION(kintera::Composition, kintera::Reaction, products, R"doc(
Set or get the products of the reaction.

Args:
  value (map(str,float)): The products of the reaction.

Returns:
  Reaction | map(str, float): class object if argument is not empty, otherwise sets the value

Examples:
  .. code-block:: python

    >> from kintera import Reaction
    >> op = Reaction("H2 + O2 => H2O2")
    >> print(op.products())
    {'H2O2': 1.0}
    )doc");
}
