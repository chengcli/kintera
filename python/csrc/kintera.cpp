// torch
#include <torch/extension.h>

// kintera
#include <kintera/thermo/thermo.hpp>  // species_names, species_weights
#include <kintera/utils/find_resource.hpp>

namespace py = pybind11;

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
}
