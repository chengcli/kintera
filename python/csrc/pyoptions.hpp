#pragma once

// C/C+
#include <string>

#define ADD_OPTION(T, st_name, op_name)                                      \
  def(#op_name, (T const &(st_name::*)() const) & st_name::op_name,          \
      py::return_value_policy::reference)                                    \
      .def(#op_name, (st_name & (st_name::*)(const T &)) & st_name::op_name, \
           py::return_value_policy::reference)

#define ADD_KINTERA_MODULE(m_name, op_name, args...)                       \
  torch::python::bind_module<kintera::m_name##Impl>(m, #m_name)            \
      .def(py::init<>(), R"(Construct a new default module.)")             \
      .def(py::init<kintera::op_name>(), "Construct a " #m_name " module", \
           py::arg("options"))                                             \
      .def_readonly("options", &kintera::m_name##Impl::options)            \
      .def("__repr__",                                                     \
           [](const kintera::m_name##Impl &a) {                            \
             std::stringstream ss;                                         \
             a.options->report(ss);                                        \
             return fmt::format(#m_name "(\n{})", ss.str());               \
           })                                                              \
      .def("module",                                                       \
           [](kintera::m_name##Impl &self, std::string name) {             \
             return self.named_modules()[name];                            \
           })                                                              \
      .def("buffer",                                                       \
           [](kintera::m_name##Impl &self, std::string name) {             \
             return self.named_buffers()[name];                            \
           })                                                              \
      .def("forward", &kintera::m_name##Impl::forward, args)
