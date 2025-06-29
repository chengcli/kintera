#pragma once

// C/C+
#include <string>

#define ADD_OPTION(T, st_name, op_name, doc)                                 \
  def(#op_name, (T const &(st_name::*)() const) & st_name::op_name,          \
      py::return_value_policy::reference, doc)                               \
      .def(#op_name, (st_name & (st_name::*)(const T &)) & st_name::op_name, \
           py::return_value_policy::reference, doc)

#define ADD_KINTERA_MODULE(m_name, op_name, doc, args...)                  \
  torch::python::bind_module<kintera::m_name##Impl>(m, #m_name)            \
      .def(py::init<>(), R"(Construct a new default module.)")             \
      .def(py::init<kintera::op_name>(), "Construct a " #m_name " module", \
           py::arg("options"))                                             \
      .def_readonly("options", &kintera::m_name##Impl::options)            \
      .def("__repr__",                                                     \
           [](const kintera::m_name##Impl &a) {                            \
             return fmt::format(#m_name "{}", a.options);                  \
           })                                                              \
      .def(                                                                \
          "module",                                                        \
          [](kintera::m_name##Impl &self) {                                \
            return py::make_iterator(self.named_modules().begin(),         \
                                     self.named_modules().end());          \
          },                                                               \
          py::keep_alive<0, 1>())                                          \
      .def("get_module",                                                   \
           [](kintera::m_name##Impl &self, std::string name) {             \
             return self.named_modules()[name];                            \
           })                                                              \
      .def(                                                                \
          "buffer",                                                        \
          [](kintera::m_name##Impl &self) {                                \
            return py::make_iterator(self.named_buffers().begin(),         \
                                     self.named_buffers().end());          \
          },                                                               \
          py::keep_alive<0, 1>())                                          \
      .def("get_buffer",                                                   \
           [](kintera::m_name##Impl &self, std::string name) {             \
             return self.named_buffers()[name];                            \
           })                                                              \
      .def("forward", &kintera::m_name##Impl::forward, doc, args)
