// torch, pybind11
#include <pybind11/stl.h>
#include <torch/extension.h>

// kintera
#include <kintera/thermo/relative_humidity.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module& m) {
  using NO = kintera::NucleationOptionsImpl;
  using TO = kintera::ThermoOptionsImpl;
  using Y = kintera::ThermoYImpl;
  using X = kintera::ThermoXImpl;
  using T = torch::Tensor;
  using py::arg;
  py::class_<NO, kintera::NucleationOptions>(m, "NucleationOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const kintera::NucleationOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("NucleationOptions({})", ss.str());
           })
      .ADD_OPTION(std::vector<double>, NO, minT)
      .ADD_OPTION(std::vector<double>, NO, maxT)
      .ADD_OPTION(std::vector<std::string>, NO, logsvp)
      .ADD_OPTION(std::vector<kintera::Reaction>, NO, reactions);

  py::class_<TO, kintera::SpeciesThermoImpl, kintera::ThermoOptions>(
      m, "ThermoOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const kintera::ThermoOptions& self) {
             std::stringstream ss;
             self->report(ss);
             return fmt::format("ThermoOptions({})", ss.str());
           })
      .def_static("from_yaml",
                  py::overload_cast<std::string const&, bool>(&TO::from_yaml),
                  arg("filename"), arg("verbose") = false)
      .def("reactions", &TO::reactions)
      .ADD_OPTION(double, TO, Tref)
      .ADD_OPTION(double, TO, Pref)
      .ADD_OPTION(kintera::NucleationOptions, TO, nucleation)
      .ADD_OPTION(int, TO, max_iter)
      .ADD_OPTION(double, TO, ftol)
      .ADD_OPTION(std::string, TO, uv_solver);

  py::enum_<kintera::SolveMode>(m, "SolveMode")
      .value("automatic", kintera::SolveMode::automatic)
      .value("eager_exit", kintera::SolveMode::eager_exit)
      .value("fixed", kintera::SolveMode::fixed);

  ADD_KINTERA_MODULE(ThermoY, ThermoOptions, &Y::forward, arg("rho"),
                     arg("intEng"), arg("yfrac"), arg("warm_start") = false,
                     arg("diag") = py::none())
      .def("compute", &Y::compute, arg("ab"), arg("args"))
      .def("take_saturation_adjustment_failures",
           &Y::take_saturation_adjustment_failures)
      .def("failures", &Y::failures)
      .def("reset_failures", &Y::reset_failures)
      .def("consume_failures", &Y::consume_failures)
      .def("compute_into", &Y::compute_into)
      .def_readwrite("solve_mode", &Y::solve_mode)
      .def("prepare",  // device, dtype and cell shape of `r`
           [](Y& s, T r) { s.prepare(r.device(), r.scalar_type(), r.sizes()); })
      .def("tables",  // the epoch of the module's table, -1 before prepare
           [](Y const& s) { return s.tables() ? s.tables()->epoch : -1; })
      .def("alloc_epoch", [](Y const& s) { return kintera::alloc_epoch(s); })
      .def("snapshot_active_set", &Y::snapshot_active_set)
      .def("restore_active_set", &Y::restore_active_set)
      .def("make_active_set_storage", &Y::make_active_set_storage);

  ADD_KINTERA_MODULE(ThermoX, ThermoOptions, &X::forward, arg("temp"),
                     arg("pres"), arg("xfrac"), arg("warm_start") = false,
                     arg("diag") = py::none())
      .def("compute", &X::compute, arg("ab"), arg("args"))
      .def("effective_cp", &X::effective_cp, arg("temp"), arg("pres"),
           arg("xfrac"), arg("gain"), arg("conc") = py::none())
      .def(
          "extrapolate_dlnp",
          [](X& self, T temp, T pres, T xfrac, double dlnp, double ds_dlnp,
             bool rainout, bool verbose) {
            kintera::ExtrapOptions opts;
            opts.dlnp(dlnp).ds_dlnp(ds_dlnp).rainout(rainout).verbose(verbose);
            return self.extrapolate_dlnp(temp, pres, xfrac, opts);
          },
          arg("temp"), arg("pres"), arg("xfrac"), arg("dlnp"),
          arg("ds_dlnp") = 0., arg("rainout") = false, arg("verbose") = false)
      .def(
          "extrapolate_dz",
          [](X& self, T temp, T pres, T xfrac, double dz, double grav,
             double ds_dz, bool rainout, bool verbose) {
            kintera::ExtrapOptions opts;
            opts.dz(dz).grav(grav).ds_dz(ds_dz).rainout(rainout).verbose(
                verbose);
            return self.extrapolate_dz(temp, pres, xfrac, opts);
          },
          arg("temp"), arg("pres"), arg("xfrac"), arg("dz"), arg("grav"),
          arg("ds_dz") = 0., arg("rainout") = false, arg("verbose") = false);

  m.def("relative_humidity", &kintera::relative_humidity, arg("temp"),
        arg("conc"), arg("stoich"), arg("op"), arg("ngas") = -1);
}
