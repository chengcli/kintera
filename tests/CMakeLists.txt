# ==========================
# A collection of unit tests
# ==========================

enable_testing()

#setup_problem(test_yaml_parse)
#setup_problem(test_kinetics_rates)

setup_test(test_reaction)
setup_test(test_thermo)
setup_test(test_kinetics)
setup_test(test_photolysis_options)
setup_test(test_photolysis_kinetics)
setup_test(test_actinic_flux)
setup_test(test_ch4_photolysis)
setup_test(test_chapman_cycle)
setup_test(test_falloff)
setup_test(test_diffusion)
setup_test(test_rt)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${inp})
endforeach()

# Python tests

file(GLOB pyfiles test*.py)
foreach(pyfile ${pyfiles})
  get_filename_component(name ${pyfile} NAME)
  message(STATUS "Copying ${pyfile} to ${name}")
  configure_file(${pyfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
  add_test(NAME ${name} COMMAND pytest ${name})
endforeach()
