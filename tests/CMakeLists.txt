# ==========================
# A collection of unit tests
# ==========================

enable_testing()

#setup_problem(test_yaml_parse)
#setup_problem(test_kinetics_rates)

setup_test(test_reaction)
setup_test(test_species)
setup_test(test_thermo)
setup_test(test_kinetics)
setup_test(test_photolysis_options)
setup_test(test_photolysis_kinetics)
setup_test(test_actinic_flux)
setup_test(test_ch4_photolysis)
setup_test(test_chapman_cycle)
setup_test(test_falloff)
setup_test(test_diffusion)
setup_test(test_kinetics_base)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${inp})
endforeach()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/kinetics_base/data/test_master.inp
  ${CMAKE_CURRENT_BINARY_DIR}/kinetics_base/data/test_master.inp COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/kinetics_base/data/test_catalog.dat
  ${CMAKE_CURRENT_BINARY_DIR}/kinetics_base/data/test_catalog.dat COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/kinetics_base/data/test_titan_minimal.pun
  ${CMAKE_CURRENT_BINARY_DIR}/kinetics_base/data/test_titan_minimal.pun COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/kinetics_base/data/test_titan_run.inp
  ${CMAKE_CURRENT_BINARY_DIR}/kinetics_base/data/test_titan_run.inp COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/kinetics_base/data/test_titan_atmosphere.pun
  ${CMAKE_CURRENT_BINARY_DIR}/kinetics_base/data/test_titan_atmosphere.pun
  COPYONLY)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kinetics_base/data/cross
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kinetics_base/data)

# Python tests

file(GLOB pyfiles test*.py)
foreach(pyfile ${pyfiles})
  get_filename_component(name ${pyfile} NAME)
  add_test(NAME ${name} COMMAND pytest ${pyfile})
  set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
