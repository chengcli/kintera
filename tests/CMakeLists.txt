# ==========================
# A collection of unit tests
# ==========================

enable_testing()

setup_problem(test_yaml_parse)
setup_problem(test_kinetics_rates)

setup_test(test_eos)
setup_test(test_thermo)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${inp})
endforeach()
