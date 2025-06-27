# ==========================
# A collection of unit tests
# ==========================

enable_testing()

#setup_problem(test_yaml_parse)
#setup_problem(test_kinetics_rates)

setup_test(test_eos)
setup_test(test_thermo)
setup_test(test_kinetics)

file(GLOB inputs *.inp *.dat *.yaml)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${inp})
endforeach()

# Python tests

file(GLOB pyfiles *.py)
foreach(pyfile ${pyfiles})
  get_filename_component(name ${pyfile} NAME)
  message(STATUS "Copying ${pyfile} to ${name}")
  configure_file(${pyfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
  add_test(NAME ${name} COMMAND python3 ${name})
endforeach()
