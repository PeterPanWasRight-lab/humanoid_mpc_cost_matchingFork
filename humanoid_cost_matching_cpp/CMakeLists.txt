cmake_minimum_required(VERSION 3.10)
project(humanoid_cost_matching_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(rclcpp REQUIRED)

# ---- Boost targets needed by ocs2_core exports ----
find_package(Boost REQUIRED COMPONENTS
  system
  filesystem
  thread
  log
  log_setup
  date_time
  regex
)

find_package(ocs2_oc REQUIRED)
find_package(ocs2_centroidal_model REQUIRED)
find_package(ocs2_core REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(ocs2_robotic_tools REQUIRED)

find_package(humanoid_common_mpc REQUIRED)
find_package(humanoid_centroidal_mpc REQUIRED)

# ---- vendored cnpy (compiled into this library, NOT exported as a separate target) ----
set(CNPY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy)

add_library(${PROJECT_NAME} STATIC
  # src/refpack_logger.cpp
  ${CNPY_DIR}/cnpy.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CNPY_DIR}
)

ament_target_dependencies(${PROJECT_NAME}
  ocs2_core
  ocs2_pinocchio_interface
  ocs2_robotic_tools
  humanoid_common_mpc
  humanoid_centroidal_mpc
)

target_link_libraries(${PROJECT_NAME}
  ZLIB::ZLIB
  Eigen3::Eigen
  Boost::system
  Boost::log
  Boost::log_setup
)

install(DIRECTORY include/ DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_dependencies(
  Eigen3
  ZLIB
  Boost
  ocs2_core
  ocs2_pinocchio_interface
  ocs2_robotic_tools
  humanoid_common_mpc
  humanoid_centroidal_mpc
)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)


# --------- humanoid_cost_matching_offline ------#
add_library(humanoid_cost_matching_offline
  src/offline/OfflineQEvaluator.cpp
  src/offline/core/ZohController.cpp
  src/offline/rollout/FixedGridRK4Rollout.cpp
  src/offline/cost/tapes/cost_tracking_tapes.cpp
  src/offline/cost/tapes/cost_com_tapes.cpp
  src/offline/cost/tapes/cost_swing_tapes.cpp
  src/offline/cost/tapes/cost_base_tapes.cpp
  src/offline/cost/tapes/cost_torque_tapes.cpp
  src/offline/cost/tapes/penalty_joint_tapes.cpp
  src/offline/cost/tapes/penalty_collision_tapes.cpp
  src/offline/cost/tapes/penalty_cop_tapes.cpp
  src/offline/cost/tapes/penalty_zero_velocity_tapes.cpp
  src/offline/cost/tapes/penalty_normal_velocity_tapes.cpp
)

target_include_directories(humanoid_cost_matching_offline PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(humanoid_cost_matching_offline
  ocs2_core
  ocs2_oc
  ocs2_centroidal_model
  humanoid_centroidal_mpc
  humanoid_common_mpc
)

# add_executable(offline_q_eval_main
#   src/tools/offline_q_eval_main.cpp
# )
# target_link_libraries(offline_q_eval_main
#   humanoid_cost_matching_offline
# )

# ament_target_dependencies(offline_q_eval_main
#   ocs2_core
#   ocs2_oc
#   ocs2_centroidal_model
#   humanoid_centroidal_mpc
#   humanoid_common_mpc
# )

# install(TARGETS offline_q_eval_main
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
# )
install(DIRECTORY include/ DESTINATION include)
install(TARGETS humanoid_cost_matching_offline
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(humanoid_cost_matching_offline)


# ----- humanoid_cost_matching_offline_eval_py & pybind11 module ----- #
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11_vendor QUIET)
if(pybind11_vendor_FOUND)
  find_package(pybind11 REQUIRED)
else()
  find_package(pybind11 REQUIRED)
endif()

find_package(ament_cmake_python REQUIRED)
ament_get_python_install_dir(PYTHON_INSTALL_DIR)

pybind11_add_module(humanoid_cost_matching_offline_eval_py
  src/bindings/pybind_offline_q_batch.cpp
)

target_link_libraries(humanoid_cost_matching_offline_eval_py
PRIVATE
  humanoid_cost_matching_cpp
  humanoid_cost_matching_offline
  rclcpp::rclcpp
  ocs2_core::ocs2_core
  ocs2_oc::ocs2_oc
)

target_include_directories(humanoid_cost_matching_offline_eval_py
PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS humanoid_cost_matching_offline_eval_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
)

ament_package()
