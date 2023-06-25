cmake_minimum_required(VERSION 3.15)
project(ceres_test)

set(CMAKE_CXX_STANDARD 20)

find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

link_directories(${CERES_LIBRARIES})

#Hello World
add_executable(${PROJECT_NAME}_hello_world hello_world.cpp)
target_link_libraries(${PROJECT_NAME}_hello_world ${CERES_LIBRARIES})

# Derivatives
## Numeric Derivatives
add_executable(${PROJECT_NAME}_numeric_derivatives numeric_derivatives.cpp)
target_link_libraries(${PROJECT_NAME}_numeric_derivatives ${CERES_LIBRARIES})
## Analytic Derivatives
add_executable(${PROJECT_NAME}_analytic_derivatives analytic_derivatives.cpp)
target_link_libraries(${PROJECT_NAME}_analytic_derivatives ${CERES_LIBRARIES})

# Powell's Function
add_executable(${PROJECT_NAME}_powells_function powells_function.cpp)
target_link_libraries(${PROJECT_NAME}_powells_function ${CERES_LIBRARIES})