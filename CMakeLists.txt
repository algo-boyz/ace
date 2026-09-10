cmake_minimum_required(VERSION 3.20)
project(face_rec_yolo11 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

find_package(OpenCV REQUIRED)

# Adjust to your ONNX Runtime install (arm64 package includes CoreML)
set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-osx-arm64-1.22.0" CACHE PATH "")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(face_rec
    src/main.cpp
    src/yolo11_face.cpp
    src/arcface.cpp
    src/align.cpp
    src/matcher.cpp
)

target_link_libraries(face_rec
    ${OpenCV_LIBS}
    onnxruntime
)

target_compile_definitions(face_rec PRIVATE USE_COREML)