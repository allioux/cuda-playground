add_executable(p1 p1.cu utils.cu utils.h)
set_target_properties(p1 PROPERTIES CUDA_ARCHITECTURES "native")
set_property(TARGET p1 PROPERTY CXX_STANDARD 20)

add_executable(p2 p2.cu utils.cu utils.h)
set_target_properties(p2 PROPERTIES CUDA_ARCHITECTURES "native")
set_property(TARGET p2 PROPERTY CXX_STANDARD 20)

add_executable(p4 p4.cu utils.cu utils.h)
set_target_properties(p4 PROPERTIES CUDA_ARCHITECTURES "native")
set_property(TARGET p4 PROPERTY CXX_STANDARD 20)