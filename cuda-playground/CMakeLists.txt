add_executable(p1 p1.cu utils.cu utils.h)
set_target_properties(p1 PROPERTIES CUDA_ARCHITECTURES "native")
set_property(TARGET p1 PROPERTY CXX_STANDARD 20)
