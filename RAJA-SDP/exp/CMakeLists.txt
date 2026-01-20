add_executable(raja-type-erasure-shared "main-type-erasure-shared.cpp")
add_executable(raja-type-erasure-unique "main-type-erasure-unique.cpp")
set_target_properties(raja-type-erasure-shared PROPERTIES COMPILE_FLAGS "--std=c++11 -g")
set_target_properties(raja-type-erasure-unique PROPERTIES COMPILE_FLAGS "--std=c++11 -g")
