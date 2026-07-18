# Wraps glslc -mfmt=c output (a bare {0x...,...} initializer list)
# into a named uint32_t array header. Usage:
#   cmake -DIN=<in.inc> -DOUT=<out.h> -DVAR=<name> -P wrap_spv.cmake
file(READ "${IN}" BODY)
file(WRITE "${OUT}"
     "#pragma once\n#include <cstdint>\n"
     "static const std::uint32_t ${VAR}[] =\n"
     "${BODY};\n")
