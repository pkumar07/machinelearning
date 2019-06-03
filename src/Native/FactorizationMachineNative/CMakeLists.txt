project (FactorizationMachineNative)

set(SOURCES
    FactorizationMachineCoreSse.cpp
    FactorizationMachineCoreAvx.cpp
    FactorizationMachineCoreFma.cpp
)

if(WIN32)
else()
    list(APPEND SOURCES ${VERSION_FILE_PATH})
    set_property(SOURCE FactorizationMachineCoreAvx.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx")
    set_property(SOURCE FactorizationMachineCoreFma.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mfma")
endif()

add_library(FactorizationMachineNative SHARED ${SOURCES} ${RESOURCES})

install_library_and_symbols (FactorizationMachineNative)