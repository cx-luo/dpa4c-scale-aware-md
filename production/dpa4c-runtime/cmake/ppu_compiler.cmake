# The pinned PPU wrapper forwards a comma-containing -Wl payload as one
# linker argument. GNU long options with = preserve the intended ELF metadata.
foreach(language CXX CUDA)
  set(CMAKE_SHARED_LIBRARY_SONAME_${language}_FLAG "-Wl,--soname=")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${language}_FLAG "-Wl,--rpath=")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${language}_FLAG_SEP ":")
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_${language}_FLAG "-Wl,--rpath-link=")
  set(CMAKE_EXECUTABLE_RUNTIME_${language}_FLAG "-Wl,--rpath=")
  set(CMAKE_EXECUTABLE_RUNTIME_${language}_FLAG_SEP ":")
  set(CMAKE_EXECUTABLE_RPATH_LINK_${language}_FLAG "-Wl,--rpath-link=")
endforeach()
