
# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)

## Default to release
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_BUILD_TYPE)

IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)



PROJECT(AdaptiveHistogramMatch)

SET(ITK_DIR D:/ITK_bin/)

FIND_PACKAGE (ITK)
IF (ITK_FOUND)
    INCLUDE (${ITK_USE_FILE})
ENDIF (ITK_FOUND)

IF (NOT ITK_FOUND)
  MESSAGE (FATAL_ERROR "\nFatal Error. ITK not found")
ENDIF (NOT ITK_FOUND)

#-----------------------------------------------------------------------------
#Set any extra compilation flags here
IF(CMAKE_COMPILER_IS_GNUCXX)
  # Add some useful warning flags
  # Note: -Wall and -Wno-deprecated are already pulled in by
  # UseITK.cmake
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wcast-qual")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow -Wcast-qual")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

#-----------------------------------------------------------------------------
#Build the ITK code if necessary or simply add the corresponding include dirs
#INCLUDE_DIRECTORIES(src)
#ADD_SUBDIRECTORY(src)


#SET (Jacobian_Response_SRC
#  Jacobian_Response.cxx
#  itkVectorFieldMagnitude.h
#  itkVectorFieldTrim.h
#  ITKUtils.h
#  ITKUtils.cxx
#  )

ADD_EXECUTABLE(AdaptiveHistogramMatch AdaptiveHistogramMatch.cxx)
TARGET_LINK_LIBRARIES(AdaptiveHistogramMatch ${ITK_LIBRARIES})
INSTALL(TARGETS AdaptiveHistogramMatch DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
