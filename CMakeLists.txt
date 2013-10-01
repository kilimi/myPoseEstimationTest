# PCL dependent
IF(USE_PCL)
   UNSET(ALL_TARGETS)

   # Example app
   ADD_EXECUTABLE(example_or example_or.cpp)
   ADD_EXECUTABLE(my_example myExample.cpp)
   TARGET_LINK_LIBRARIES(example_or ${COVIS_LIBRARIES})
   TARGET_LINK_LIBRARIES(my_example ${COVIS_LIBRARIES})

   LIST(APPEND ALL_TARGETS "example_or")
   LIST(APPEND ALL_TARGETS "my_example")
   
   # GUI app, depends on Nuklei and QVTK
   IF(NUKLEI_FOUND)
      FIND_LIBRARY(PE_QVTK_FOUND QVTK)
      IF(PE_QVTK_FOUND)
         ADD_SUBDIRECTORY(PoseEstimationApp)
         LIST(APPEND ALL_TARGETS "poseestimationapp")
      ENDIF()
   ENDIF()
   
   # Add a target for the whole library
   ADD_CUSTOM_TARGET(PoseEstimationDemos DEPENDS ${ALL_TARGETS})
ENDIF()
