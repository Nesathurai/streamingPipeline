# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sc/streamingPipeline

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sc/streamingPipeline/build

# Include any dependencies generated for this target.
include CMakeFiles/mergePC.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mergePC.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mergePC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mergePC.dir/flags.make

CMakeFiles/mergePC.dir/src/mergePC.cpp.o: CMakeFiles/mergePC.dir/flags.make
CMakeFiles/mergePC.dir/src/mergePC.cpp.o: ../src/mergePC.cpp
CMakeFiles/mergePC.dir/src/mergePC.cpp.o: CMakeFiles/mergePC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mergePC.dir/src/mergePC.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergePC.dir/src/mergePC.cpp.o -MF CMakeFiles/mergePC.dir/src/mergePC.cpp.o.d -o CMakeFiles/mergePC.dir/src/mergePC.cpp.o -c /home/sc/streamingPipeline/src/mergePC.cpp

CMakeFiles/mergePC.dir/src/mergePC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mergePC.dir/src/mergePC.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sc/streamingPipeline/src/mergePC.cpp > CMakeFiles/mergePC.dir/src/mergePC.cpp.i

CMakeFiles/mergePC.dir/src/mergePC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mergePC.dir/src/mergePC.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sc/streamingPipeline/src/mergePC.cpp -o CMakeFiles/mergePC.dir/src/mergePC.cpp.s

CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o: CMakeFiles/mergePC.dir/flags.make
CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o: ../src/kinect_capture.cpp
CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o: CMakeFiles/mergePC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o -MF CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o.d -o CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o -c /home/sc/streamingPipeline/src/kinect_capture.cpp

CMakeFiles/mergePC.dir/src/kinect_capture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mergePC.dir/src/kinect_capture.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sc/streamingPipeline/src/kinect_capture.cpp > CMakeFiles/mergePC.dir/src/kinect_capture.cpp.i

CMakeFiles/mergePC.dir/src/kinect_capture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mergePC.dir/src/kinect_capture.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sc/streamingPipeline/src/kinect_capture.cpp -o CMakeFiles/mergePC.dir/src/kinect_capture.cpp.s

CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o: CMakeFiles/mergePC.dir/flags.make
CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o: ../src/camera_alignment.cpp
CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o: CMakeFiles/mergePC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o -MF CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o.d -o CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o -c /home/sc/streamingPipeline/src/camera_alignment.cpp

CMakeFiles/mergePC.dir/src/camera_alignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mergePC.dir/src/camera_alignment.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sc/streamingPipeline/src/camera_alignment.cpp > CMakeFiles/mergePC.dir/src/camera_alignment.cpp.i

CMakeFiles/mergePC.dir/src/camera_alignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mergePC.dir/src/camera_alignment.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sc/streamingPipeline/src/camera_alignment.cpp -o CMakeFiles/mergePC.dir/src/camera_alignment.cpp.s

CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o: CMakeFiles/mergePC.dir/flags.make
CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o: ../src/multi_kinect_capture.cpp
CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o: CMakeFiles/mergePC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o -MF CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o.d -o CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o -c /home/sc/streamingPipeline/src/multi_kinect_capture.cpp

CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sc/streamingPipeline/src/multi_kinect_capture.cpp > CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.i

CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sc/streamingPipeline/src/multi_kinect_capture.cpp -o CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.s

CMakeFiles/mergePC.dir/src/util.cpp.o: CMakeFiles/mergePC.dir/flags.make
CMakeFiles/mergePC.dir/src/util.cpp.o: ../src/util.cpp
CMakeFiles/mergePC.dir/src/util.cpp.o: CMakeFiles/mergePC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/mergePC.dir/src/util.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergePC.dir/src/util.cpp.o -MF CMakeFiles/mergePC.dir/src/util.cpp.o.d -o CMakeFiles/mergePC.dir/src/util.cpp.o -c /home/sc/streamingPipeline/src/util.cpp

CMakeFiles/mergePC.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mergePC.dir/src/util.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sc/streamingPipeline/src/util.cpp > CMakeFiles/mergePC.dir/src/util.cpp.i

CMakeFiles/mergePC.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mergePC.dir/src/util.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sc/streamingPipeline/src/util.cpp -o CMakeFiles/mergePC.dir/src/util.cpp.s

CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o: CMakeFiles/mergePC.dir/flags.make
CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o: ../src/texture_mapping.cpp
CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o: CMakeFiles/mergePC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o -MF CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o.d -o CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o -c /home/sc/streamingPipeline/src/texture_mapping.cpp

CMakeFiles/mergePC.dir/src/texture_mapping.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mergePC.dir/src/texture_mapping.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sc/streamingPipeline/src/texture_mapping.cpp > CMakeFiles/mergePC.dir/src/texture_mapping.cpp.i

CMakeFiles/mergePC.dir/src/texture_mapping.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mergePC.dir/src/texture_mapping.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sc/streamingPipeline/src/texture_mapping.cpp -o CMakeFiles/mergePC.dir/src/texture_mapping.cpp.s

# Object files for target mergePC
mergePC_OBJECTS = \
"CMakeFiles/mergePC.dir/src/mergePC.cpp.o" \
"CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o" \
"CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o" \
"CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o" \
"CMakeFiles/mergePC.dir/src/util.cpp.o" \
"CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o"

# External object files for target mergePC
mergePC_EXTERNAL_OBJECTS =

mergePC: CMakeFiles/mergePC.dir/src/mergePC.cpp.o
mergePC: CMakeFiles/mergePC.dir/src/kinect_capture.cpp.o
mergePC: CMakeFiles/mergePC.dir/src/camera_alignment.cpp.o
mergePC: CMakeFiles/mergePC.dir/src/multi_kinect_capture.cpp.o
mergePC: CMakeFiles/mergePC.dir/src/util.cpp.o
mergePC: CMakeFiles/mergePC.dir/src/texture_mapping.cpp.o
mergePC: CMakeFiles/mergePC.dir/build.make
mergePC: /home/sc/open3d_install_0.16.0/lib/libOpen3D.so
mergePC: /usr/local/lib/libapriltag.so.3.2.0
mergePC: /usr/lib/x86_64-linux-gnu/libk4a.so.1.4.1
mergePC: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
mergePC: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
mergePC: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
mergePC: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
mergePC: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
mergePC: CMakeFiles/mergePC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sc/streamingPipeline/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable mergePC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mergePC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mergePC.dir/build: mergePC
.PHONY : CMakeFiles/mergePC.dir/build

CMakeFiles/mergePC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mergePC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mergePC.dir/clean

CMakeFiles/mergePC.dir/depend:
	cd /home/sc/streamingPipeline/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sc/streamingPipeline /home/sc/streamingPipeline /home/sc/streamingPipeline/build /home/sc/streamingPipeline/build /home/sc/streamingPipeline/build/CMakeFiles/mergePC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mergePC.dir/depend

