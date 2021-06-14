# Commands: clean dox cov libia libd play playc test testc bench benchc

# Auxillary parameters
BROWSER := ns
# BROWSER := ck
PWD := $(shell pwd)

CXX := g++
# CXX := /usr/bin/g++

# Flags
CXXFLAGS += -DLINUX
CXXFLAGS += -std=c++11
CXXFLAGS += -Wno-write-strings
# CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
# CXXFLAGS += -lpthread
# CXXFLAGS += -Wall # Enable the 'all' set of warnings
# CXXFLAGS += -Werror=unused-variable # Treat all warnings as error
# CXXFLAGS += -Wshadow # Warn when shadowing variables
# CXXFLAGS += -Wextra # Enable additional warnings

CUDA_FLAGS = -gencode arch=compute_61,code=compute_61 --machine 64 -ccbin=$(CXX)

# Search paths
INCS := -Iinclude # Location of header files
INCS += -I/usr/local/cuda-11.0/include # Cuda headers
INCS += -L/usr/local/cuda-11.0/lib64 # Cuda libraries
INCS += -I./../Celero/include # Celero headers (benchmarking)
INCS += -L./../Celero/build # Celero libraries (benchmarking)

# Libraries
LDLIBS := $(shell pkg-config --libs libadq) # Digitiser library
LDLIBS += -lcudart # Cuda runtime (high level functions)

# Source and object files evaluted from the BUILD_SOURCE that will need to be supplied
BUILD_FOLDER := ./build/${BUILD_SOURCE}
CPP_SOURCE := $(wildcard $(BUILD_SOURCE)/*.cpp)
CPP_OBJECTS := $(patsubst $(BUILD_SOURCE)/%.cpp,$(BUILD_FOLDER)/%.o,$(CPP_SOURCE))
CUDA_SOURCE := $(wildcard $(BUILD_SOURCE)/*.cu)
CUDA_OBJECTS := $(patsubst $(BUILD_SOURCE)/%.cu,$(BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)

.PHONY: fresh build test bench

fresh:
	@echo "🗑  Cleaning build objects"
	@rm -rfv *.out ./build/*/*.o *.gcda *.gcno *.log *.o
	@echo "✅ [***] Done"
	@echo

clean:
	@echo "🗑  Cleaning project!"
	@rm -rfv ./bin ./html ./build ./latex ./coverage *.log *.o
	@echo "✅ [***] Done"
	@echo

###############################################################################
# Generic templates for building.
# - Run with @make build
# - pass in BUILD_SOURCE, BUILD_FLAGS, BUILD_LDLIBS, BUILD_BIN, EXTRA_BIN_FLAGS, EXTRA_CXX_FLAGS
###############################################################################
make_folders:
	@mkdir -p $(BUILD_FOLDER) ./build ./bin ./dump

# Triggers the building of cpp and cu files (see *)
# Otherwise dummy recipe
build_objects: $(OBJECTS)
ifdef BUILD_SOURCE
	@echo "✅ [***] Built files in \"$(BUILD_SOURCE)\""
	@echo
endif
	@:

# Before making binary, always build the core code in the ./src directory
build_core_objects:
	@make --no-print-directory \
		BUILD_SOURCE="./src" \
		build_objects

# Building of cpp object files (*)
# PositionIndependentCode so that it can be compiled into shared library
$(BUILD_FOLDER)/%.o: $(BUILD_SOURCE)/%.cpp | make_folders
	@echo 🍕 [cpp] Building $^ ⟶ $@
	@$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) $(BUILD_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -c -o $@ $^

# Building of cuda object files
$(BUILD_FOLDER)/%.o: $(BUILD_SOURCE)/%.cu | make_folders
	@echo 🐍 [cu ] Building $^ ⟶ $@
	@nvcc $(CUDA_FLAGS) $(INCS) $(BUILD_FLAGS) --compiler-options -fPIC -c -o $@ $^

# Collect up all the built files
compile_binary:
	@echo "🏗  [***] Building binary file $(BUILD_BIN)"
	@echo "Using flags: $(BUILD_FLAGS)"
	@echo
	@$(CXX) \
		$(CXXFLAGS) \
		$(INCS) \
		$(LDLIBS) \
		$(BUILD_LDLIBS) \
		$(EXTRA_BIN_FLAGS) \
		-o $(BUILD_BIN) \
		$(wildcard build/$(BUILD_SOURCE)/*.o) $(wildcard build/src/*.o)
	@echo "✅ [***] Built binary $(BUILD_BIN)"
	@echo

# Main entrypoint for building. Builds the target, core objects and compiles them
build:
	@make --no-print-directory build_objects
	@make --no-print-directory build_core_objects
	@make --no-print-directory compile_binary

###############################################################################
#                              Single executables                             #
###############################################################################
PLAY_SOURCE := ./playground

# PLAY_FLAGS := -D R_POINTS=254000
PLAY_FLAGS := -D R_POINTS=2048
PLAY_FLAGS += -D R_POINTS_PER_GPU_CHUNK=1024
PLAY_FLAGS += -D SP_POINTS=10
# PLAY_FLAGS += -D DEBUG=1
# PLAY_FLAGS += -D TESTENV

PLAY_BIN := ./bin/run

playc: | fresh play
play:
	@make --no-print-directory \
		BUILD_SOURCE="$(PLAY_SOURCE)" \
		BUILD_FLAGS="$(PLAY_FLAGS)" \
		BUILD_LDLIBS="$(PLAY_LDLIBS)" \
		BUILD_BIN="$(PLAY_BIN)" \
		build
	$(PLAY_RUN) $(PLAY_BIN)

###############################################################################
#                                    Tests                                    #
###############################################################################
TEST_SOURCE := ./test

TEST_FLAGS := -DTESTENV
TEST_FLAGS += -D R_POINTS=8
TEST_FLAGS += -D R_POINTS_PER_GPU_CHUNK=2
TEST_FLAGS += -D SP_POINTS=3
TEST_FLAGS += -D DEBUG=1
TEST_FLAGS += -D G1_DIGITISER_POINTS=101

TEST_EXTRA_CXX_FLAGS := -fprofile-arcs -ftest-coverage

TEST_EXTRA_BIN_FLAGS := -lgcov -fprofile-arcs

TEST_LDLIBS := $(shell pkg-config --libs cppunit) # Unittest library

TEST_BIN := ./bin/test

# TEST_RUN=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64 cuda-memcheck --tool racecheck

testc: | fresh test;
test:
	@make --no-print-directory \
		BUILD_SOURCE="$(TEST_SOURCE)" \
		BUILD_FLAGS="$(TEST_FLAGS)" \
		EXTRA_CXX_FLAGS="$(TEST_EXTRA_CXX_FLAGS)" \
		EXTRA_BIN_FLAGS="$(TEST_EXTRA_BIN_FLAGS)" \
		BUILD_LDLIBS="$(TEST_LDLIBS)" \
		BUILD_BIN="$(TEST_BIN)" \
		build
	$(TEST_RUN) $(TEST_BIN)

# Coverage report from the tests ##############################################
CORE_BUILD_FOLDER := ./build/src
TEST_BUILD_FOLDER := ./build/$(TEST_SOURCE)

COVERAGE_FOLDER := ./coverage
COVERAGE_FILE := $(COVERAGE_FOLDER)/coverage.info

# Add it as an order-only prerequisite | coverage-wipe
# As coverage is done by injecting extra code to keep track of execution
# we need to delete .o files that already exist to force rebuilding
# and remove them after coverage.
coverage-wipe:
	@rm -rf $(CORE_BUILD_FOLDER)/*.o
	@echo "🗑  Removed all the .o files before/after coverage report!"
	@echo

coverage_report:
	@mkdir -p $(COVERAGE_FOLDER)
	@gcov --branch-probabilities -o $(CORE_BUILD_FOLDER) $(wildcard $(CORE_BUILD_FOLDER)/*.o)
	# Generate coverage report
	@lcov --capture --base-directory . --directory $(CORE_BUILD_FOLDER) --output $(COVERAGE_FILE)
	# Clean from std libraries and test functions
	@lcov --remove $(COVERAGE_FILE) "/usr*" --output $(COVERAGE_FILE)
	@lcov --remove $(COVERAGE_FILE) "*_test.cpp" --output $(COVERAGE_FILE)
	@lcov --remove $(COVERAGE_FILE) "*test_runner.cpp" --output $(COVERAGE_FILE)
	@genhtml --output coverage -t "🐋 Test Coverage" $(COVERAGE_FILE)
	@rm -rf *.log
	@${BROWSER} ${PWD}/coverage/index.html &

cov: | coverage-wipe
	@make --no-print-directory test
	@make --no-print coverage_report
	@make --no-print coverage-wipe

###############################################################################
#                              Speed benchmarking                             #
###############################################################################
BENCH_SOURCE := ./bench

BENCH_FLAGS := -DTESTENV
BENCH_FLAGS += -D R_POINTS=128000 # 129055 for 400 points, 254200 for 200 points
BENCH_FLAGS += -D SP_POINTS=400
BENCH_FLAGS += -D R_POINTS_PER_GPU_CHUNK=1000
BENCH_FLAGS += -D DEBUG=0

BENCH_LDLIBS := -lcelero # Bechmarking library

BENCH_BIN := ./bin/bench

#This is used during runtime to locate Celero. cuda-memcheck can be used to audit the GPU for errors
BENCH_RUN=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64 #cuda-memcheck

benchc: | fresh bench;
bench:
	@make --no-print-directory \
		BUILD_SOURCE="$(BENCH_SOURCE)" \
		BUILD_FLAGS="$(BENCH_FLAGS)" \
		BUILD_LDLIBS="$(BENCH_LDLIBS)" \
		BUILD_BIN="$(BENCH_BIN)" \
		build
	$(BENCH_RUN) $(BENCH_BIN)

###############################################################################
#         Library for the digitiser - only a few function are exported        #
###############################################################################
LIBADQ_BIN := ./bin/ADQAPIia.so
LIBADQ_SOURCE := ./src/sp_digitiser.cpp ./src/logging.cpp

libadq: | make_folders
	@echo "📡  [***] Building digitiser library $(LIBADQ_BIN)"
	$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -fPIC -shared -o $(LIBADQ_BIN) $(LIBADQ_SOURCE)
	@echo "📡  [***] Done"

###############################################################################
#                                   Library                                   #
###############################################################################
LIBIA_FLAGS := -D PYTHON=1
# LIBIA_FLAGS += -D TESTENV=1

# LIBIA_FLAGS passed in like so:
# R_POINTS=1000 SP_POINTS=400 R_POINTS_PER_GPU_CHUNK=1 make libia
ifeq (libia, $(firstword $(MAKECMDGOALS)))
ifndef R_POINTS
$(error "❌ Need to define R_POINTS for building library")
endif
ifndef SP_POINTS
$(error "❌ Need to define SP_POINTS for building library")
endif
ifndef R_POINTS_PER_GPU_CHUNK
$(error "❌ Need to define R_POINTS_PER_GPU_CHUNK for building library")
endif

LIBIA_FLAGS += -D SP_POINTS=$(SP_POINTS) -D R_POINTS=$(R_POINTS) -D R_POINTS_PER_GPU_CHUNK=$(R_POINTS_PER_GPU_CHUNK)
endif

LIBIA_EXTRA_BIN_FLAGS := "-shared"

LIBIA_BIN := ./bin/libia.so

# Library is always built from scratch, in order to propagate the -D arguments
libia: | fresh
	@echo "Using flags: $(LIBIA_FLAGS)"
	@echo
	@BUILD_FLAGS="$(LIBIA_FLAGS)" make --no-print-directory \
			EXTRA_BIN_FLAGS=$(LIBIA_EXTRA_BIN_FLAGS) \
			BUILD_BIN=$(LIBIA_BIN) \
			build

###############################################################################
#                                Documentation                                #
###############################################################################
dox:
	@doxygen
	@$(BROWSER) html/index.html
