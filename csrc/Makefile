# Auxillary parameters
#BROWSER := ns
BROWSER := ck
PWD := $(shell pwd)

CXX := g++
# CXX := /usr/bin/g++

# Flags
CXXFLAGS += -DLINUX
CXXFLAGS += -std=c++11
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

.PHONY: clean build test bench

clean:
	@echo "🗑  Cleaning build objects"
	@rm -rfv *.out ./build/*/*.o *.gcda *.gcno *.log *.o ./bin/*
	@echo "✅ [***] Done"
	@echo

###############################################################################
# Generic templates for building.
# - Run with @make build
# - pass in BUILD_SOURCE, BUILD_FLAGS, BUILD_LDLIBS, BUILD_BIN, EXTRA_BIN_FLAGS, EXTRA_CXX_FLAGS
###############################################################################
make_folders:
	@mkdir -p $(BUILD_FOLDER) ./build ./bin

# Triggers the building of cpp and cu files (see *)
build_objects: $(OBJECTS)
	@echo "✅ [***] Built files in $(BUILD_SOURCE)"
	@echo

# Before making binary, always build the core code in the ./src directory
build_core_objects:
	@make --no-print-directory \
		BUILD_SOURCE="./src" \
		build_objects

# Building of cpp object files (*)
# PositionIndependentCode
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
	@$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) $(BUILD_LDLIBS) $(EXTRA_BIN_FLAGS)\
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

PLAY_FLAGS := -D R_POINTS=254000
PLAY_FLAGS += -D R_POINTS_PER_CHUNK=1000
PLAY_FLAGS += -D SP_POINTS=200
PLAY_FLAGS += -D DEBUG=1

PLAY_BIN := ./bin/run

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
TEST_FLAGS += -D R_POINTS_PER_CHUNK=1
TEST_FLAGS += -D SP_POINTS=3
TEST_FLAGS += -D DEBUG=0

TEST_EXTRA_CXX_FLAGS := -fprofile-arcs -ftest-coverage

TEST_EXTRA_BIN_FLAGS := -lgcov -fprofile-arcs

TEST_LDLIBS := $(shell pkg-config --libs cppunit) # Unittest library

TEST_BIN := ./bin/test

# TEST_RUN=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64 cuda-memcheck --tool racecheck

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
BENCH_FLAGS += -D R_POINTS_PER_CHUNK=1000
BENCH_FLAGS += -D DEBUG=0

BENCH_LDLIBS := -lcelero # Bechmarking library

BENCH_BIN := ./bin/bench

#This is used during runtime to locate Celero. cuda-memcheck can be used to audit the GPU for errors
BENCH_RUN=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64 #cuda-memcheck

bench:
	@make --no-print-directory \
		BUILD_SOURCE="$(BENCH_SOURCE)" \
		BUILD_FLAGS="$(BENCH_FLAGS)" \
		BUILD_LDLIBS="$(BENCH_LDLIBS)" \
		BUILD_BIN="$(BENCH_BIN)" \
		build
	$(BENCH_RUN) $(BENCH_BIN)

###############################################################################
#                                   Library                                   #
###############################################################################
LIB_BIN = ./bin/ia_1488.so

LIB_FLAGS += -D R_POINTS=128000 # 129055 for 400 points, 254200 for 200 points
LIB_FLAGS += -D SP_POINTS=400
LIB_FLAGS += -D R_POINTS_PER_CHUNK=1000

lib:
	@make --no-print-directory \
		BUILD_BIN="$(LIB_BIN)" \
		BUILD_FLAGS="$(LIB_FLAGS)" \
		EXTRA_BIN_FLAGS="-shared" \
		build
