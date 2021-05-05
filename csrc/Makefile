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

# Folders
FOLDER_SOURCE:=./src
FOLDER_BUILD:=./build

# Binaries
BIN_APP=./bin/run
BIN_LIB=./bin/ia_1488.so

# Source and object files
# CPP_SOURCE := ./src/ia_ADQAPI.cpp #example
CPP_SOURCE := $(wildcard $(FOLDER_SOURCE)/*.cpp)
CPP_OBJECTS := $(patsubst $(FOLDER_SOURCE)/%.cpp,$(FOLDER_BUILD)/%.o,$(CPP_SOURCE))
CUDA_SOURCE := $(wildcard $(FOLDER_SOURCE)/*.cu)
CUDA_OBJECTS := $(patsubst $(FOLDER_SOURCE)/%.cu,$(FOLDER_BUILD)/%.o,$(CUDA_SOURCE))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)

# Auxillary parameters
#BROWSER := ns
BROWSER := ck
PWD := $(shell pwd)

# Tests and benchmarks #######################################################################
TEST_FLAGS := -DTESTENV
TEST_FLAGS += -D R_POINTS=4
TEST_FLAGS += -D NP_POINTS=3
TEST_FLAGS += -D PROCESSING_ARRAY_TYPE=int
TEST_FLAGS += -D THREADS_PER_BLOCK=1024

TEST_LDLIBS += $(shell pkg-config --libs cppunit) # Unittest library

TEST_FOLDER:=./test
TEST_BUILD_FOLDER:=./build/test_build
COVERAGE_FOLDER:=./coverage
COVERAGE_FILE:=$(COVERAGE_FOLDER)/coverage.info

TEST_BIN=./bin/test

# TEST_SOURCE := $(TEST_FOLDER)/power_gpu_test.cpp $(TEST_FOLDER)/test_runner.cpp #example
TEST_SOURCE = $(wildcard $(TEST_FOLDER)/*.cpp)
TEST_OBJECTS = $(patsubst $(TEST_FOLDER)/%.cpp,$(TEST_BUILD_FOLDER)/%.o,$(TEST_SOURCE))
TEST_OBJECTS += $(patsubst $(FOLDER_SOURCE)/%.cpp,$(TEST_BUILD_FOLDER)/%.o,$(CPP_SOURCE))
TEST_OBJECTS += $(patsubst $(FOLDER_SOURCE)/%.cu,$(TEST_BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

# Benchmarking speed of CPU vs GPU ############################################
BENCH_FLAGS := -DTESTENV
BENCH_FLAGS += -D R_POINTS=1000
BENCH_FLAGS += -D NP_POINTS=100
BENCH_FLAGS += -D PROCESSING_ARRAY_TYPE=int
BENCH_FLAGS += -D THREADS_PER_BLOCK=1024
BENCH_FLAGS += -D SAMPLES=10
BENCH_FLAGS += -D ITERATIONS=10000

BENCH_LDLIBS += -lcelero # Bechmarking library
#This is used during runtime to locate Celero. cuda-memcheck can be used to audit the GPU for errors
BENCH_RUN=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64 #cuda-memcheck

BENCH_FOLDER:=./bench
BENCH_BUILD_FOLDER:=./build/bench_build

BENCH_BIN=./bin/bench

.PHONY: clean test bench

play:
	$(CXX) $(CXXFLAGS) $(BENCH_FLAGS) $(BENCH_LDLIBS) $(INCS) -o $(BIN_APP) bench.cpp
	$(BENCH_RUN) $(BIN_APP)

###############################################################################
#           Same as for test, but with replaced flags and libraries           #
###############################################################################
bench:
	@make \
		TEST_FLAGS="$(BENCH_FLAGS)" \
		TEST_LDLIBS="$(BENCH_LDLIBS)" \
		TEST_FOLDER="$(BENCH_FOLDER)" \
		TEST_BUILD_FOLDER="$(BENCH_BUILD_FOLDER)" \
		TEST_BIN="$(BENCH_BIN)" \
		TEST_RUN="$(BENCH_RUN)" \
		test

###############################################################################
#                                   Library                                   #
###############################################################################
# PositionIndependentCode
# Building of cpp object files
$(FOLDER_BUILD)/%.o: $(FOLDER_SOURCE)/%.cpp
		@echo 🐋 [cpp] Building $^ ⟶ $@
		$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -fPIC -c -o $@ $^

# Building of cuda object files
$(FOLDER_BUILD)/%.o: $(FOLDER_SOURCE)/%.cu
	@echo 🐍 [cu] Building $^ ⟶ $@
	@nvcc $(CUDA_FLAGS) $(INCS) --compiler-options -fPIC -c -o $@ $^

lib: $(OBJECTS)
		$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -shared -o $(BIN_LIB) $^

###############################################################################
#                                     Test                                    #
###############################################################################
# First build all the object files for the test ###############################
$(TEST_BUILD_FOLDER)/%.o: $(TEST_FOLDER)/%.cpp
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [test] file: $^ ⟶ $@
	@$(CXX) \
		$(TEST_FLAGS) -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) $(TEST_LDLIBS) \
		-c -o $@ $^

$(TEST_BUILD_FOLDER)/%.o: $(FOLDER_SOURCE)/%.cpp
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [cpp] file: $^ ⟶ $@
	@$(CXX) \
		$(TEST_FLAGS) -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) $(TEST_LDLIBS) \
		-c -o $@ $^

$(TEST_BUILD_FOLDER)/%.o: $(FOLDER_SOURCE)/%.cu
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [cuda] file: $^ ⟶ $@
	@nvcc $(TEST_FLAGS) $(CUDA_FLAGS) $(INCS) -c -o $@ $^

test: $(TEST_OBJECTS)
	@echo 🤖 Linking files and running test
	@$(CXX) \
		$(TEST_FLAGS) -fprofile-arcs -ftest-coverage -lgcov \
		$(CXXFLAGS) $(INCS) $(LDLIBS) $(TEST_LDLIBS)\
		-o $(TEST_BIN) $^ -pthread
	$(TEST_RUN) $(TEST_BIN)

# Run pre-test to remove .o files for coverage information.
# Add it as an order-only prerequisite | pre-test
pre-test:
	@rm -rf $(TEST_BUILD_FOLDER)/*.o
	@echo "⚰  Removed all test .o files before test!"

cov: $(TEST_OBJECTS) | test pre-test
				@mkdir -p $(COVERAGE_FOLDER)
				@gcov --branch-probabilities -o $(TEST_BUILD_FOLDER) $^
				# Generate coverage report
				@lcov --capture --base-directory . --directory $(TEST_BUILD_FOLDER) --output $(COVERAGE_FILE)
				# Clean from std libraries and test functions
				@lcov --remove $(COVERAGE_FILE) "/usr*" --output $(COVERAGE_FILE)
				@lcov --remove $(COVERAGE_FILE) "*_test.cpp" --output $(COVERAGE_FILE)
				@lcov --remove $(COVERAGE_FILE) "*test_runner.cpp" --output $(COVERAGE_FILE)
				@genhtml --output coverage -t "🐋 Test Coverage" $(COVERAGE_FILE)
				@rm -rf *.log
				@echo "Launching in browser"
				@${BROWSER} ${PWD}/coverage/index.html &

clean:
		@echo "🖌 Cleaning build objects"
		@rm -rfv *.out $(BENCH_BUILD_FOLDER) $(TEST_BUILD_FOLDER) $(FOLDER_BUILD)/*.o *.gcda *.gcno *.log *.o ./bin/*
