# CXX := /usr/bin/g++
CXX := g++

CXXFLAGS += -DLINUX
CXXFLAGS += -std=c++11
# CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
# CXXFLAGS += -lpthread
# CXXFLAGS += -Wall # Enable the 'all' set of warnings
# CXXFLAGS += -Werror=unused-variable # Treat all warnings as error
# CXXFLAGS += -Wshadow # Warn when shadowing variables
# CXXFLAGS += -Wextra # Enable additional warnings

CUDA_FLAGS = -gencode arch=compute_61,code=compute_61 --machine 64 -ccbin=$(CXX)

TFLAGS := -DTESTENV
TFLAGS += -D R_POINTS=3
TFLAGS += -D NP_POINTS=4
TFLAGS += -D PROCESSING_ARRAY_TYPE=int
TFLAGS += -D THREADS_PER_BLOCK=1024

LDLIBS := $(shell pkg-config --libs libadq) # Digitiser library
LDLIBS += -lcudart # Cuda runtime (high level functions)

TLDLIBS += -lcelero
TLDLIBS += $(shell pkg-config --libs cppunit) # Unittest library

BENCHMARKING=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64 #cuda-memcheck

INCS := -Iincludes # Location of header files
INCS += -I/usr/local/cuda-11.0/include # Cuda headers
INCS += -L/usr/local/cuda-11.0/lib64 # Cuda libraries
INCS += -I./../Celero/include # Celero headers
INCS += -L./../Celero/build # Celero libraries

#BROWSER := ns
BROWSER := ck
PWD := $(shell pwd)

# Binaries
BIN_TEST=./bin/test
BIN_APP=./bin/run
BIN_LIB=./bin/ia_1488.so

# Folders
SOURCE_FOLDER:=./src
BUILD_FOLDER:=./build
TEST_FOLDER:=./test
TEST_BUILD_FOLDER:=./build/test_build
COVERAGE_FOLDER:=./coverage

# Aux files
COVERAGE_FILE:=$(COVERAGE_FOLDER)/coverage.info

# Source and object files
# CPP_SOURCE := ./src/ia_ADQAPI.cpp
CPP_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.cpp)
CPP_OBJECTS := $(patsubst $(SOURCE_FOLDER)/%.cpp,$(BUILD_FOLDER)/%.o,$(CPP_SOURCE))

# C_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.c)
# C_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.c,$(BUILD_FOLDER)/%.o,$(C_SOURCE))

CUDA_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.cu)
CUDA_OBJECTS := $(patsubst $(SOURCE_FOLDER)/%.cu,$(BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS) $(C_OBJECTS)

# TEST_SOURCE := $(wildcard $(TEST_FOLDER)/*.cpp)
TEST_SOURCE = ./test/test_runner.cpp ./test/utils_test.cpp
# TEST_SOURCE := $(TEST_FOLDER)/power_gpu_test.cpp $(TEST_FOLDER)/test_runner.cpp
# TEST_SOURCE := $(TEST_FOLDER)/power_gpu_utils_test.cpp $(TEST_FOLDER)/test_runner.cpp
# TEST_SOURCE := $(TEST_FOLDER)/playground_test.cpp $(TEST_FOLDER)/test_runner.cpp
TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.cpp,$(TEST_BUILD_FOLDER)/%.o,$(TEST_SOURCE))
TEST_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.cpp,$(TEST_BUILD_FOLDER)/%.o,$(CPP_SOURCE))
TEST_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.cu,$(TEST_BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

.PHONY: clean test


play:
	$(CXX) $(CXXFLAGS) $(LDLIBS) $(INCS)  -o $(BIN_APP) bench.cpp
	$(BENCHMARKING) $(BIN_APP)

###############################################################################
#                                   Library                                   #
###############################################################################
# PositionIndependentCode
# Building of cpp object files
$(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cpp
		@echo 🐋 [cpp] Building $^ ⟶ $@
		$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -fPIC -c -o $@ $^

# Building of c object files
# $(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.c
#		@echo 🐋 [c] Building $^ ⟶ $@
#		$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -fPIC -c -o $@ $^

# Building of cuda object files
$(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	@echo 🐍 [cu] Building $^ ⟶ $@
	@nvcc $(CUDA_FLAGS) $(INCS) --compiler-options -fPIC -c -o $@ $^

# src/ia_ADQAPI.cpp
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
		$(TFLAGS) -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) $(TLDLIBS) \
		-c -o $@ $^

$(TEST_BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cpp
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [cpp] file: $^ ⟶ $@
	@$(CXX) \
		$(TFLAGS) -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) $(TLDLIBS) \
		-c -o $@ $^

$(TEST_BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [cuda] file: $^ ⟶ $@
	@nvcc $(TFLAGS) $(CUDA_FLAGS) $(INCS) -c -o $@ $^

test: $(TEST_OBJECTS)
	@echo 🤖 Linking files and running test
	@$(CXX) \
		$(TFLAGS) -fprofile-arcs -ftest-coverage -lgcov \
		$(CXXFLAGS) $(INCS) $(LDLIBS) $(TLDLIBS)\
		-o $(BIN_TEST) $^ -pthread
	$(BENCHMARKING) $(BIN_TEST)

post-test: | test
	@rm -rf $(TEST_BUILD_FOLDER)/*.o
	@echo "⚰  Removed all test .o files after test!"

# Run pre-test to remove .o files for coverage information.
# Add it as an order-only prerequisite | pre-test
pre-test:
	@rm -rf $(TEST_BUILD_FOLDER)/*.o
	@echo "⚰  Removed all test .o files before test!"

cov: $(TEST_OBJECTS) | pre-test link-and-run-test
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
		@rm -rfv *.out $(TEST_BUILD_FOLDER) $(BUILD_FOLDER)/*.o *.gcda *.gcno *.log *.o ./bin/*
