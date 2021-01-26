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
LDLIBS += $(shell pkg-config --libs cppunit) # Unittest library
LDLIBS += -lcudart # Cuda runtime (high level functions)
LDLIBS += -lcelero

BENCH=LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:../Celero/build:/usr/local/cuda-11.0/lib64

INCS := -Iincludes # Location of header files
INCS += -I/usr/local/cuda-11.0/include # Cuda headers
INCS += -L/usr/local/cuda-11.0/lib64 # Cuda libraries
INCS += -I./../Celero/include # Celeto headers
INCS += -L./../Celero/build # Celeto libraries

#BROWSER := ns
BROWSER := ck
PWD := $(shell pwd)

# Binaries
BIN_TEST=./bin/test
BIN_APP=./bin/run

# Folders
SOURCE_FOLDER:=./src
BUILD_FOLDER:=./build
TEST_FOLDER:=./test
TEST_BUILD_FOLDER:=./build/test_build
COVERAGE_FOLDER:=./coverage

# Aux files
COVERAGE_FILE:=$(COVERAGE_FOLDER)/coverage.info

# Source and object files
CPP_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.cpp)
CPP_OBJECTS := $(patsubst $(SOURCE_FOLDER)/%.cpp,$(BUILD_FOLDER)/%.o,$(CPP_SOURCE))

CUDA_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.cu)
CUDA_OBJECTS := $(patsubst $(SOURCE_FOLDER)/%.cu,$(BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)

# TEST_SOURCE := $(wildcard $(TEST_FOLDER)/*.cpp)
TEST_SOURCE := $(TEST_FOLDER)/power_gpu_test.cpp $(TEST_FOLDER)/test_runner.cpp
TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.cpp,$(TEST_BUILD_FOLDER)/%.o,$(TEST_SOURCE))
TEST_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.cpp,$(TEST_BUILD_FOLDER)/%.o,$(CPP_SOURCE))
TEST_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.cu,$(TEST_BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

.PHONY: clean test

# Just for testing out ########################################################
play:
	$(CXX) $(CXXFLAGS) $(LDLIBS) $(INCS)  -o testrun bench.cpp
	$(BENCH) ./testrun

# nvcc $(CUDA_FLAGS) $(INCS) $(CUDA_SOURCE) -o $(BIN_APP)
# @./$(BIN_APP)

# $(CXX) -DTESTENV -fprofile-arcs -ftest-coverage $(CXXFLAGS) $(LDLIBS) $(INCS) -c test/test_runner.cpp test/utils_gpu_test.cpp
# g++ $(INCS) -c src/utils_gpu.cpp
# nvcc $(INCS) -c src/utils_gpu.cu
# @echo --------------------
# @$(CXX) -DTESTENV -fprofile-arcs -ftest-coverage -lgcov \
# $(LDLIBS) -L/usr/local/cuda-11.0/lib64 -lcudart \
# -o testrun test_runner.o gpu_utils.o utils_gpu_test.o
#nvcc $(LDLIBS) -lgcov -o testrun test_runner.o gpu_utils.o utils_gpu_test.o example_test.o
# ./testrun
#$(CXX) $(LDLIBS) $(INCS) -o testrun test_runner.o gpu_utils.o utils_gpu_test.o

###############################################################################
#                                    Common                                   #
###############################################################################
# Building of cuda object files
$(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	@echo 🐍 Building $^ ⟶ $@
	@nvcc $(CUDA_FLAGS) $(INCS) -c -o $@ $^

# Building of cpp object files
$(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cpp
	@echo 🐋 Building $^ ⟶ $@
	@$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -c -o $@ $^

###############################################################################
#                                   Library                                   #
###############################################################################
lib: src/ia_ADQAPI.cpp
		@$(CXX) -Wall $(CXXFLAGS) $(INCS) $(LDLIBS) -shared -o $(TARGET) -fPIC $^
		@echo Moving $(TARGET) to /usr/lib 🐳
		@sudo cp -f $(TARGET) /usr/lib

###############################################################################
#                                     Test                                    #
###############################################################################
# Run pre-test to remove .o files for coverage information.
# Add it as an order-only prerequisite | pre-test
pre-test:
	@rm -rf $(TEST_BUILD_FOLDER)/*.o
	@echo "⚰  Removed all test .o files before test!"

# First build all the object files for the test ###############################
$(TEST_BUILD_FOLDER)/%.o: $(TEST_FOLDER)/%.cpp
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [test] file: $^ ⟶ $@
	@$(CXX) \
		$(TFLAGS) -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-c -o $@ $^

$(TEST_BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cpp
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [cpp] file: $^ ⟶ $@
	@$(CXX) \
		$(TFLAGS) -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-c -o $@ $^

$(TEST_BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	@mkdir -p $(TEST_BUILD_FOLDER)
	@echo 🤖 Building [cuda] file: $^ ⟶ $@
	@nvcc $(TFLAGS) $(CUDA_FLAGS) $(INCS) -c -o $@ $^
# nvcc -DTESTENV $(INCS) -ccbin=$(CXX) -c -o $@ $^

link-and-run-test: $(TEST_OBJECTS)
	@echo 🤖 Linking files and running test
	@$(CXX) \
		$(TFLAGS) -fprofile-arcs -ftest-coverage -lgcov \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-o $(BIN_TEST) $^ -pthread
	$(BENCH) cuda-memcheck $(BIN_TEST)
# @nvcc -DTESTENV -lgcov $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $(INCS) -o $(BIN_TEST) $^

test: | link-and-run-test
	@rm -rf $(TEST_BUILD_FOLDER)/*.o
	@echo "⚰  Removed all test .o files after test!"

cov: $(TEST_OBJECTS) | test
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
		@rm -rf *.out $(TEST_BUILD_FOLDER) $(BUILD_FOLDER)/*.o *.gcda *.gcno *.log *.o
