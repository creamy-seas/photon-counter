CXX := g++

CUDAFLAGS = -gencode arch=compute_61,code=compute_61 --machine 64

CXXFLAGS += -DLINUX
CXXFLAGS += -std=c++11
# CXXFLAGS += -Wall # Enable the 'all' set of warnings
# CXXFLAGS += -Werror=unused-variable # Treat all warnings as error
# CXXFLAGS += -Wshadow # Warn when shadowing variables
# CXXFLAGS += -Wextra # Enable additional warnings

TFLAGS := -DTESTENV -fprofile-arcs -ftest-coverage

LDFLAGS = $(shell pkg-config --cflags libadq) # Digitiser library
LDLIBS := $(shell pkg-config --libs libadq) # Digitiser library
LDLIBS += $(shell pkg-config --libs cppunit) # Unittest library
LDLIBS += -lcudart # Cuda runtime (high leevel functions)

INCS := -Iincludes # Location of header files
INCS += -I/usr/local/cuda-11.0/include # Cuda headers
INCS += -L/usr/local/cuda-11.0/lib64 # Cuda libraries

#BROWSER := ns
BROWSER := ck
PWD := $(shell pwd)

# Binaries
BIN_TEST=./bin/test
BIN_APP=./bin/run

# Folders
SOURCE_FOLDER=./src
BUILD_FOLDER=./build
TEST_FOLDER=./test
TBUILD_FOLDER=./build/test_build

# Source and object files
CPP_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.cpp)
CPP_OBJECTS := $(patsubst $(SOURCE_FOLDER)/%.cpp,$(BUILD_FOLDER)/%.o,$(CPP_SOURCE))

CUDA_SOURCE := $(wildcard $(SOURCE_FOLDER)/*.cu)
CUDA_OBJECTS := $(patsubst $(SOURCE_FOLDER)/%.cu,$(BUILD_FOLDER)/%.o,$(CUDA_SOURCE))

OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)

TEST_SOURCE := $(wildcard $(TEST_FOLDER)/*.cpp)
TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.cpp,$(TBUILD_FOLDER)/%.o,$(TEST_SOURCE))
TEST_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.cpp,$(TBUILD_FOLDER)/%.o,$(CPP_SOURCE))
TEST_OBJECTS += $(patsubst $(SOURCE_FOLDER)/%.cu,$(TBUILD_FOLDER)/%.o,$(CUDA_SOURCE))

.PHONY: clean test

play:
	$(CXX) -DTESTENV -fprofile-arcs -ftest-coverage $(CXXFLAGS) $(LDLIBS) $(INCS) -c test/test_runner.cpp test/utils_gpu_test.cpp
	g++ $(INCS) -c src/utils_gpu.cpp
	# nvcc $(INCS) -c src/utils_gpu.cu
	@echo --------------------
	@$(CXX) -DTESTENV -fprofile-arcs -ftest-coverage -lgcov \
		$(LDLIBS) -L/usr/local/cuda-11.0/lib64 -lcudart \
		-o testrun test_runner.o utils_gpu.o utils_gpu_test.o
	#nvcc $(LDLIBS) -lgcov -o testrun test_runner.o utils_gpu.o utils_gpu_test.o example_test.o
	./testrun

#$(CXX) $(LDLIBS) $(INCS) -o testrun test_runner.o utils_gpu.o utils_gpu_test.o

# Object files ################################################################
# Building of cuda object files
$(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	@echo 🐍 Building $^ into $@
	nvcc $(INCS) -ccbin=$(CXX) -c -o $@ $^

# Building of cpp object files
$(BUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cpp
	@echo 🐋 Building $^ into $@
	$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -c -o $@ $^

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
# First build all the object files for the test ###############################
$(TBUILD_FOLDER)/%.o: $(TEST_FOLDER)/%.cpp
	@mkdir -p $(TBUILD_FOLDER)
	@echo 🤖🐋 Building $^ into $@
	$(CXX) \
		-DTESTENV -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-c -o $@ $^

$(TBUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cpp
	@mkdir -p $(TBUILD_FOLDER)
	@echo 🤖🐋 Building $^ into $@
	$(CXX) \
		-DTESTENV -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-c -o $@ $^

$(TBUILD_FOLDER)/%.o: $(SOURCE_FOLDER)/%.cu
	@mkdir -p $(TBUILD_FOLDER)
	@echo 🤖🐍 Building $^ into $@
	$(CXX) \
		-DTESTENV -fprofile-arcs -ftest-coverage \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-c -o $@ $^
	# nvcc -DTESTENV $(INCS) -ccbin=$(CXX) -c -o $@ $^

test: $(TEST_OBJECTS)
	# @nvcc -DTESTENV -lgcov $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $(INCS) -o $(BIN_TEST) $^
	@$(CXX) \
		-DTESTENV -fprofile-arcs -ftest-coverage -lgcov \
		$(CXXFLAGS) $(INCS) $(LDLIBS) \
		-o $(BIN_TEST) $^
	@$(BIN_TEST)
	@mkdir -p ./coverage/build
	@mv -f $(TBUILD_FOLDER)/*.gcno $(TBUILD_FOLDER)/*.gcda ./coverage/build

cov: $(TEST_OBJECTS)
				@gcov --branch-probabilities -o ./coverage/build $^
				# Generate coverage report
				@lcov --capture --base-directory . --directory ./coverage/build --output ./coverage/build/coverage.info
				# Clean from std libraries and test functions
				@lcov --remove ./coverage/build/coverage.info "/usr*" --output ./coverage/build/coverage.info
				@lcov --remove ./coverage/build/coverage.info "*_test.cpp" --output ./coverage/build/coverage.info
				@lcov --remove ./coverage/build/coverage.info "*test_runner.cpp" --output ./coverage/build/coverage.info
				@genhtml --output coverage -t "🐋 Test Coverage" ./coverage/build/coverage.info
				@rm -rf *.log
				@echo "Launching in browser"
				@${BROWSER} ${PWD}/coverage/index.html &

# # Remove cpp, Remove src and put in build folder
# play: src/playground.cpp src/ia_ADQAPI.cpp
#			@$(eval TARGET=$(patsubst src/%.cpp,bin/%,$<))
#			@$(CXX) $(CXXFLAGS) $(INCS) $(LDLIBS) -o $(TARGET) $^
#			@echo "Build complete 🐋"
#			@./$(TARGET)

clean:
		@echo "🖌 Cleaning build objects"
		@rm -rf *.out $(TBUILD_FOLDER) $(BUILD_FOLDER)/*.o *.gcda *.gcno *.log *.o
