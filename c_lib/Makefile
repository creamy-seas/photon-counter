CXX := g++

CPPFLAGS += -DLINUX -std=c++11

LDFLAGS = $(shell pkg-config --cflags libadq) # Digitiser library
LDLIBS := $(shell pkg-config --libs libadq) # Digitiser library
LDLIBS += $(shell pkg-config --libs cppunit) # Unittest library

INCS := -Iincludes # Location of header files

BROWSER := ns
#BROWSER := ck
PWD := $(shell pwd)

.PHONY: clean

lib: src/ia_ADQAPI.cpp
	@$(eval TARGET=$(patsubst src/%.cpp,build/%,$<))
	@$(CXX) -Wall $(CPPFLAGS) $(INCS) $(LDLIBS) -shared -o $(TARGET) -fPIC $^
	@echo Moving $(TARGET) to /usr/lib 🐳
	@sudo cp -f $(TARGET) /usr/lib

# Remove cpp, Remove src and put in build folder
play: src/playground.cpp src/ia_ADQAPI.cpp
	@$(eval TARGET=$(patsubst src/%.cpp,build/%,$<))
	@$(CXX) $(CPPFLAGS) $(INCS) $(LDLIBS) -o $(TARGET) $^
	@echo "Build complete 🐋"
	@./$(TARGET)

cov: tst
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

tst: test/utils_test.cpp test/test_runner.cpp test/example_test.cpp test/ia_ADQAPI_test.cpp test/progress_bar_test.cpp src/ia_ADQAPI.cpp src/progress_bar.cpp src/power_cpu.cpp test/power_cpu_test.cpp
	@$(CXX) -DTESTENV $(CPPFLAGS) $(LDFLAGS) $(LDLIBS) $(INCS) -o build/test -fprofile-arcs -ftest-coverage $^
	@./build/test
	@mkdir -p ./coverage/build
	@mv -f *.gcno *.gcda ./coverage/build

clean:
	@echo "🖌 Cleaning build objects"
	@rm -rf *.out *.o *.gcda *.gcno *.log
