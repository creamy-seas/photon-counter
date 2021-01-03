CXX := g++

CPPFLAGS += -DLINUX -std=c++11

# LDFLAGS = $(shell pkg-config --cflags libadq) # Digitiser library
# LDLIBS := $(shell pkg-config --libs libadq) # Digitiser library
LDLIBS += $(shell pkg-config --libs cppunit) # Unittest library

INCS := -Iincludes # Location of header files

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

.PHONY: clean test

test: test/testRunner.cpp test/exampleTest.cpp # test/iaADQAPITest.cpp src/iaADQAPI.cpp
	@$(CXX) $(CPPFLAGS) $(LDFLAGS) $(LDLIBS) $(INCS) -o build/test $^
	@./build/test

clean:
	@echo "🖌 Cleaning build objects"
	@rm -rf *.out *.o
