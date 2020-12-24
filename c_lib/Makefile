CC := g++

CFLAGS := $(shell pkg-config --cflags libadq)
CFLAGS += -DLINUX -std=c++11

INCS := -Iincludes
LIBS := $(shell pkg-config --libs libadq)

lib: src/ia_ADQAPI.cpp
	@$(eval TARGET=$(patsubst src/%.cpp,build/%,$<))
	@$(CC) -Wall $(CFLAGS) $(INCS) $(LIBS) -shared -o $(TARGET) -fPIC $^
	@echo Moving $(TARGET) to /usr/lib 🐳
	@sudo cp -f $(TARGET) /usr/lib

# Remove cpp, Remove src and put in build folder
play: src/playground.cpp src/ia_ADQAPI.cpp
	@$(eval TARGET=$(patsubst src/%.cpp,build/%,$<))
	@$(CC) $(CFLAGS) $(INCS) $(LIBS) -o $(TARGET) $^
	@echo "Build complete 🐋"
	@./$(TARGET)

.PHONY: clean
clean:
	@echo "🖌 Cleaning"
	@rm -rf *.out *.o build
