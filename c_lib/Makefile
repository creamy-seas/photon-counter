CC := g++

CFLAGS := $(shell pkg-config --cflags libadq)
CFLAGS += -Wall -DLINUX -std=c++11

INCS := -Iincludes
LIBS := $(shell pkg-config --libs libadq)

# Remove cpp, Remove src and put in build folder
play: src/playground.cpp
	$(eval TARGET=$(patsubst src/%.cpp,build/%,$<))
	@echo $(CC) $(CFLAGS) $(INCS) $(LIBS) -o $(TARGET) $^
	@./$(TARGET)

.PHONY: clean
clean:
	@echo "🖌 Cleaning"
	@rm -rf *.out *.o build
