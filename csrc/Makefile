CFLAGS := $(shell pkg-config --cflags libadq)
CFLAGS += -Wall -DLINUX

LIBS := $(shell pkg-config --libs libadq)

# Get all .c files
SRC = ia_ADQAPI.c
# Make libraries
OBJ = $(addprefix build/,$(SRC:.c=.os))

all: $(SRC)
	@gcc $(CFLAGS) $(LIBS) -shared -o $(OBJ) -fPIC $(SRC)
	@echo 🐳 Moving to /usr/lib 🐳
	@sudo cp -f $(OBJ) /usr/lib

# More clear
run: build/run
build/run: sp_digitiser.c
	@gcc -Wall $(LIBS) $(CFLAGS) -o $@ $<

cuda: main.cu
	@nvcc main.cu -o run

.PHONY: clean
clean:
	@echo "🖌 Cleaning"
	@rm -rf *.out *.o build
