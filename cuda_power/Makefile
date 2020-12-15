INCLUDES:=-Iincludes

run: build
	@./ia_power

build: main.cu
	nvcc $(INCLUDES) -o ia_power $<

.PHONY: clean

clean:
	@rm -rf *.out *.o
