APPS=Example_1/example_1 helloworld/hello helloworld/helloworld basic_c/helloworld\
	 basic_c/threads basic_c/threads_2

all: ${APPS}

%: %.cu
	nvcc -g -G -I../tclap-1.4.0-rc1/include -I../Common -diag-suppress 940 -diag-suppress 611 \
		 -diag-suppress 191 -o $@ $<

%: %.c
	gcc -pthread -g -o $@ $<

%: %.cpp
	g++ -pthread -g -o $@ $<

clean:
	rm -f ${APPS}