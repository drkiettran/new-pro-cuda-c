APPS=Example_1/example_1 helloworld/hello helloworld/helloworld

all: ${APPS}

%: %.cu
#       nvcc -O2 -arch=sm_20 -o $@ $<
#       nvcc -O2  -g -G -o $@ $<
	nvcc -g -G -I../tclap-1.4.0-rc1/include -I../Common -diag-suppress 940 -diag-suppress 611 -diag-suppress 191 -o $@ $<

%: %.c
	g++ -pthread -g -o $@ $<

clean:
	rm -f ${APPS}