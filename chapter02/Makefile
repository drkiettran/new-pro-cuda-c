APPS = checkDeviceInfor/checkDeviceInfor \
	   checkDimension/checkDimension \
	   checkThreadIndex/checkThreadIndex \
	   defineGridBlock/defineGridBlock \
	   sumArraysOnGPU-small-case/sumArraysOnGPU-small-case \
	   sumArraysOnHost/sumArraysOnHost \
	   sumArraysOnGPU-timer/sumArraysOnGPU-timer \
	   sumMatrixOnGPU-1D-grid-1D-block/sumMatrixOnGPU-1D-grid-1D-block \
	   sumMatrixOnGPU-2D-grid-1D-block/sumMatrixOnGPU-2D-grid-1D-block \
	   sumMatrixOnGPU-2D-grid-2D-block/sumMatrixOnGPU-2D-grid-2D-block \
	   vector_add/vector_add dot_product/dot_product dot_product_tree/dot_product_tree \
	   dot_product_binary_tree_sm/dot_product_binary_tree_sm \
	   matrix_add/matrix_add

all: ${APPS}

%: %.cu
	nvcc -g -G -I./include -I../tclap-1.4.0-rc1/include -I../Common -diag-suppress 940 -diag-suppress 611 \
         -diag-suppress 191 -Wno-deprecated-gpu-targets -o $@ $<

%: %.c
	gcc -pthread -g -o $@ $<

%: %.cpp
	g++ -pthread -I../Common -I../tclap-1.4.0-rc1/include -fpermissive -g -o $@ $<

clean:
	rm -f ${APPS}