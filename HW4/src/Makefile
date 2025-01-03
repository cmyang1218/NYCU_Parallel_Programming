CXX = mpicxx
SRCS = main.cc
SRCS += matrix.cc

OBJS = ${SRCS:.cc=.o}

.PHONY: all clean

all: matmul
	
matmul: ${OBJS}
	${CXX} -o $@ $^

%.o: %.cc
	${CXX} -c $^ -o $@

clean:
	rm -r matmul ${OBJS}