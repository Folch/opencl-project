HOME    = $(PWD)
EXEC    = $(HOME)/main
SRC_SCL = $(HOME)/simple-opencl
CC      = g++
SRC = main.c $(SRC_SCL)/simpleCL.c 
CFLAGS  = -Wall -Wextra -pedantic -O3

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
INCL_P  = -I$(HOME)/inc -I/usr/local/cuda/include
LIBS   = -lm -lOpenCL -lrt
INCL_AMD = -I$(HOME)/inc -I$(AMDAPPSDKROOT)/include 
LIBS_AMD = -L$(AMDAPPSDKROOT)/lib/x86_64 $(LIBS)
CFLAGS_AMD  = $(CFLAGS) -DATI_OS_LINUX 
endif
ifeq ($(UNAME), Darwin)
INCL_P   = -I$(HOME)/inc
LIBS     = -framework OpenCL
INCL_AMD = $(INCL_P) 
LIBS_AMD = $(LIBS)
CFLAGS_AMD = $(CFLAGS)
endif

all:
	$(CC) $(CFLAGS) $(INCL_P) -c $(SRC)
	$(CC) $(CFLAGS) *.o -o $(EXEC) $(LIBS)
	rm -f *.o

amd:

	$(CC) $(CFLAGS_AMD) -c $(SRC) $(INCL_AMD) 
	$(CC) $(CFLAGS) *.o -o $(EXEC) $(LIBS_AMD)
	rm -f *.o

clean:
	rm -f *.o $(EXEC)

cleanbin:
	rm -f snapshot* image*



