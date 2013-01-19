SHELL=C:/Windows/System32/cmd.exe
CC = gcc
FLAGS = -std=c99 -Wall -O3
OPENCL = -lOpenCL64

all: cpu_single_32.exe cpu_single_64.exe cpu_multiple_32.exe cpu_multiple_64.exe gpu_single.exe gpu_multiple.exe

cpu_single_32.exe:
	$(CC) $(FLAGS) -m32 -c cpu_single.c particle.c
	$(CC) $(FLAGS) -m32 cpu_single.o particle.o -o cpu_single_32.exe

cpu_single_64.exe:
	$(CC) $(FLAGS) -m64 -c cpu_single.c particle.c
	$(CC) $(FLAGS) -m64 cpu_single.o particle.o -o cpu_single_64.exe

cpu_multiple_32.exe:
	$(CC) $(FLAGS) -m32 -c cpu_multiple.c particle.c
	$(CC) $(FLAGS) -m32 cpu_multiple.o particle.o -o cpu_multiple_32.exe

cpu_multiple_64.exe:
	$(CC) $(FLAGS) -m64 -c cpu_multiple.c particle.c
	$(CC) $(FLAGS) -m64 cpu_multiple.o particle.o -o cpu_multiple_64.exe

oclSetup.o:
	$(CC) $(FLAGS) -c oclSetup.c

gpu_single.exe: oclSetup.o
	$(CC) $(FLAGS) -c gpu_single.c
	$(CC) $(FLAGS) $(OPENCL) gpu_single.o oclSetup.o -o gpu_single.exe

gpu_multiple.exe: oclSetup.o
	$(CC) $(FLAGS) -c gpu_multiple.c
	$(CC) $(FLAGS) $(OPENCL) gpu_multiple.o oclSetup.o -o gpu_multiple.exe

clean:
	rm *.o *.exe
