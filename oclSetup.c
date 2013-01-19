#include <stdio.h>
#include <stdlib.h>

#include "oclSetup.h"

/*
 * Perform basic OpenCL setup and error checking.
 */
void oclSetup(cl_platform_id* cpPlatform,
              cl_device_id* device_id,
              cl_context* context,
              cl_command_queue* queue,
              cl_program* program,
              char* sourceFile)
{
    cl_int err;

    // platform
    err = clGetPlatformIDs(1, cpPlatform, NULL);
    if (err != 0) {
        fprintf(stderr, "Error getting platform ID\n");
        exit(-1);
    }

    // device
    err = clGetDeviceIDs(*cpPlatform, CL_DEVICE_TYPE_GPU, 1, device_id, NULL);
    if (err != 0) {
        fprintf(stderr, "Error getting device ID\n");
        exit(-1);
    }

    // context
    *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
    if (err != 0) {
        fprintf(stderr, "Error creating context\n");
        exit(-1);
    }

    // command queue
    *queue = clCreateCommandQueue(*context, *device_id, 0, &err);
    if (err != 0) {
        fprintf(stderr, "Error creating command queue\n");
        exit(-1);
    }

    // program
    char* source = readSourceFile(sourceFile);
    *program = clCreateProgramWithSource(*context, 1,
                                         (const char**)&source, NULL, &err);
    err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t len; char buffer[20480];
        clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        fprintf(stderr, "Error building program\n");
        fprintf(stderr, "%s\n", buffer);
    }
}

/*
 * Reads the contents of a file into a char*.
 */
char* readSourceFile(const char* filename)
{
    FILE* fp;
    char* source;
    int size = -1;  // default error value

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error openning kernel file: %s\n", filename);
        exit(-1);
    }

    if (fseek(fp, 0, SEEK_END) == 0) {
        size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
    }

    if (size == -1) {
        fprintf(stderr, "Error getting size of file: %s\n", filename);
        exit(-1);
    }


    source = (char*)malloc(size+1);
    if (source == NULL) {
        fprintf(stderr, "Error allocating memory for kernel source");
        exit(-1);
    }

    if (fread(source, 1, size, fp) != size) {
        fprintf(stderr, "Error reading kernel file: %s\n", filename);
        exit(-1);
    }

    source[size] = '\0';

    return source;
}