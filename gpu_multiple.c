#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Cl/opencl.h>

#include "oclSetup.h"
/*
 * Creates a number of "pseudo-particles" which have an x-, y-, and z-coordiate
 * and a mass. These pseudo-particles do not move but they interact
 * gravitationally with a test particles which moves freely based on its
 * interaction with the pseudo-particles but not other test-particles.
 */
int main(int argc, char* argv[])
{
    if (argc != 7) {
        fprintf(stderr, "requires 6 command line arguments:\n");
        fprintf(stderr, "\tlocal work group size (2 dimensions)\n");
        fprintf(stderr, "\t#test particles\n\t#pseudo-particles\n");
        fprintf(stderr, "\t#iterations\n\tsize of time step\n");
        exit(-1);
    }

    const int numTParticles = atoi(argv[3]);
    const int numPParticles = atoi(argv[4]);
    const int iterations = atoi(argv[5]);
    const float timeStep = (float)atof(argv[6]);

    size_t localSize[2] = {atoi(argv[1]), atoi(argv[2])};
    size_t globalSize[2] = {ceil(numTParticles / (float)localSize[0]) * localSize[0],
                            ceil(numPParticles / (float)localSize[1]) * localSize[1]};

    cl_int err;
    cl_platform_id cpPlatform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    oclSetup(&cpPlatform, &device_id, &context, &queue,
             &program, "oclNBody_multiple.cl");

    // initialize pseduo-particles
    size_t pParticles_size = numPParticles*sizeof(cl_float4);
    cl_float4* h_pParticles = (cl_float4*)malloc(pParticles_size);
    for (int i = 0; i < numPParticles; i++) {
        cl_float4 temp = {{rand(), rand(), rand(), 10000*rand()}};
        h_pParticles[i] = temp;
    }

    // initialize test particles
    size_t tParticles_size = numTParticles*sizeof(cl_float8);
    cl_float8* h_tParticles = (cl_float8*)malloc(tParticles_size);
    for (int i = 0; i < numTParticles; i++) {
        cl_float8 temp = {{rand(),   rand(),   rand(),   10000*(rand()%500),
                           rand()%5, rand()%5, rand()%5, 0.0f       }};
        h_tParticles[i] = temp;
    }


    // create buffers
    cl_mem d_pParticles = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         pParticles_size, NULL, &err);
    cl_mem d_tParticles = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         tParticles_size, NULL, &err);
    cl_mem d_accel      = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         numTParticles*numPParticles*sizeof(cl_float4),
                                         NULL, &err);
    if (err != 0) {
        fprintf(stderr, "Error creating buffers\n");
        exit(-1);
    }


    // copy data to device
    err  = clEnqueueWriteBuffer(queue, d_pParticles, CL_TRUE, 0,
                                pParticles_size, h_pParticles, 0,
                                NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_tParticles, CL_TRUE, 0,
                                tParticles_size, h_tParticles, 0,
                                NULL, NULL);
    if (err != 0) {
        fprintf(stderr, "Error copying data to device\n");
        exit(-1);
    }

    // setup kernels
    cl_kernel kernel_compute = clCreateKernel(program, "compute", &err);
    cl_kernel kernel_reduce  = clCreateKernel(program, "reduce",  &err);
    cl_kernel kernel_update  = clCreateKernel(program, "update",  &err);
    if (err != 0) {
        fprintf(stderr, "Error creating kernels\n");
        exit(-1);
    }

    // set the arguments for kernels
    err  = clSetKernelArg(kernel_compute, 0, sizeof(cl_mem), &d_tParticles);
    err |= clSetKernelArg(kernel_compute, 1, sizeof(cl_mem), &d_pParticles);

    if (err != 0) {
        fprintf(stderr, "Error setting 1  kernel arguments\n");
        exit(-1);
    }
    err |= clSetKernelArg(kernel_compute, 2, sizeof(cl_mem), &d_accel);
    err |= clSetKernelArg(kernel_compute, 3, sizeof(int),    &numTParticles);
    err |= clSetKernelArg(kernel_compute, 4, sizeof(int),    &numPParticles);

    err  = clSetKernelArg(kernel_reduce, 0, sizeof(cl_mem), &d_accel);
    err |= clSetKernelArg(kernel_reduce, 1, sizeof(int),    &numTParticles);
    err |= clSetKernelArg(kernel_reduce, 2, sizeof(int),    &numPParticles);
    err |= clSetKernelArg(kernel_reduce, 3, localSize[0]*localSize[1]
                                            *sizeof(cl_float4), NULL);

    if (err != 0) {
        printf("%d", err);
        fprintf(stderr, "Error setting 2 kernel arguments\n");
        exit(-1);
    }
    err  = clSetKernelArg(kernel_update, 0, sizeof(cl_mem), &d_accel);
    err |= clSetKernelArg(kernel_update, 1, sizeof(cl_mem), &d_tParticles);
    err |= clSetKernelArg(kernel_update, 2, sizeof(int),    &numTParticles);
    err |= clSetKernelArg(kernel_update, 3, sizeof(int),    &numPParticles);
    err |= clSetKernelArg(kernel_update, 4, sizeof(float),  &timeStep);

    if (err != 0) {
        fprintf(stderr, "Error setting  3 kernel arguments\n");
        exit(-1);
    }

    // print initial position of first test particle
    printf("position: (%.12f, %.12f, %.12f)\n",
            h_tParticles[0].s[0], h_tParticles[0].s[1], h_tParticles[0].s[2]);

    // print intial x position of first 3 test particles
    // printf("position: (%.12f, %.12f, %.12f)\n",
            // h_tParticles[0].s[0], h_tParticles[1].s[0], h_tParticles[2].s[0]);

    // execute kernels
    for (int i = 0; i < iterations; i++) {
        // compute acceleration
        err = clEnqueueNDRangeKernel(queue, kernel_compute, 2,
                                     NULL, globalSize, localSize,
                                     0, NULL, NULL);
        if (err) {
            printf("Error executing compute kernel\n");
        }
        clFinish(queue);

        // sum acceleration
        size_t n = (size_t)numPParticles;
        while (n > 1) {
            err = clSetKernelArg(kernel_reduce, 2, sizeof(int), &n);
            err = clEnqueueNDRangeKernel(queue, kernel_reduce, 2,
                                         NULL, globalSize, localSize,
                                         0, NULL, NULL);
        if (err) {
            printf("Error executing reduce kernel\n");
        }
            clFinish(queue);

            n = ceil(n / (float)localSize[0]);
        }

        // update position
        err = clEnqueueNDRangeKernel(queue, kernel_update, 2,
                                     NULL, globalSize, localSize,
                                     0, NULL, NULL);
        if (err) {
            printf("Error executing update kernel\n");
        }
        clFinish(queue);
    }

    // copy data from device to host
    err = clEnqueueReadBuffer(queue, d_tParticles, CL_TRUE, 0,
                              tParticles_size, h_tParticles, 0,
                              NULL, NULL);    if (err != 0) {
    fprintf(stderr, "error copying data to host\n");
        exit(-1);
    }


    // print final position of first test particle
    printf("position: (%.12f, %.12f, %.12f)\n",
            h_tParticles[0].s[0], h_tParticles[0].s[1], h_tParticles[0].s[2]);

    // print final x positions of first 3 test particles
    // printf("position: (%.12f, %.12f, %.12f)\n",
            // h_tParticles[0].s[0], h_tParticles[1].s[0], h_tParticles[2].s[0]);

}