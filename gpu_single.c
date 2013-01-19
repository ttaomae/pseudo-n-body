#include <stdio.h>
#include <stdlib.h>
#include <Cl/opencl.h>

#include "oclSetup.h"

/*
 * Creates a number of "pseudo-particles" which have an x-, y-, and z-coordiate
 * and a mass. These pseudo-particles do not move but they interact
 * gravitationally with a test particle which moves freely based on its
 * interaction with the pseudo-particles.
 */
int main(int argc, char* argv[])
{
    if (argc != 5) {
        fprintf(stderr, "requires 4 command line arguments:\n");
        fprintf(stderr, "\tlocal work group size\n\t#pseudo-particles\n");
        fprintf(stderr, "\t#iterations\n\tsize of time step\n");
        exit(-1);
    }

    const int numParticles = atoi(argv[2]);
    const int iterations = atoi(argv[3]);
    const float timeStep = (float)atof(argv[4]);

    size_t localSize = atoi(argv[1]);
    size_t globalSize = ceil(numParticles / (float)localSize) * localSize;

    cl_int err;
    cl_platform_id cpPlatform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    oclSetup(&cpPlatform, &device_id, &context, &queue,
             &program, "oclNBody_single.cl");

    // initialize data
    size_t particles_size = numParticles*sizeof(cl_float4);
    cl_float4* h_particles = (cl_float4*)malloc(particles_size);
    for (int i = 0; i < numParticles; i++) {
        cl_float4 temp = {{rand(), rand(), rand(), rand()}};
        h_particles[i] = temp;
    }

    cl_float8 h_particle = {{rand(),   rand(),   rand(),   rand()%500,
                             rand()%5, rand()%5, rand()%5, 0.0f       }};


    // create buffers
    cl_mem d_particles = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        particles_size, NULL, &err);
    cl_mem d_particle  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(cl_float8), NULL, &err);
    cl_mem d_accel     = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        particles_size, NULL, &err);
    if (err != 0) {
        fprintf(stderr, "Error creating buffers\n");
        exit(-1);
    }

    // copy data to device
    err  = clEnqueueWriteBuffer(queue, d_particles, CL_TRUE, 0,
                                particles_size, h_particles, 0,
                                NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_particle, CL_TRUE, 0,
                                sizeof(cl_float8), &h_particle, 0,
                                NULL, NULL);
    if (err != 0) {
        fprintf(stderr, "Error setting kernel arguments\n");
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
    err  = clSetKernelArg(kernel_compute, 0, sizeof(cl_mem), &d_particles);
    err |= clSetKernelArg(kernel_compute, 1, sizeof(cl_mem), &d_accel);
    err |= clSetKernelArg(kernel_compute, 2, sizeof(int),    &numParticles);
    err |= clSetKernelArg(kernel_compute, 3, sizeof(cl_mem), &d_particle);

    err  = clSetKernelArg(kernel_reduce, 0, sizeof(cl_mem), &d_accel);
    err |= clSetKernelArg(kernel_reduce, 1, sizeof(int),    &numParticles);
    err |= clSetKernelArg(kernel_reduce, 2, localSize*sizeof(cl_float4), NULL);

    err  = clSetKernelArg(kernel_update, 0, sizeof(cl_mem), &d_accel);
    err |= clSetKernelArg(kernel_update, 1, sizeof(cl_mem), &d_particle);
    err |= clSetKernelArg(kernel_update, 2, sizeof(int),    &timeStep);

    if (err != 0) {
        fprintf(stderr, "Error setting kernel arguments\n");
        exit(-1);
    }


    // print initial position of test particle
    printf("position: (%.12f, %.12f, %.12f)\n",
           h_particle.s[0], h_particle.s[1], h_particle.s[2]);

    // execute kernel
    for (int i = 0; i < iterations; i++) {
        // compute acceleration
        err = clEnqueueNDRangeKernel(queue, kernel_compute, 1,
                                     NULL, &globalSize, &localSize,
                                     0, NULL, NULL);
        clFinish(queue);

        // sum acceleration
        size_t n = (size_t)numParticles;
        // may require multiple reductions
        do {
            err = clSetKernelArg(kernel_reduce, 1, sizeof(int), &n);
            err = clEnqueueNDRangeKernel(queue, kernel_reduce, 1,
                                         NULL, &globalSize, &localSize,
                                         0, NULL, NULL);
            clFinish(queue);

            // after executing a reduction, the partial sums should be
            // in the first n (calculated below) array elements
            n = ceil(n / (float)localSize);
        } while(n > 1);

        // update position
        err = clEnqueueNDRangeKernel(queue, kernel_update, 1,
                                     NULL, &globalSize, &localSize,
                                     0, NULL, NULL);
        clFinish(queue);
    }

    if (err != 0) {
        fprintf(stderr, "Error executing kernels\n");
        exit(-1);
    }
    // copy data from device to host
    err = clEnqueueReadBuffer(queue, d_particle, CL_TRUE, 0,
                              sizeof(cl_float8), &h_particle, 0,
                              NULL, NULL);    if (err != 0) {
    fprintf(stderr, "Error copying data to host\n");
        exit(-1);
    }

    // print final position of test particle
    printf("position: (%.12f, %.12f, %.12f)\n",
           h_particle.s[0], h_particle.s[1], h_particle.s[2]);


}
