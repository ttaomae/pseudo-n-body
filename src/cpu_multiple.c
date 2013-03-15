#include <stdio.h>
#include <stdlib.h>

#include "particle.h"

/*
 * Creates a number of "pseudo-particles" which have an x-, y-, and z-coordiate
 * and a mass. These pseudo-particles do not move but they interact
 * gravitationally with a test particles which moves freely based on its
 * interaction with the pseudo-particles but not other test-particles.
 */
int main(int argc, char** argv)
{
    if (argc != 5) {
        fprintf(stderr, "requires 4 command line arguments:\n");
        fprintf(stderr, "\t#test particles\n\t#pseudo-particles\n");
        fprintf(stderr, "\t#iterations\n\tsize of time step\n");
        exit(-1);
    }

    const int numTParticles = atoi(argv[1]);
    const int numPParticles = atoi(argv[2]);
    const int iterations = atoi(argv[3]);
    const float timeStep = (float)atof(argv[4]);

    // positions and masses of pseudo-particles
    float* x = (float*)malloc(numPParticles*sizeof(float));
    float* y = (float*)malloc(numPParticles*sizeof(float));
    float* z = (float*)malloc(numPParticles*sizeof(float));
    float* m = (float*)malloc(numPParticles*sizeof(float));

    // array of test particles
    particle_t** tParticles = (particle_t**)malloc(numTParticles*sizeof(particle_t*));

    // initialize pseudo-particles
    for (int i = 0; i < numPParticles; i++) {
        x[i] = rand();
        y[i] = rand();
        z[i] = rand();
        m[i] = rand();
    }
    // initialize test particles
    for (int i = 0; i < numTParticles; i++) {
        tParticles[i] = (particle_t*)malloc(sizeof(particle_t));
        tParticles[i]->x = rand();
        tParticles[i]->y = rand();
        tParticles[i]->z = rand();
        tParticles[i]->m =  rand() % 500;
        tParticles[i]->vx = rand() % 5;
        tParticles[i]->vy = rand() % 5;
        tParticles[i]->vz = rand() % 5;
    }

    // print the initial position of the first test particle
    printf("position: (%.12f, %.12f, %.12f)\n",
            tParticles[0]->x, tParticles[0]->y, tParticles[0]->z);

    // print the initial x position of the first 3 test particles.
    // printf("position: (%.12f, %.12f, %.12f)\n",
            // tParticles[0]->x, tParticles[1]->x, tParticles[2]->x);

    // update particles over a number of iterations
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < numTParticles; j++) {
            updateParticle(tParticles[j], x, y, z, m, numPParticles, timeStep);
        }
    }

    // print the final position of the first test particles
    printf("position: (%.12f, %.12f, %.12f)\n",
            tParticles[0]->x, tParticles[0]->y, tParticles[0]->z);

    // print the final x position of the first 3 test particles
    // printf("position: (%.12f, %.12f, %.12f)\n",
            // tParticles[0]->x, tParticles[1]->x, tParticles[2]->x);

}
