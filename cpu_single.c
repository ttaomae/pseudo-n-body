#include <stdio.h>
#include <stdlib.h>

#include "particle.h"

/*
 * Creates a number of "pseudo-particles" which have an x-, y-, and z-coordiate
 * and a mass. These pseudo-particles do not move but they interact
 * gravitationally with a test particle which moves freely based on its
 * interaction with the pseudo-particles.
 */
int main(int argc, char** argv)
{
    if (argc != 4) {
        fprintf(stderr, "requires 3 command line arguments:\n");
        fprintf(stderr, "\t#pseudo-particles\n\t#iterations\n");
        fprintf(stderr, "\tsize of time step\n");
        exit(-1);
    }

    const int numPParticles = atoi(argv[1]);    // number of pseudo-particles
    const int iterations = atoi(argv[2]);
    const float timeStep = (float)atof(argv[3]);

    // positions and masses of pseudo-particles
    float* x = (float*)malloc((numPParticles)*sizeof(float));
    float* y = (float*)malloc((numPParticles)*sizeof(float));
    float* z = (float*)malloc((numPParticles)*sizeof(float));
    float* m = (float*)malloc((numPParticles)*sizeof(float));

    // initialize pseudo-particles
    for (int i = 0; i < numPParticles; i++) {
        x[i] = rand();
        y[i] = rand();
        z[i] = rand();
        m[i] = rand();
    }

    // initialize test particle
    particle_t* testParticle = (particle_t*)malloc(sizeof(particle_t));
    testParticle->x = rand();
    testParticle->y = rand();
    testParticle->z = rand();
    testParticle->m = rand() % 500;
    testParticle->vx = rand() % 5;
    testParticle->vy = rand() % 5;
    testParticle->vz = rand() % 5;

    // print initial position
    printf("position: (%.12f, %.12f, %.12f)\n",
            testParticle->x, testParticle->y, testParticle->z);

    // update particle over a number of iterations
    for (int i = 0; i < iterations; i++) {
        updateParticle(testParticle, x, y, z, m, numPParticles, timeStep);
    }

    // print final position
    printf("position: (%.12f, %.12f, %.12f)\n",
            testParticle->x, testParticle->y, testParticle->z);
}
