#include <math.h>
#include "particle.h"

/*
 * Takes one test particle, n amount of x- y- and z-coordiates, and masses
 * and a time-step t.
 * Updates the position and velocity of the test particle.
 */
void updateParticle(particle_t* p, float* x, float* y, float* z, float* m, int n, float t)
{
    float ax = 0.0;
    float ay = 0.0;
    float az = 0.0;

    // loop through each pseudo-particle and calculate acceleration
    for (int i = 0; i < n; i++) {
        float dx = x[i] - p->x;
        float dy = y[i] - p->y;
        float dz = z[i] - p->z;

        float invr = 1.0 / sqrt(dx*dx + dy*dy + dz*dz + EPSILON);
        float invr3 = invr*invr*invr;
        float f = m[i] * invr3;

        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }

    // update position and velocity of the test particle
    p->x += p->vx * t + 0.5 * ax * t*t;
    p->y += p->vy * t + 0.5 * ay * t*t;
    p->z += p->vz * t + 0.5 * az * t*t;

    p->vx += ax * t;
    p->vy += ay * t;
    p->vz += az * t;
}