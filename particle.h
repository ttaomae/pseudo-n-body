#define EPSILON 0.000000001

typedef struct {
    float x;
    float y;
    float z;
    float m;
    float vx;
    float vy;
    float vz;
} particle_t;

void updateParticle(particle_t*, float*, float*, float*, float*, int, float);