#define INDEX(i, j, w) i*w+j
#define EPSILON 0.000000001

/*
 * Compute the acceleration on the test particles (tParticles) cause by
 * each pseduo-particle (pParticles).
 * The accelerations are stored such that the acceleration caused by
 * pParticles[4] on tParticles[10] is stored in accel[10][4].
 *
 * get_global_id(0) == which test particle to operate on
 * get_global_id(1) == which pseduo-particle to operate on
 * len1 == number of particles
 * len2 == number of pseduo-particles
 */
__kernel void compute(__global float8* tParticles,
                      __global float4* pParticles,
                      __global float4* accel,
                      const unsigned int len1,
                      const unsigned int len2)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i < len1 && j < len2) {
        float4 d = pParticles[j] - tParticles[i].lo;

        float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + EPSILON);
        float invr3 = invr*invr*invr;
        float f = pParticles[j].s3 * invr3;    // s3 = mass

        // accel[INDEX(i, j, len2)] = (float4)(f*d.x, f*d.y, f*d.z, 0.0);
        accel[i*len2 +j] = (float4)(f*d.x, f*d.y, f*d.z, 0.0);
        // accel[idx] = f*d;
    }
}


/*
 * Sum up the accelerations calculated in the compute() kernel.
 * The final sum is stored such that the acceleration on tParticles[10]
 * is stored in accel[10][0].
 *
 * Adapted from NVIDIA's tutorial on reductions in CUDA.
 * The tutorial can be found here:
 * http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
 */
__kernel void reduce(__global float4* accel,
                     const unsigned int len1,
                     const unsigned int len2,
                     __local float4* sdata)
{
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);
    // unsigned int gid = INDEX(global_i, global_j, len2);
    unsigned int gid = global_i*len2 + global_j;

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);
    // local index
    // unsigned int tid = INDEX(local_i, local_j, get_local_size(1));
    unsigned int tid = local_i*get_local_size(1) + local_j;

    // write data to shared memory
    if (global_i < len1 && global_j < len2) {
        sdata[tid] = accel[gid];
    } else {
        sdata[tid] = (float4)(0.0, 0.0, 0.0, 0.0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s=get_local_size(1)/2; s>0; s>>=1) {
        if (local_j < s &&  local_j + s < get_local_size(1)) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (local_j == 0) {
        unsigned int idx = get_group_id(0)*get_num_groups(1) + get_group_id(1);
        accel[idx] = sdata[local_j];
    }
}


/*
 * Update the position and velocity of each test particle (tParticles).
 */
__kernel void update(__global float4* accel,
                     __global float8* tParticles,
                     const int len1,
                     const int len2,
                     const float t)
{
    int i = get_global_id(0);

    if (get_global_id(1) == 0 && i < len1) {
    // particles[i].s0 = 3;
        //update position
        float4 v = tParticles[i].hi;

        // int idx = INDEX(i, 0, len2);
        int idx = i*len2 + 0;
        tParticles[i].s0 += v.x * t + 0.5 * accel[idx].x * t*t;
        tParticles[i].s1 += v.y * t + 0.5 * accel[idx].y * t*t;
        tParticles[i].s2 += v.z * t + 0.5 * accel[idx].z * t*t;

        // update velocity
        tParticles[i].s4 += accel[idx].x * t;
        tParticles[i].s5 += accel[idx].y * t;
        tParticles[i].s6 += accel[idx].z * t;
    }
}