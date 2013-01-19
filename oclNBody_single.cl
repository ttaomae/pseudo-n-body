#define EPSILON 0.000000001

/*
 * Compute the acceleration on the test particle (tParticle) caused by
 * each pseudo-particle (pParticles).
 */
__kernel void compute(__global float4* pParticles,
                      __global float4* accel,
                      const unsigned int len,
                      __global float8* tParticle)
{
    unsigned int idx = get_global_id(0);

    if (idx < len) {
        float4 d = pParticles[idx] - (*tParticle).lo;

        // EPSILON allows particles to "pass through" each other
        float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + EPSILON);
        float invr3 = invr*invr*invr;
        float f = pParticles[idx].s3 * invr3;    // s3 = mass

        accel[idx] = (float4)(f*d.x, f*d.y, f*d.z, 0.0f);
        // accel[idx] = f*d;
    }
}

/*
 * Sum up the accelerations calculated in the compute() kernel.
 * Adapted from NVIDIA's tutorial on reductions in CUDA.
 * The tutorial can be found here:
 * http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
 */
__kernel void reduce(__global float4* accel,
                     const unsigned int len,
                     __local float4* sdata)
{
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    sdata[tid] = (gid < len) ? accel[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (tid == 0) {
        accel[get_group_id(0)] = sdata[0];
    }
}

/*
 * Update the position and velocity of the test particle (tParticle).
 */
__kernel void update(__global float4* accel,
                     __global float8* tParticle,
                     const float t)
{
    if (get_global_id(0) == 0) {

        //update position
        float4 v = (*tParticle).hi;
        (*tParticle).s0 += v.x * t + 0.5 * accel[0].x * t*t;
        (*tParticle).s1 += v.y * t + 0.5 * accel[0].y * t*t;
        (*tParticle).s2 += v.z * t + 0.5 * accel[0].z * t*t;

        // update velocity
        (*tParticle).s4 += accel[0].x * t;
        (*tParticle).s5 += accel[0].y * t;
        (*tParticle).s6 += accel[0].z * t;
    }
}