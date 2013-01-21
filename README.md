#Pseudo-n-body Problem
The pseudo-n-body problem is a variation of the classic [n-body problem](http://en.wikipedia.org/wiki/N-body_problem "n-body problem"). In the classic n-body problem, there are a number of point masses which interact with each other gravitationally. In this variation, there are a number of *pseudo-particles* which do not move and one or more *test particles* which move based on their interactions with pseudo-particles, but not other test particles.

##Implementation
There is a (single-threaded) CPU version and a GPU (OpenCL) version for either a single or multiple test particles. Each version allows you to specify amount of pseudo-particles, iterations, and the duration of the time steps used to calculate the movement. The version which allows multiple test particles, allows you to specify the number of test particles. The GPU versions allows you to specify the local work group size.

##Output
Each version outputs the start and end position of the (first) test particle. Comparison of the output of the CPU and GPU versions have shown various discrepancies. When running the program with more iterations, the results tend to diverge. I suspect that some of these discrepancies are explained by the fact that the GPU has limited precision for floating point numbers. Even though I used single precision floating point numbers for both the CPU and GPU versions, I suspect that some of the deviation is because the CPU uses extended precision for intermediate calculations. More information on the issue can be found [here](http://stackoverflow.com/questions/11176990/opencl-floating-point-precision "OpenCL floating point precision"). Compiling the CPU version as a 32-bit executable seemed to reduce, but not eliminate, the differences. I suspect that the remaining discrepancies are a result of the parallel nature of GPU computing, and the non-associativity of floating point numbers.

##Results
For the version with a single test particle, the GPU version performs on par or better than the CPU version. The GPU version sees greater advantage when there are a larger number of pseudo-particles. This result is likely caused by the fact that a larger number of pseudo-particles allows for more data parallel operations, which favors the GPU.

The GPU version with multiple test particles seems to perform on par or worse than the CPU version. The reason for this result is unclear, but it is possibly due to poorly optimized GPU code.

There is definitely room for improvement in both the CPU and GPU versions, so it is unclear whether the GPU can provide significant performance benefits for this application.