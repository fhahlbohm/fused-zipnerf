#pragma once

// counter-based random numbers: every value is a hash of (seed, stream, index), so kernels regenerate identical values
// instead of storing them (shared with the runtime-compiled kernels, so no includes)

namespace zipnerf::random {

    enum Stream : unsigned long long {
        pixel = 0,
        background = 1,
        jitter = 2,  // + level
        pattern = 16,  // + level
    };

    __device__ inline unsigned long long splitmix64(unsigned long long x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    __device__ inline unsigned long long hash(const unsigned long long seed, const unsigned long long stream, const unsigned long long index) {
        return splitmix64(splitmix64(seed ^ (stream << 56)) ^ index);
    }

    // uniform in [0, 1) with 24 random bits
    __device__ inline float uniform(const unsigned long long seed, const unsigned long long stream, const unsigned long long index) {
        return static_cast<float>(hash(seed, stream, index) >> 40) * (1.0f / 16777216.0f);
    }

    // uniform integer in [0, n)
    __device__ inline unsigned long long uniform_int(const unsigned long long seed, const unsigned long long stream, const unsigned long long index, const unsigned long long n) {
        return __umul64hi(hash(seed, stream, index), n);
    }

}
