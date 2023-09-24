#include <Sha/Sha.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <cstdio>
#include <cstring>

namespace Sha
{
    ShaInfo::ShaInfo(unsigned long seed) :
        digest{
            0x67452301L + seed,
            0xefcdab89L + seed,
            0x98badcfeL + seed,
            0x10325476L + seed,
            0xc3d2e1f0L + seed
        }
    {};

    void ShaInfo::digestFile(std::string fileUrl)
    {
        constexpr unsigned blockSize = 8192;
        char buffer[blockSize];

        std::ifstream file(fileUrl, file.binary);
        unsigned i = 0;
        while (file.good()){
            file.read(buffer, blockSize);
            update(reinterpret_cast<unsigned char*>(buffer), file.gcount());
        }
        final();
    };

    void ShaInfo::update(unsigned char* buffer, unsigned size)
    {
        if ((countLow + ((unsigned long) size << 3)) < countLow) {
            ++countHigh;
        }
        countLow += (unsigned long) size << 3;
        countHigh += (unsigned long) size >> 29;
        while (size >= sizeof(data)) {
            std::memcpy(data, buffer, sizeof(data));
        #ifdef LITTLE_ENDIAN
            byteReverse(data, sizeof(data));
        #endif /* LITTLE_ENDIAN */
            transform();
            buffer += sizeof(data);
            size -= sizeof(data);
        }
        std::memcpy(data, buffer, size);
    };

    void ShaInfo::final()
    {
        int count;
        unsigned long lowBitCount;
        unsigned long highBitCount;

        lowBitCount = countLow;
        highBitCount = countHigh;
        count = (int) ((lowBitCount >> 3) & 0x3f);
        ((unsigned char *) data)[count++] = 0x80;
        if (count > 56) {
            std::memset((unsigned char *) &data + count, 0, 64 - count);
        #ifdef LITTLE_ENDIAN
            byteReverse(data, sizeof(data));
        #endif /* LITTLE_ENDIAN */
            transform();
            std::memset(data, 0, 56);
        } else {
            std::memset((unsigned char *) &data + count, 0, 56 - count);
        }
    #ifdef LITTLE_ENDIAN
        byteReverse(data, sizeof(data));
    #endif /* LITTLE_ENDIAN */
        data[14] = highBitCount;
        data[15] = lowBitCount;
        transform();
    };

    void ShaInfo::transform()
    {
    #define ROT32(x,n)      ((x << n) | (x >> (32 - n)))
    #define f1(x,y,z)       ((x & y) | (~x & z))
    #define f2(x,y,z)       (x ^ y ^ z)
    #define f3(x,y,z)       ((x & y) | (x & z) | (y & z))
    #define f4(x,y,z)       (x ^ y ^ z)
    #define CONST1          0x5a827999L
    #define CONST2          0x6ed9eba1L
    #define CONST3          0x8f1bbcdcL
    #define CONST4          0xca62c1d6L
    #define FUNC(n,i)       \
        tmp = ROT32(A,5) + f##n(B,C,D) + E + W[i] + CONST##n;  \
        E = D; D = C; C = ROT32(B,30); B = A; A = tmp

        unsigned long tmp, A, B, C, D, E, W[80];

        #pragma unroll
        for (unsigned i = 0; i < 16; ++i) {
            W[i] = data[i];
        }
        #pragma unroll
        for (unsigned i = 16; i < 80; ++i) {
            W[i] = W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16];
    #ifdef USE_MODIFIED_SHA
            W[i] = ROT32(W[i], 1);
    #endif /* USE_MODIFIED_SHA */
        }
        A = digest[0];
        B = digest[1];
        C = digest[2];
        D = digest[3];
        E = digest[4];
        #pragma unroll
        for (unsigned i = 0; i < 20; ++i){
            FUNC(1,i);
        }
        #pragma unroll
        for (unsigned i = 20; i < 40; ++i){
            FUNC(2,i);
        }
        #pragma unroll
        for (unsigned i = 40; i < 60; ++i){
            FUNC(3,i);
        }
        #pragma unroll
        for (unsigned i = 60; i < 80; ++i){
            FUNC(4,i);
        }
        digest[0] += A;
        digest[1] += B;
        digest[2] += C;
        digest[3] += D;
        digest[4] += E;

    #undef ROT32
    #undef f1
    #undef f2
    #undef f3
    #undef f4
    #undef CONST1
    #undef CONST2
    #undef CONST3
    #undef CONST4
    #undef FUNC
    };

    void byteReverse(unsigned long* buffer, unsigned size)
    {
        constexpr unsigned ratio = sizeof(unsigned long)/sizeof(unsigned char);
        unsigned char tmpBytes[ratio];
        unsigned char *currBytes = (unsigned char *) buffer;

        const unsigned n = size / sizeof(unsigned long);
        for (unsigned i = 0; i < n; ++i) {
            #pragma unroll
            for(unsigned j = 0; j < ratio; ++j){
                tmpBytes[j] = currBytes[j];
            }
            #pragma unroll
            for(unsigned j = 0; j < ratio; ++j){
                currBytes[j] = tmpBytes[ratio-1-j];
            }
            currBytes += sizeof(unsigned long);
        }
    };

    std::ostream& operator<<(std::ostream& os, const ShaInfo& sha)
    {
        return (os 
            << std::hex
            << sha.digest[0] << " "
            << sha.digest[1] << " "
            << sha.digest[2] << " "
            << sha.digest[3] << " "
            << sha.digest[4] 
            << std::dec
        );
    };
}
