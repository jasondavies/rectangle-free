// C interface for the partition solver; cross-checked against hashlib.
// No external crypto dependency.
#include "sha256_c.h"
#include <string.h>
static uint32_t rotate(uint32_t v, unsigned n) { return (v>>n)|(v<<(32-n)); }
static void compress(RectSha256* h, const uint8_t* in) {
    static const uint32_t constants[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    uint32_t w[64];
    for(unsigned i=0;i<16;i++)
        w[i]=(uint32_t)in[4*i]<<24|(uint32_t)in[4*i+1]<<16|
             (uint32_t)in[4*i+2]<<8|in[4*i+3];
    for(unsigned i=16;i<64;i++) {
        uint32_t x=w[i-15], y=w[i-2];
        w[i]=w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+w[i-7]+
             (rotate(y,17)^rotate(y,19)^(y>>10));
    }
    uint32_t a=h->state[0],b=h->state[1],c=h->state[2],d=h->state[3];
    uint32_t e=h->state[4],f=h->state[5],g=h->state[6],z=h->state[7];
    for(unsigned i=0;i<64;i++) {
        uint32_t t1=z+(rotate(e,6)^rotate(e,11)^rotate(e,25))+
                    ((e&f)^(~e&g))+constants[i]+w[i];
        uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+
                    ((a&b)^(a&c)^(b&c));
        z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h->state[0]+=a;h->state[1]+=b;h->state[2]+=c;h->state[3]+=d;
    h->state[4]+=e;h->state[5]+=f;h->state[6]+=g;h->state[7]+=z;
}
void rect_sha256_init(RectSha256* h) {
    const uint32_t initial[8]={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                              0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    memset(h,0,sizeof(*h));memcpy(h->state,initial,sizeof(initial));
}
void rect_sha256_update(RectSha256* h,const void* data,size_t length) {
    const uint8_t* bytes=(const uint8_t*)data;
    h->bytes+=length;
    while(length) {
        size_t take=64-h->buffered;
        if(take>length)take=length;
        memcpy(h->block+h->buffered,bytes,take);
        h->buffered+=take;bytes+=take;length-=take;
        if(h->buffered==64){compress(h,h->block);h->buffered=0;}
    }
}
void rect_sha256_finish(RectSha256* h,char hex[65]) {
    const char digits[]="0123456789abcdef";
    uint64_t bits=h->bytes*8;
    h->block[h->buffered++]=0x80;
    if(h->buffered>56) {
        memset(h->block+h->buffered,0,64-h->buffered);
        compress(h,h->block);h->buffered=0;
    }
    memset(h->block+h->buffered,0,56-h->buffered);
    for(unsigned i=0;i<8;i++)h->block[63-i]=(uint8_t)(bits>>(8*i));
    compress(h,h->block);
    for(unsigned i=0;i<64;i++)hex[i]=digits[(h->state[i/8]>>(28-4*(i%8)))&15];
    hex[64]=0;
}
