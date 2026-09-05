#pragma once
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "../../src/common/sha256.hpp"

namespace common_catalog {
struct Entry { uint64_t key,coefficient;uint8_t primes; };
// Local research artifacts: explicit little-endian fields, no struct padding.
struct File {
    FILE* file;Sha256 hash;
    explicit File(const std::string& path,bool write):file(std::fopen(path.c_str(),write?"wbx":"rb")) {
        if(!file)throw std::runtime_error("cannot open fresh/readable artifact: "+path);
    }
    ~File(){if(file)std::fclose(file);}
    void put(const void* p,size_t n){if(std::fwrite(p,1,n,file)!=n)throw std::runtime_error("artifact write failed");hash.update(p,n);}
    void get(void* p,size_t n){if(std::fread(p,1,n,file)!=n)throw std::runtime_error("truncated artifact");hash.update(p,n);}
    void put64(uint64_t x){uint8_t b[8];for(unsigned i=0;i<8;++i)b[i]=uint8_t(x>>(8*i));put(b,8);}
    uint64_t get64(){uint8_t b[8];get(b,8);uint64_t x=0;for(unsigned i=0;i<8;++i)x|=uint64_t(b[i])<<(8*i);return x;}
    std::string finish_write(){auto digest=hash.finish_hex();
        if(std::fwrite(digest.data(),1,64,file)!=64||std::fflush(file))throw std::runtime_error("artifact footer failed");
        return digest;}
    std::string finish_read(){auto digest=hash.finish_hex();char wanted[64];
        if(std::fread(wanted,1,64,file)!=64||digest!=std::string(wanted,64)||std::fgetc(file)!=EOF)
            throw std::runtime_error("artifact checksum/trailer mismatch");return digest;}
};
template<class Records> std::string write(const std::string& path,unsigned slack,const Records& records,const std::vector<uint8_t>& primes) {
    if(records.size()!=primes.size())throw std::runtime_error("missing query prime counts");
    File f(path,true);f.put("HCCAT001",8);f.put64(slack);f.put64(records.size());
    for(size_t i=0;i<records.size();++i){f.put64(records[i].key);f.put64(records[i].coefficient);f.put(&primes[i],1);}
    return f.finish_write();
}
inline std::vector<Entry> read(const std::string& path,unsigned& slack,std::string& digest) {
    File f(path,false);char magic[8];f.get(magic,8);
    if(std::memcmp(magic,"HCCAT001",8))throw std::runtime_error("unknown catalog representation");
    slack=unsigned(f.get64());uint64_t count=f.get64();
    if(slack<1||slack>3||count>45007139)throw std::runtime_error("catalog dimensions outside gate");
    const uint64_t expected[4]={0,29,36398,45007139};
    if(count!=expected[slack])throw std::runtime_error("catalog is not a complete known census");
    std::vector<Entry> out(count);
    for(auto& r:out){r.key=f.get64();r.coefficient=f.get64();f.get(&r.primes,1);
        unsigned d=unsigned(r.key>>60),used=unsigned(__builtin_popcountll(r.key&((UINT64_C(1)<<60)-1)));
        if(!r.primes||r.primes>4||!r.coefficient||d>2*slack||used<2*d||used-2*d>2*slack)
            throw std::runtime_error("invalid catalog entry");}
    digest=f.finish_read();return out;
}
}
