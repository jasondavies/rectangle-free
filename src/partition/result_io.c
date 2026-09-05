#define _POSIX_C_SOURCE 200809L
#include "partition_poly.h"
#include "../common/sha256_c.h"

// Bump this when the decomposition's mathematical meaning changes.
#define POLY_ALGORITHM "partition-structure-v2"
#ifndef RECT_PARTITION_SOURCE_ID
#define RECT_PARTITION_SOURCE_ID "unversioned"
#endif

static void hash_line(RectSha256* hash, const char* text) {
    rect_sha256_update(hash,text,strlen(text));
}
void poly_task_space_digest(int depth,int reorder,long long full_tasks,char out[65]) {
    RectSha256 hash;
    rect_sha256_init(&hash);
    char line[160];
    snprintf(line,sizeof(line),"%s rows=%d cols=%d depth=%d reorder=%d count4=%d feasibility=%d tasks=%lld\n",
             POLY_ALGORITHM,g_rows,g_cols,depth,reorder,RECT_COUNT_K4,
             RECT_COUNT_K4_FEASIBILITY,full_tasks);
    hash_line(&hash,line);
    // Hash the actual ordered partitions, independent of struct padding/ABI.
    for(int p=0;p<num_partitions;p++) {
        for(int r=0;r<g_rows;r++) {
            snprintf(line,sizeof(line),"%u,",partitions[p].mapping[r]);
            hash_line(&hash,line);
        }
        hash_line(&hash,"\n");
    }
    if(depth==2)for(long long i=0;i<g_live_prefix2_count;i++) {
        snprintf(line,sizeof(line),"%u,%u\n",g_live_prefix2_i[i],g_live_prefix2_j[i]);
        hash_line(&hash,line);
    }
    rect_sha256_finish(&hash,out);
}

static void write_coeff(FILE* f,PolyCoeff value) {
    // Also handles the most negative signed coefficient without signed negation.
    unsigned __int128 magnitude=(unsigned __int128)value;
    if(value<0){fputc('-',f);magnitude=0-magnitude;}
    char digits[40];
    int n=0;
    do {digits[n++]=(char)('0'+magnitude%10);magnitude/=10;}while(magnitude);
    while(n)fputc(digits[--n],f);
}

void write_poly_file(const char* path,const Poly* poly,const PolyFileMeta* meta) {
    const char* source=RECT_PARTITION_SOURCE_ID;
    if(strlen(source)!=64 || strspn(source,"0123456789abcdef")!=64) {
        fprintf(stderr,"Polynomial output requires a source identity; build with the repository Makefile\n");
        exit(1);
    }
    char* payload=NULL;
    size_t length=0;
    FILE* memory=open_memstream(&payload,&length);
    if(!memory){perror("polynomial payload");exit(1);}
    fprintf(memory,"RECT_POLY_V2\nalgorithm %s\nsolver_source %s\n",
            POLY_ALGORITHM,RECT_PARTITION_SOURCE_ID);
    fprintf(memory,"mode %s\nrows %d\ncols %d\nprefix_depth %d\nreorder %d\ntask_space %s\n",
            meta->count_k4?"count4":"polynomial",meta->rows,meta->cols,
            meta->prefix_depth,meta->reorder,meta->task_space);
    fprintf(memory,"task_start %lld\ntask_end %lld\nfull_tasks %lld\ndeg %d\n",
            meta->task_start,meta->task_end,meta->full_tasks,poly->deg);
    for(int i=0;i<=poly->deg;i++) {
        fprintf(memory,"coeff %d ",i);
        write_coeff(memory,poly->coeffs[i]);
        fputc('\n',memory);
    }
    int payload_failed=ferror(memory);
    if(fclose(memory))payload_failed=1;
    if(payload_failed){perror("polynomial payload");free(payload);exit(1);}
    RectSha256 hash;char digest[65];
    rect_sha256_init(&hash);rect_sha256_update(&hash,payload,length);rect_sha256_finish(&hash,digest);

    size_t path_length=strlen(path)+16;
    char* temporary=malloc(path_length);
    if(!temporary){free(payload);exit(1);}
    snprintf(temporary,path_length,"%s.tmp.XXXXXX",path);
    int fd=mkstemp(temporary);
    FILE* f=fd<0?NULL:fdopen(fd,"wb");
    if(!f) {
        if(fd>=0){close(fd);unlink(temporary);}
        perror("polynomial temporary file");free(temporary);free(payload);exit(1);
    }
    int failed=fwrite(payload,1,length,f)!=length;
    if(fprintf(f,"sha256 %s\nend\n",digest)<0)failed=1;
    if(fflush(f)||fsync(fd))failed=1;
    if(fclose(f))failed=1;
    if(!failed && rename(temporary,path))failed=1;
    if(failed) {
        perror("polynomial atomic write");unlink(temporary);
        free(temporary);free(payload);exit(1);
    }
    free(temporary);free(payload);
}
