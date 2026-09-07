#include <cerrno>
#include <cstdio>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

static std::string fault, trace;
static bool interrupted = false;
static bool is_directory(int fd) {
    struct stat info{};
    if (fstat(fd, &info)) throw std::runtime_error("fstat failed");
    return S_ISDIR(info.st_mode);
}
static ssize_t test_write(int fd,const void* data,size_t size) {
    trace += 'W';
    if (fault == "write") { errno=ENOSPC; return -1; }
    if (fault == "short" && !interrupted) { interrupted=true; errno=EINTR; return -1; }
    return write(fd,data,fault == "short" && size>1 ? 1 : size);
}
static int test_sync(int fd) {
    bool dir=is_directory(fd); trace += dir ? 'D' : 'F';
    if (fault == (dir ? "directory" : "file")) { errno=EIO; return -1; }
    return fsync(fd);
}
static int test_close(int fd) {
    bool dir=is_directory(fd); trace += dir ? 'd' : 'f';
    int result=close(fd);
    if (!dir && fault == "close") { errno=EIO; return -1; }
    return result;
}
static int test_link(const char* from,const char* to) {
    trace += 'L'; if (fault == "link") { errno=EIO; return -1; }
    return link(from,to);
}
static int test_rename(const char* from,const char* to) {
    trace += 'R'; if (fault == "rename") { errno=EIO; return -1; }
    return rename(from,to);
}
#define RECT_FILE_WRITE test_write
#define RECT_FILE_FSYNC test_sync
#define RECT_FILE_CLOSE test_close
#define RECT_FILE_LINK test_link
#define RECT_FILE_RENAME test_rename
#include "../src/common/durable_file.h"

static void require(bool value) { if(!value)throw std::runtime_error("publication check failed: "+fault+" "+trace); }
static std::string contents(const std::filesystem::path& path) {
    std::ifstream f(path);return std::string(std::istreambuf_iterator<char>(f),{});
}
int main() {
    char pattern[]="/tmp/rectangle-durable-XXXXXX";
    char* name=mkdtemp(pattern);if(!name)return 1;
    std::filesystem::path root=name;
    try {
        for(int replace:{0,1})for(const char* mode:{"", "short", "write", "file", "close", "link", "rename", "directory"}) {
            auto target=root/"result";
            std::filesystem::remove(target);
            if(replace){std::ofstream out(target);out<<"old";}
            fault=mode;trace.clear();interrupted=false;
            RectFilePart parts[]={{"new",3},{" payload",8}};
            int result=rect_publish_file(target.c_str(),parts,2,replace);
            bool before_failure=fault=="write"||fault=="file"||fault=="close"||
                (replace?fault=="rename":fault=="link");
            require((result!=0)==(before_failure||fault=="directory"));
            if(before_failure) {
                require(replace?contents(target)=="old":!std::filesystem::exists(target));
            } else {
                require(contents(target)=="new payload");
                require(trace.find('F')<trace.find('f') && trace.find('f')<trace.find(replace?'R':'L') &&
                        trace.find(replace?'R':'L')<trace.find('D'));
            }
            for(auto& entry:std::filesystem::directory_iterator(root))require(entry.path()==target);
        }
        auto target=root/"result";fault.clear();trace.clear();
        RectFilePart part{"replacement",11};
        require(rect_publish_file(target.c_str(),&part,1,0)<0 && errno==EEXIST);
        require(contents(target)=="new payload");
        std::filesystem::remove_all(root);
        std::cout<<"DURABLE_FILE_TEST exact=OK cases=17\n";
    } catch (...) {std::filesystem::remove_all(root);throw;}
}
