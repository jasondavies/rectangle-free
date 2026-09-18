#include "cut_export.hpp"
int main(int argc,char** argv) try {
    initialise_tables();
    if(argc==2&&std::string(argv[1])=="--self-test"){reuse_cut::self_test();return 0;}
    if(argc!=3&&argc!=4)throw std::runtime_error("usage: INPUT_TSV four|all [SHORTLIST_TSV] | --self-test");
    reuse_cut::run(argv[1],argv[2],argc==4?argv[3]:"");
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
