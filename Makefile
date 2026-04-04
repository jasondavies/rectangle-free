CC := gcc
MAKE ?= make
NAUTY_DIR ?= ./third_party/nauty
NAUTY_BUILD_DIR ?= ./third_party/nauty-build
NAUTY_CONFIGURE_FLAGS ?= --enable-tls
NAUTY_BUILD_CFLAGS ?= -O3 -march=native
LTO ?= 0
UNAME_S := $(shell uname -s)
LIBOMP_DIR ?= /opt/homebrew/opt/libomp

ifeq ($(UNAME_S),Darwin)
OPENMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(LIBOMP_DIR)/include
OPENMP_LDFLAGS ?= -L$(LIBOMP_DIR)/lib -lomp
else
OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?=
endif

PARTITION_CFLAGS ?= -O3 -march=native $(OPENMP_CFLAGS) -DUSE_TLS -I$(NAUTY_BUILD_DIR) -I$(NAUTY_DIR)
PARTITION_PROFILE_CFLAGS ?= -DRECT_PROFILE=1
LDFLAGS ?= $(NAUTY_BUILD_DIR)/nautyT.a -lm $(OPENMP_LDFLAGS)
PARTITION_POLY_7_NAUTY_CFLAGS ?= -DWORDSIZE=64 -DMAXN=WORDSIZE
PARTITION_POLY_7_CACHE_CFLAGS ?= -DRAW_CACHE_BITS=15 -DRAW_CACHE_PROBE=12
PARTITION_POLY_7_LDFLAGS ?= $(NAUTY_BUILD_DIR)/nautyTL1.a -lm $(OPENMP_LDFLAGS)
PARTITION_SHARED_SRCS := partition_poly.c src/runtime.c src/partitions.c

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89 -std=c++17 -I./inspiration/cpads/include

CFLAGS_5XN ?= -O3 -march=native -std=c11

ifneq ($(LTO),0)
NAUTY_BUILD_CFLAGS += -flto
PARTITION_CFLAGS += -flto
LDFLAGS += -flto
PARTITION_POLY_7_LDFLAGS += -flto
CFLAGS_5XN += -flto
endif

all: 5xn_count4 partition_count4 partition_poly partition_poly_7 partition_poly_profile partition_poly_7_profile small_graph_lookup_gen connected_canon_lookup_gen

$(NAUTY_BUILD_DIR)/.prepared:
	rm -rf $(NAUTY_BUILD_DIR)
	cp -R $(NAUTY_DIR) $(NAUTY_BUILD_DIR)
	touch $@

$(NAUTY_BUILD_DIR)/.configured-tls: $(NAUTY_BUILD_DIR)/.prepared
	cd $(NAUTY_BUILD_DIR) && ./configure $(NAUTY_CONFIGURE_FLAGS) CC="$(CC)" CFLAGS="$(NAUTY_BUILD_CFLAGS)"
	touch $@

$(NAUTY_BUILD_DIR)/nautyT.a: $(NAUTY_BUILD_DIR)/.configured-tls
	$(MAKE) -C $(NAUTY_BUILD_DIR) nautyT.a

$(NAUTY_BUILD_DIR)/nautyTL1.a: $(NAUTY_BUILD_DIR)/.configured-tls
	$(MAKE) -C $(NAUTY_BUILD_DIR) nautyTL1.a

5xn_count4: 5xn_count4.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

partition_count4: $(PARTITION_SHARED_SRCS) $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -DRECT_COUNT_K4=1 -DRECT_COUNT_K4_FEASIBILITY=1 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly: $(PARTITION_SHARED_SRCS) $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly_profile: $(PARTITION_SHARED_SRCS) $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly_7: $(PARTITION_SHARED_SRCS) $(NAUTY_BUILD_DIR)/nautyTL1.a
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_7_NAUTY_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=17 -o $@ $(PARTITION_SHARED_SRCS) $(PARTITION_POLY_7_LDFLAGS)

partition_poly_7_profile: $(PARTITION_SHARED_SRCS) $(NAUTY_BUILD_DIR)/nautyTL1.a
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) $(PARTITION_POLY_7_NAUTY_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=17 -o $@ $(PARTITION_SHARED_SRCS) $(PARTITION_POLY_7_LDFLAGS)

small_graph_lookup_gen: small_graph_lookup_gen.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

connected_canon_lookup_gen: connected_canon_lookup_gen.c src/runtime.c src/partitions.c $(NAUTY_BUILD_DIR)/nautyTL1.a
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_7_NAUTY_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=17 -o $@ connected_canon_lookup_gen.c src/runtime.c src/partitions.c $(PARTITION_POLY_7_LDFLAGS)

clean:
	rm -f 5xn_count4 partition_count4 partition_poly partition_poly_7 partition_poly_profile partition_poly_7_profile small_graph_lookup_gen connected_canon_lookup_gen

clean-nauty:
	rm -rf $(NAUTY_BUILD_DIR)

.PHONY: all clean clean-nauty 5xn_count4 partition_count4 partition_poly partition_poly_profile partition_poly_7 partition_poly_7_profile small_graph_lookup_gen connected_canon_lookup_gen
