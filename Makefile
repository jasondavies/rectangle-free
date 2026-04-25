CC := gcc
MAKE ?= make
LTO ?= 1
UNAME_S := $(shell uname -s)
LIBOMP_DIR ?= /opt/homebrew/opt/libomp

ifeq ($(UNAME_S),Darwin)
OPENMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(LIBOMP_DIR)/include
OPENMP_LDFLAGS ?= -L$(LIBOMP_DIR)/lib -lomp
else
OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?=
endif

PARTITION_CFLAGS ?= -O3 -march=native $(OPENMP_CFLAGS) -DUSE_TLS
PARTITION_PROFILE_CFLAGS ?= -DRECT_PROFILE=1
PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS ?= -DDEFAULT_ADAPTIVE_SUBDIVIDE=1 -DDEFAULT_ADAPTIVE_MAX_DEPTH=5 -DDEFAULT_ADAPTIVE_WORK_BUDGET=1000
LDFLAGS ?= -lm $(OPENMP_LDFLAGS)
PARTITION_POLY_7_CACHE_CFLAGS ?= -DRAW_CACHE_BITS=14 -DRAW_CACHE_PROBE=16 -DCACHE_PROBE=12 -DDEFAULT_HARD_CACHE_BITS=22 -DDEFAULT_HARD_CACHE_MAX_ENTRIES=2000000
PARTITION_SHARED_SRCS := src/runtime.c src/partitions.c src/poly.c src/graph.c src/cache.c src/main.c src/solver.c src/canon.c

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89 -std=c++17 -I./inspiration/cpads/include

CFLAGS_5XN ?= -O3 -march=native -std=c11

ifneq ($(LTO),0)
PARTITION_CFLAGS += -flto
LDFLAGS += -flto
CFLAGS_5XN += -flto
endif

all: 5xn_count4 partition_count4 partition_poly partition_poly_7 partition_poly_profile partition_poly_7_profile small_graph_lookup_gen

5xn_count4: 5xn_count4.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

partition_count4: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) -DRECT_COUNT_K4=1 -DRECT_COUNT_K4_FEASIBILITY=1 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly_profile: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly_7: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_ROWS=7 -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

partition_poly_7_profile: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_ROWS=7 -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

small_graph_lookup_gen: small_graph_lookup_gen.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

clean:
	rm -f 5xn_count4 partition_count4 partition_poly partition_poly_7 partition_poly_profile partition_poly_7_profile small_graph_lookup_gen

.PHONY: all clean 5xn_count4 partition_count4 partition_poly partition_poly_profile partition_poly_7 partition_poly_7_profile small_graph_lookup_gen
