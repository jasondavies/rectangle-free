CC := gcc
MAKE ?= make
NAUTY_DIR ?= ./third_party/nauty
NAUTY_BUILD_DIR ?= ./third_party/nauty-build
NAUTY_CONFIGURE_FLAGS ?= --enable-tls
NAUTY_BUILD_CFLAGS ?= -O3 -march=native
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
LDFLAGS ?= $(NAUTY_BUILD_DIR)/nautyT.a -lm $(OPENMP_LDFLAGS)

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89 -std=c++17 -I./inspiration/cpads/include

CFLAGS_5XN ?= -O3 -march=native -std=c11

all: 5xn_count4 partition_count4 partition_poly partition_poly_7 small_graph_lookup_gen

$(NAUTY_BUILD_DIR)/.prepared:
	rm -rf $(NAUTY_BUILD_DIR)
	cp -R $(NAUTY_DIR) $(NAUTY_BUILD_DIR)
	touch $@

$(NAUTY_BUILD_DIR)/.configured-tls: $(NAUTY_BUILD_DIR)/.prepared
	cd $(NAUTY_BUILD_DIR) && ./configure $(NAUTY_CONFIGURE_FLAGS) CC="$(CC)" CFLAGS="$(NAUTY_BUILD_CFLAGS)"
	touch $@

$(NAUTY_BUILD_DIR)/nautyT.a: $(NAUTY_BUILD_DIR)/.configured-tls
	$(MAKE) -C $(NAUTY_BUILD_DIR) nautyT.a

5xn_count4: 5xn_count4.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

partition_count4: partition_count4.c $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

partition_poly: partition_poly.c $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

partition_poly_7: partition_poly.c $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=17 -o $@ $< $(LDFLAGS)

small_graph_lookup_gen: small_graph_lookup_gen.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

clean:
	rm -f 5xn_count4 partition_count4 partition_poly partition_poly_7 small_graph_lookup_gen

clean-nauty:
	rm -rf $(NAUTY_BUILD_DIR)

.PHONY: all clean clean-nauty 5xn_count4 partition_count4 partition_poly partition_poly_7 small_graph_lookup_gen
