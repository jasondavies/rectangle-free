CC := gcc
MAKE ?= make
NAUTY_DIR ?= ./third_party/nauty
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

PARTITION_CFLAGS ?= -O3 -march=native $(OPENMP_CFLAGS) -DUSE_TLS -I$(NAUTY_DIR)
LDFLAGS ?= $(NAUTY_DIR)/nautyT.a -lm $(OPENMP_LDFLAGS)

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_89 -std=c++17 -I./inspiration/cpads/include

CFLAGS_5XN ?= -O3 -march=native -std=c11

all: partition_poly partition_count4 5xn

$(NAUTY_DIR)/.configured-tls:
	cd $(NAUTY_DIR) && ./configure $(NAUTY_CONFIGURE_FLAGS) CC="$(CC)" CFLAGS="$(NAUTY_BUILD_CFLAGS)"
	touch $@

$(NAUTY_DIR)/nautyT.a: $(NAUTY_DIR)/.configured-tls
	$(MAKE) -C $(NAUTY_DIR) nautyT.a

partition_poly: partition_poly.c $(NAUTY_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

partition_count4: partition_count4.c $(NAUTY_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

5xn: 5xn.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

# Backwards-compatible aliases for the old solver names.
solver: partition_poly

solver4: partition_count4

solver_5xn: 5xn

clean:
	rm -f partition_poly partition_count4 5xn

clean-nauty:
	-$(MAKE) -C $(NAUTY_DIR) clean
	rm -f $(NAUTY_DIR)/nautyT.a $(NAUTY_DIR)/.configured-tls

.PHONY: all clean clean-nauty solver solver4 solver_5xn
