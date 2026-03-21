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

all: 5xn 6xn 6xn_poly 7xn_poly

$(NAUTY_BUILD_DIR)/.prepared:
	rm -rf $(NAUTY_BUILD_DIR)
	cp -R $(NAUTY_DIR) $(NAUTY_BUILD_DIR)
	touch $@

$(NAUTY_BUILD_DIR)/.configured-tls: $(NAUTY_BUILD_DIR)/.prepared
	cd $(NAUTY_BUILD_DIR) && ./configure $(NAUTY_CONFIGURE_FLAGS) CC="$(CC)" CFLAGS="$(NAUTY_BUILD_CFLAGS)"
	touch $@

$(NAUTY_BUILD_DIR)/nautyT.a: $(NAUTY_BUILD_DIR)/.configured-tls
	$(MAKE) -C $(NAUTY_BUILD_DIR) nautyT.a

5xn: 5xn.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

6xn: 6xn.c $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

6xn_poly: 6xn_poly.c $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

7xn_poly: 7xn_poly.c $(NAUTY_BUILD_DIR)/nautyT.a
	$(CC) $(PARTITION_CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f 5xn 6xn 6xn_poly 7xn_poly

clean-nauty:
	rm -rf $(NAUTY_BUILD_DIR)

.PHONY: all clean clean-nauty 5xn 6xn 6xn_poly 7xn_poly
