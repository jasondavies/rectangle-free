CC := gcc
MAKE ?= make
LTO ?= 1
BUILD_DIR ?= build
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
PARTITION_POLY_8_CACHE_CFLAGS ?= -DDEFAULT_HARD_CACHE_BITS=22 -DDEFAULT_HARD_CACHE_MAX_ENTRIES=2000000 -DDEFAULT_TREEWIDTH_LIMIT=5 -DDEFAULT_TREEWIDTH_MIN_N=18 -DDEFAULT_TERMINAL_AGGREGATE_BITS=12 -DDEFAULT_TERMINAL_AGGREGATE_MULTI_BITS=10
PARTITION_POLY_8_PGO_DIR := $(abspath $(BUILD_DIR))/partition_poly_8_pgo
PARTITION_SHARED_SRCS := src/partition/runtime.c src/partition/partitions.c src/partition/poly.c src/partition/graph.c src/partition/cache.c src/partition/main.c src/partition/solver.c src/partition/treewidth.c src/partition/aggregate.c src/partition/canon.c

NVCC ?= nvcc
PACKED_PREFETCH_MIB ?= 4608
NVCCFLAGS ?= -O3 -arch=sm_89 -std=c++17

.PHONY: gpu-production
gpu-production: twocolour_7x7_solve_gpu twocolour_7x9_solve_gpu \
		twocolour_7x9_four_owner_gpu twocolour_7x9_cache_build \
		twocolour_8x8_solve_gpu

.PHONY: gpu-code-dump
gpu-code-dump:
	python3 tools/make_gpu_code_dump.py $(BUILD_DIR)/code-dump.txt

$(BUILD_DIR)/gpu_result_checkpoint_test: tests/gpu/gpu_result_checkpoint_test.cpp \
		src/gpu/gpu_result_checkpoint.hpp src/common/sha256.hpp
	$(CXX) -O2 -std=c++17 -Isrc/gpu -o $@ $<

gpu-campaign-test:
	python3 -m unittest -v tests.gpu.test_aggregate_gpu_v3

CFLAGS_5XN ?= -O3 -march=native -std=c11

ifneq ($(LTO),0)
PARTITION_CFLAGS += -flto
LDFLAGS += -flto
CFLAGS_5XN += -flto
endif

all: 5xn_count4 partition_count4 partition_poly partition_poly_7 partition_poly_8 partition_poly_profile partition_poly_7_profile partition_poly_8_profile small_graph_lookup_gen

$(BUILD_DIR)/5xn_count4: src/small/5xn_count4.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/partition_count4: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) -DRECT_COUNT_K4=1 -DRECT_COUNT_K4_FEASIBILITY=1 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/partition_poly: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/partition_poly_profile: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/partition_poly_7: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_ROWS=7 -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/partition_poly_7_profile: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_7_CACHE_CFLAGS) -DMAX_ROWS=7 -DMAX_COLS=7 -DDEFAULT_ROWS=7 -DDEFAULT_COLS=7 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/partition_poly_8: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_8_CACHE_CFLAGS) -DMAX_ROWS=8 -DMAX_COLS=8 -DDEFAULT_ROWS=8 -DDEFAULT_COLS=8 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/partition_poly_8_pgo: $(PARTITION_SHARED_SRCS)
	$(RM) -r $(PARTITION_POLY_8_PGO_DIR)
	mkdir -p $(PARTITION_POLY_8_PGO_DIR)
	$(CC) $(PARTITION_CFLAGS) -fprofile-generate=$(PARTITION_POLY_8_PGO_DIR) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_8_CACHE_CFLAGS) -DMAX_ROWS=8 -DMAX_COLS=8 -DDEFAULT_ROWS=8 -DDEFAULT_COLS=8 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS) -fprofile-generate=$(PARTITION_POLY_8_PGO_DIR)
	OMP_NUM_THREADS=1 RECT_PROGRESS_STEP=1000000 ./$@ 8 5 --prefix-depth 2 --task-start 0 --task-end 1 >/dev/null
	$(CC) $(PARTITION_CFLAGS) -fprofile-use=$(PARTITION_POLY_8_PGO_DIR) -fprofile-correction $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_8_CACHE_CFLAGS) -DMAX_ROWS=8 -DMAX_COLS=8 -DDEFAULT_ROWS=8 -DDEFAULT_COLS=8 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS) -fprofile-use=$(PARTITION_POLY_8_PGO_DIR) -fprofile-correction

$(BUILD_DIR)/partition_poly_8_profile: $(PARTITION_SHARED_SRCS)
	$(CC) $(PARTITION_CFLAGS) $(PARTITION_PROFILE_CFLAGS) $(PARTITION_POLY_DEFAULT_ADAPTIVE_CFLAGS) $(PARTITION_POLY_8_CACHE_CFLAGS) -DMAX_ROWS=8 -DMAX_COLS=8 -DDEFAULT_ROWS=8 -DDEFAULT_COLS=8 -DCACHE_BITS=18 -o $@ $(PARTITION_SHARED_SRCS) $(LDFLAGS)

$(BUILD_DIR)/small_graph_lookup_gen: tools/generators/small_graph_lookup_gen.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/right_prefix_overlap_census: research/probes/right_prefix_overlap_census.cpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) -o $@ $<

$(BUILD_DIR)/prefix_hierarchy_8x8_census: research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) -o $@ $<

$(BUILD_DIR)/pairmask_transfer_probe: research/probes/pairmask_transfer_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/completion_oracle_probe: research/probes/completion_oracle_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/c4free_zdd_probe: research/probes/c4free_zdd_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/dense_c4free_pair_probe: research/probes/dense_c4free_pair_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $<

$(BUILD_DIR)/dense_residual_hypergraph_probe: research/probes/dense_residual_hypergraph_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $< -lnauty

$(BUILD_DIR)/rectangle_closure_lattice_probe: research/probes/rectangle_closure_lattice_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $< -lnauty

$(BUILD_DIR)/dense_c4free_mitm_probe: research/gpu/dense_c4free_mitm_probe.cu
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

$(BUILD_DIR)/hafnian_int8_block_probe: research/gpu/hafnian_int8_block_probe.cu \
		src/hafnian/six_by_twenty_eight_catalog.hpp \
		src/hafnian/six_by_twenty_nine_catalog.hpp src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

$(BUILD_DIR)/hafnian_int8_sign_probe: research/gpu/hafnian_int8_sign_probe.cu \
		src/hafnian/six_by_twenty_eight_catalog.hpp src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

$(BUILD_DIR)/hafnian_blocked_hessenberg_probe: research/probes/hafnian_blocked_hessenberg_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $<

.PHONY: hafnian-blocked-hessenberg-test
hafnian-blocked-hessenberg-test: $(BUILD_DIR)/hafnian_blocked_hessenberg_probe
	./$(BUILD_DIR)/hafnian_blocked_hessenberg_probe --samples 20

$(BUILD_DIR)/clique_pivoter_probe: research/probes/clique_pivoter_probe.c
	$(CC) $(CFLAGS_5XN) $(OPENMP_CFLAGS) -o $@ $< $(OPENMP_LDFLAGS)

$(BUILD_DIR)/column_tensor_rank_probe: research/probes/column_tensor_rank_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twobit_decomposition_probe: research/probes/twobit_decomposition_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/binary_prefix_orbit_probe: research/probes/binary_prefix_orbit_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twobit_orbit_contraction_probe: research/probes/twobit_orbit_contraction_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twobit_full_orbit_probe: research/probes/twobit_full_orbit_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twocolour_prefix_distribution_probe: research/probes/twocolour_prefix_distribution_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/binary_orbit_burnside_probe: research/probes/binary_orbit_burnside_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twocolour_7x5_canonical_census: archive/gpu/twocolour_7x5_canonical_census.cu \
		src/gpu/twocolour_7x7_gpu.cu src/gpu/twocolour_7x7_engine.cuh \
		src/gpu/twocolour_gpu_common.cuh
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fopenmp -o $@ $<

$(BUILD_DIR)/binary_orbit_augment: tools/corpus/binary_orbit_augment.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/binary_orbit_augment_6x9: tools/corpus/binary_orbit_augment.c
	$(CC) $(CFLAGS_5XN) -DORBIT_ROWS=6 -DORBIT_MAX_COLUMNS=9 \
		-DORBIT_ROW_BITS=10 -DORBIT_MAGIC='"R6ORB01"' -o $@ $<

$(BUILD_DIR)/binary_orbit_augment_7x8: tools/corpus/binary_orbit_augment.c
	$(CC) $(CFLAGS_5XN) -DORBIT_ROWS=7 -DORBIT_MAX_COLUMNS=8 \
		-DORBIT_ROW_BITS=8 -DORBIT_MAGIC='"R7ORB01"' -o $@ $<

$(BUILD_DIR)/binary_orbit_augment_7x9: tools/corpus/binary_orbit_augment.c
	$(CC) $(CFLAGS_5XN) -DORBIT_ROWS=7 -DORBIT_MAX_COLUMNS=9 \
		-DORBIT_ROW_BITS=9 -DORBIT_MAGIC='"R7ORB09"' -o $@ $<

$(BUILD_DIR)/binary_orbit_augment_8x8: tools/corpus/binary_orbit_augment.c
	$(CC) $(CFLAGS_5XN) $(OPENMP_CFLAGS) \
		-DORBIT_ROWS=8 -DORBIT_MAX_COLUMNS=8 \
		-DORBIT_ROW_BITS=8 -DORBIT_MAGIC='"R8ORB01"' \
		-o $@ $< $(OPENMP_LDFLAGS)

$(BUILD_DIR)/s8_prefix_module_probe: research/probes/s8_prefix_module_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/symmetric_kernel_rank_probe: research/probes/symmetric_kernel_rank_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twocolour_3x3_sampler: research/probes/twocolour_3x3_sampler.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twocolour_4x4_probe: research/probes/twocolour_4x4_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/canonical_query_circuit_probe: research/probes/canonical_query_circuit_probe.c \
		research/probes/twocolour_4x4_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/token_plane_quotient_probe: research/probes/token_plane_quotient_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $<

$(BUILD_DIR)/six_by_thirty_matching_probe: research/probes/six_by_thirty_matching_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) -o $@ $< $(OPENMP_LDFLAGS)

$(BUILD_DIR)/six_by_twenty_eight_defect_census: research/probes/six_by_twenty_eight_defect_census.cpp \
		src/hafnian/six_by_twenty_nine_catalog.hpp src/common/sha256.hpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) -o $@ $< \
		$(OPENMP_LDFLAGS) -lnauty

$(BUILD_DIR)/colour_plane_permanent_probe: research/probes/colour_plane_permanent_probe.cpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $<

.PHONY: six-by-twenty-eight-census-test
six-by-twenty-eight-census-test: six_by_twenty_eight_defect_census
	./$(BUILD_DIR)/six_by_twenty_eight_defect_census --slack 1 --threads 2 --raw \
		| grep -q 'raw_unions=83071 symmetry_orbits=29.*exact=OK'
	./$(BUILD_DIR)/six_by_twenty_eight_defect_census --slack 2 --threads 2 --graph-isomorphism \
		| grep -q 'symmetry_orbits=36398 graph_orbits=36398.*exact=OK'
	./$(BUILD_DIR)/six_by_twenty_eight_defect_census --slack 2 --threads 2 \
		| grep -q 'DEFECT28_FRIEDLAND_BOUND .*required_31bit_primes=8 exact=OK'

.PHONY: colour-plane-permanent-test
colour-plane-permanent-test: colour_plane_permanent_probe
	./$(BUILD_DIR)/colour_plane_permanent_probe --samples 500 --rank-sample 128 \
		| grep -q 'maximum_local_assignments=756756.*sampled_bond_rank=128 exact=OK'

$(BUILD_DIR)/six_by_thirty_hafnian: src/hafnian/six_by_thirty_hafnian.cpp src/common/sha256.hpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) -o $@ $< $(OPENMP_LDFLAGS)

$(BUILD_DIR)/six_by_thirty_hafnian_gpu: src/hafnian/six_by_thirty_hafnian_gpu.cu src/hafnian/hafnian_gpu_core.cuh src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

$(BUILD_DIR)/six_by_twenty_nine_hafnian_cpu: src/hafnian/six_by_twenty_nine_hafnian_cpu.cpp \
		src/hafnian/six_by_twenty_nine_catalog.hpp src/common/sha256.hpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) -o $@ $< $(OPENMP_LDFLAGS)

$(BUILD_DIR)/six_by_twenty_eight_catalog_test: tests/hafnian/six_by_twenty_eight_catalog_test.cpp \
		src/hafnian/six_by_twenty_eight_catalog.hpp \
		src/hafnian/six_by_twenty_nine_catalog.hpp src/common/sha256.hpp
	$(CXX) -O3 -march=native -std=c++17 -o $@ $<

$(BUILD_DIR)/fixed_montgomery_arithmetic_test: \
		tests/hafnian/fixed_montgomery_arithmetic_test.cu src/hafnian/hafnian_gpu_core.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

.PHONY: fixed-montgomery-arithmetic-test
fixed-montgomery-arithmetic-test: $(BUILD_DIR)/fixed_montgomery_arithmetic_test
	./$(BUILD_DIR)/fixed_montgomery_arithmetic_test

$(BUILD_DIR)/six_by_twenty_eight_hafnian_cpu: src/hafnian/six_by_twenty_nine_hafnian_cpu.cpp \
		src/hafnian/six_by_twenty_eight_catalog.hpp \
		src/hafnian/six_by_twenty_nine_catalog.hpp src/common/sha256.hpp
	$(CXX) -O3 -march=native -std=c++17 $(OPENMP_CFLAGS) \
		-DSIX_BY_TWENTY_EIGHT=1 -o $@ $< $(OPENMP_LDFLAGS)

$(BUILD_DIR)/six_by_twenty_nine_hafnian_gpu: src/hafnian/six_by_twenty_nine_hafnian_gpu.cu \
		src/hafnian/six_by_twenty_nine_catalog.hpp src/hafnian/hafnian_gpu_core.cuh src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

$(BUILD_DIR)/six_by_twenty_eight_hafnian_gpu: src/hafnian/six_by_twenty_eight_hafnian_gpu.cu \
		src/hafnian/six_by_twenty_eight_catalog.hpp src/hafnian/six_by_twenty_nine_catalog.hpp \
		src/hafnian/hafnian_gpu_core.cuh src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -o $@ $<

$(BUILD_DIR)/six_by_twenty_eight_runtime_montgomery_control: \
		src/hafnian/six_by_twenty_eight_hafnian_gpu.cu \
		src/hafnian/six_by_twenty_eight_catalog.hpp src/hafnian/six_by_twenty_nine_catalog.hpp \
		src/hafnian/hafnian_gpu_core.cuh src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -DHAFNIAN_RUNTIME_MONTGOMERY_CONTROL=1 -o $@ $<

.PHONY: six-by-twenty-nine-hafnian-test
six-by-twenty-nine-hafnian-test: six_by_twenty_nine_hafnian_cpu
	./$(BUILD_DIR)/six_by_twenty_nine_hafnian_cpu --query 0 --prime 2147483647 --begin 0 --end 16 --threads 1 | \
		grep -q 'residue=791700040.*exact=OK'
	python3 -m unittest -v tests.hafnian.test_six_by_twenty_nine_hafnian

.PHONY: six-by-twenty-eight-hafnian-test
six-by-twenty-eight-hafnian-test: six_by_twenty_eight_catalog_test \
		six_by_twenty_eight_hafnian_cpu
	./$(BUILD_DIR)/six_by_twenty_eight_catalog_test | \
		grep -q 'queries=36398.*prime_histogram_3=36395 prime_histogram_4=3.*exact=OK'
	./$(BUILD_DIR)/six_by_twenty_eight_hafnian_cpu --query 0 --prime 2147483647 \
		--begin 0 --end 16 --threads 1 | grep -q 'residue=2020296484.*exact=OK'
	./$(BUILD_DIR)/six_by_twenty_eight_hafnian_cpu --query 3321 --prime 2147483647 \
		--begin 0 --end 16 --threads 1 | grep -q 'residue=1061461801.*exact=OK'
	python3 -m unittest -v tests.hafnian.test_six_by_twenty_eight_hafnian

.PHONY: six-by-thirty-hafnian-test
six-by-thirty-hafnian-test: six_by_thirty_hafnian
	./$(BUILD_DIR)/six_by_thirty_hafnian --self-test
	python3 -m unittest -v tests.hafnian.test_six_by_thirty_hafnian

$(BUILD_DIR)/twocolour_gpu_64.bin: twocolour_4x4_probe
	./$(BUILD_DIR)/twocolour_4x4_probe 1024 0 64 -1 0 $@

$(BUILD_DIR)/twocolour_gpu_bench: archive/gpu/twocolour_gpu_bench.cu
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -lineinfo -o $@ $<

$(BUILD_DIR)/twocolour_7x7_solve_gpu: src/gpu/twocolour_7x7_gpu.cu src/gpu/twocolour_7x7_engine.cuh \
		src/gpu/twocolour_gpu_common.cuh
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fopenmp -o $@ $<

$(BUILD_DIR)/twocolour_7x7_prefix_gpu: legacy/gpu/twocolour_prefix_legacy_main.cu \
		legacy/gpu/twocolour_prefix_legacy_helpers.cuh \
		legacy/gpu/twocolour_prefix_legacy_layout.cuh \
		src/gpu/twocolour_prefix_core.cuh src/gpu/twocolour_prefix_algebra.cuh
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fopenmp \
		-o $@ $<

$(BUILD_DIR)/twocolour_6x9_gpu: src/gpu/twocolour_7x7_gpu.cu src/gpu/twocolour_7x7_engine.cuh \
		src/gpu/twocolour_gpu_common.cuh
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fopenmp \
		-DGRID_ROWS=6 -DGRID_COLUMNS=9 -DLEFT_COLUMNS=4 -DRIGHT_COLUMNS=5 \
		-DORBIT_ROW_BITS=10 -DORBIT_MAGIC='"R6ORB01"' -o $@ $<

$(BUILD_DIR)/twocolour_7x8_gpu: src/gpu/twocolour_7x7_gpu.cu src/gpu/twocolour_7x7_engine.cuh \
		src/gpu/twocolour_gpu_common.cuh
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fopenmp \
		-DGRID_ROWS=7 -DGRID_COLUMNS=8 -DLEFT_COLUMNS=4 -DRIGHT_COLUMNS=4 \
		-DORBIT_ROW_BITS=8 -DORBIT_MAGIC='"R7ORB01"' -o $@ $<

$(BUILD_DIR)/twocolour_7x8_prefix_gpu: legacy/gpu/twocolour_prefix_legacy_main.cu \
		legacy/gpu/twocolour_prefix_legacy_helpers.cuh \
		legacy/gpu/twocolour_prefix_legacy_layout.cuh \
		src/gpu/twocolour_prefix_core.cuh src/gpu/twocolour_prefix_algebra.cuh
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fopenmp \
		-DGRID_ROWS=7 -DGRID_COLUMNS=8 -DLEFT_COLUMNS=4 -DRIGHT_COLUMNS=4 \
		-DORBIT_ROW_BITS=8 -DORBIT_MAGIC='"R7ORB01"' -o $@ $<

$(BUILD_DIR)/twocolour_7x9_prefix_gpu: legacy/gpu/twocolour_prefix_legacy_main.cu \
		legacy/gpu/twocolour_prefix_legacy_helpers.cuh \
		legacy/gpu/twocolour_prefix_legacy_layout.cuh \
		src/gpu/twocolour_prefix_core.cuh src/gpu/twocolour_prefix_algebra.cuh
	$(NVCC) -O3 -std=c++17 -arch=sm_89 -Xcompiler=-fopenmp \
		-DGPU_PREFIX_BUILDER \
		-DSTREAMED_RIGHT_PREFIX_PROBE \
		-DGRID_ROWS=7 -DGRID_COLUMNS=9 -DLEFT_COLUMNS=4 -DRIGHT_COLUMNS=5 \
		-DORBIT_ROW_BITS=9 -DORBIT_MAGIC='"R7ORB09"' -o $@ $<

$(BUILD_DIR)/twocolour_7x9_solve_gpu: src/gpu/twocolour_7x9_packed_solve.cu \
		src/gpu/twocolour_7x9_engine.cuh \
		src/gpu/twocolour_canonical_device.cuh \
		src/gpu/twocolour_weight_class_join.cuh src/gpu/gpu_cuda_utils.cuh \
		src/gpu/twocolour_prefix_core.cuh src/gpu/twocolour_prefix_algebra.cuh \
		src/gpu/twocolour_gpu_common.cuh src/gpu/gpu_memory_policy.hpp \
		src/gpu/gpu_result_checkpoint.hpp src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fopenmp \
		'-DPACKED_PREFETCH_BYTES=(UINT64_C($(PACKED_PREFETCH_MIB))<<20)' \
		-o $@ $<

$(BUILD_DIR)/twocolour_7x9_cache_build: src/gpu/twocolour_7x9_cache_build.cu \
		src/gpu/twocolour_prefix_core.cuh src/gpu/twocolour_prefix_algebra.cuh \
		src/gpu/twocolour_gpu_common.cuh src/gpu/gpu_cuda_utils.cuh src/gpu/gpu_memory_policy.hpp src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fopenmp -o $@ $<

$(BUILD_DIR)/twocolour_7x9_four_owner_gpu: src/gpu/twocolour_7x9_four_owner_solve.cu \
		src/gpu/twocolour_7x9_engine.cuh src/gpu/twocolour_canonical_device.cuh \
		src/gpu/twocolour_weight_class_join.cuh \
		src/gpu/gpu_cuda_utils.cuh src/gpu/twocolour_prefix_core.cuh \
		src/gpu/twocolour_prefix_algebra.cuh src/gpu/twocolour_gpu_common.cuh \
		src/gpu/gpu_memory_policy.hpp src/gpu/gpu_result_checkpoint.hpp src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fopenmp \
		'-DPACKED_PREFETCH_BYTES=(UINT64_C($(PACKED_PREFETCH_MIB))<<20)' \
		-o $@ $<

$(BUILD_DIR)/twocolour_8x8_solve_gpu: src/gpu/twocolour_8x8_prefix_solve.cu \
		src/gpu/twocolour_weight_class_join.cuh src/gpu/twocolour_canonical_device.cuh \
		src/gpu/gpu_cuda_utils.cuh \
		src/gpu/twocolour_prefix_algebra.cuh src/gpu/twocolour_gpu_common.cuh \
		src/gpu/gpu_memory_policy.hpp src/gpu/gpu_result_checkpoint.hpp src/common/sha256.hpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fopenmp -o $@ $<

$(BUILD_DIR)/prefix_portfolio_8x8_oracle: research/probes/prefix_portfolio_8x8_oracle.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/prefix_bucket_tt_rank_census: research/probes/prefix_bucket_tt_rank_census.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/prefix_bmma_cost_census: research/probes/prefix_bmma_cost_census.cpp \
		research/probes/prefix_bucket_tt_rank_census.cpp research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/prefix_bmma_portfolio_8x8_oracle: research/probes/prefix_bmma_portfolio_8x8_oracle.cpp \
		research/probes/prefix_bmma_cost_census.cpp research/probes/prefix_bucket_tt_rank_census.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/column_split_8x8_oracle: research/probes/column_split_8x8_oracle.cpp \
		research/probes/prefix_bmma_cost_census.cpp research/probes/prefix_bucket_tt_rank_census.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/column_split_8x8_transform: research/probes/column_split_8x8_transform.cpp
	$(CXX) -O3 -std=c++17 -o $@ $<

$(BUILD_DIR)/column_split_8x8_selector: research/probes/column_split_8x8_selector.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/pair_projection_8x8_census: research/probes/pair_projection_8x8_census.cpp \
		research/probes/prefix_bmma_cost_census.cpp research/probes/prefix_bucket_tt_rank_census.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/weight_class_bitset_8x8_census: research/probes/weight_class_bitset_8x8_census.cpp \
		research/probes/pair_projection_8x8_census.cpp research/probes/prefix_bmma_cost_census.cpp \
		research/probes/prefix_bucket_tt_rank_census.cpp research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/offline_row_gauge_8x8_census: research/probes/offline_row_gauge_8x8_census.cpp \
		research/probes/pair_projection_8x8_census.cpp research/probes/prefix_bmma_cost_census.cpp \
		research/probes/prefix_bucket_tt_rank_census.cpp research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/demanded_query_reuse_8x8_census: research/probes/demanded_query_reuse_8x8_census.cpp \
		research/probes/prefix_bucket_tt_rank_census.cpp research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/behavioral_distribution_8x8_census: research/probes/behavioral_distribution_8x8_census.cpp \
		research/probes/prefix_bmma_cost_census.cpp research/probes/prefix_bucket_tt_rank_census.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/colour_cut_cardinality_census: research/probes/colour_cut_cardinality_census.cpp \
		research/probes/prefix_bucket_tt_rank_census.cpp \
		research/probes/prefix_hierarchy_8x8_census.cpp
	$(CXX) -O3 -std=c++17 -fopenmp -march=native -o $@ $<

$(BUILD_DIR)/universal_state_symmetry_probe: research/probes/universal_state_symmetry_probe.cpp
	$(CXX) -O3 -std=c++17 -march=native -o $@ $<

$(BUILD_DIR)/twocolour_3x4_probe: research/probes/twocolour_3x4_probe.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

$(BUILD_DIR)/twocolour_7x7_solve: src/gpu/twocolour_7x7_solve.c
	$(CC) $(CFLAGS_5XN) -o $@ $<

BUILD_TARGETS := 5xn_count4 partition_count4 partition_poly partition_poly_7 \
	partition_poly_8 partition_poly_8_pgo partition_poly_profile \
	partition_poly_7_profile partition_poly_8_profile small_graph_lookup_gen \
	right_prefix_overlap_census prefix_hierarchy_8x8_census \
	pairmask_transfer_probe completion_oracle_probe c4free_zdd_probe \
	dense_c4free_pair_probe dense_residual_hypergraph_probe rectangle_closure_lattice_probe \
	dense_c4free_mitm_probe hafnian_int8_block_probe \
	clique_pivoter_probe column_tensor_rank_probe twobit_decomposition_probe \
	binary_prefix_orbit_probe twobit_orbit_contraction_probe \
	twobit_full_orbit_probe twocolour_prefix_distribution_probe \
	binary_orbit_burnside_probe twocolour_7x5_canonical_census \
	binary_orbit_augment binary_orbit_augment_6x9 binary_orbit_augment_7x8 \
	binary_orbit_augment_7x9 binary_orbit_augment_8x8 s8_prefix_module_probe \
	symmetric_kernel_rank_probe twocolour_3x3_sampler twocolour_4x4_probe \
	canonical_query_circuit_probe token_plane_quotient_probe \
	six_by_thirty_matching_probe six_by_twenty_eight_defect_census \
	colour_plane_permanent_probe \
	six_by_thirty_hafnian \
	six_by_thirty_hafnian_gpu \
	six_by_twenty_eight_catalog_test \
	six_by_twenty_eight_hafnian_cpu \
	six_by_twenty_eight_hafnian_gpu \
	six_by_twenty_nine_hafnian_cpu \
	six_by_twenty_nine_hafnian_gpu \
	twocolour_gpu_64.bin twocolour_gpu_bench twocolour_7x7_solve_gpu \
	twocolour_7x7_prefix_gpu twocolour_6x9_gpu twocolour_7x8_gpu \
	twocolour_7x8_prefix_gpu twocolour_7x9_prefix_gpu \
	twocolour_7x9_solve_gpu twocolour_7x9_four_owner_gpu \
	twocolour_7x9_cache_build \
	twocolour_8x8_solve_gpu prefix_portfolio_8x8_oracle \
	prefix_bucket_tt_rank_census prefix_bmma_cost_census \
	colour_cut_cardinality_census \
	prefix_bmma_portfolio_8x8_oracle column_split_8x8_oracle \
	column_split_8x8_transform column_split_8x8_selector \
	pair_projection_8x8_census behavioral_distribution_8x8_census \
	demanded_query_reuse_8x8_census weight_class_bitset_8x8_census \
	offline_row_gauge_8x8_census \
	universal_state_symmetry_probe twocolour_3x4_probe twocolour_7x7_solve \
	gpu_result_checkpoint_test

.PHONY: $(BUILD_TARGETS)
$(BUILD_TARGETS): %: $(BUILD_DIR)/%

$(addprefix $(BUILD_DIR)/,$(BUILD_TARGETS)): | $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

clean:
	$(RM) -r $(BUILD_DIR)

.PHONY: all clean gpu-campaign-test
