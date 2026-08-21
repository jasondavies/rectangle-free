// Heavy-bucket census and exact CPU query comparison.  This statement-level
// include intentionally depends on the profiling locals at its inclusion site.
        if (right_layout.host_buckets.size() != right_layout.bucket_count) {
            throw std::runtime_error("profile structure buckets unavailable");
        }
        std::vector<unsigned long long> host_profile_pairs(
            right_layout.bucket_count);
        std::vector<unsigned long long> host_profile_queries(
            right_layout.bucket_count);
        std::vector<unsigned long long> host_profile_popcounts(
            right_layout.bucket_count);
        std::vector<unsigned long long> host_profile_comparisons(
            right_layout.bucket_count);
        CUDA_CHECK(cudaMemcpy(host_profile_pairs.data(), device_profile_pairs,
                              profile_bucket_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_profile_queries.data(),
                              device_profile_queries, profile_bucket_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_profile_popcounts.data(),
                              device_profile_popcounts, profile_bucket_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_profile_comparisons.data(),
                              device_profile_comparisons,
                              profile_bucket_bytes, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> ranked(right_layout.bucket_count);
        std::iota(ranked.begin(), ranked.end(), uint32_t(0));
        size_t wanted = std::min<size_t>(16, ranked.size());
        std::partial_sort(
            ranked.begin(), ranked.begin() + wanted, ranked.end(),
            [&](uint32_t lhs, uint32_t rhs) {
                return host_profile_comparisons[lhs] >
                       host_profile_comparisons[rhs];
            });
        for (size_t rank = 0; rank < wanted; rank++) {
            uint32_t bucket_index = ranked[rank];
            if (!host_profile_comparisons[bucket_index]) continue;
            PrefixBucket bucket = right_layout.host_buckets[bucket_index];
            std::vector<PrefixEntry> entries(bucket.count);
            CUDA_CHECK(cudaMemcpy(
                entries.data(), right_layout.entries + bucket.entry_offset,
                entries.size() * sizeof(PrefixEntry), cudaMemcpyDeviceToHost));
            std::vector<uint32_t> suffixes(entries.size());
            for (size_t index = 0; index < entries.size(); index++) {
                suffixes[index] = entries[index].suffix;
            }
            std::sort(suffixes.begin(), suffixes.end());
            uint64_t trie_nodes = 1;
            for (unsigned depth = 1; depth <= 32; depth++) {
                unsigned shift = 32 - depth;
                uint64_t distinct = 0;
                uint32_t previous = 0;
                bool have_previous = false;
                for (uint32_t suffix : suffixes) {
                    uint32_t prefix = shift ? suffix >> shift : suffix;
                    if (!have_previous || prefix != previous) {
                        distinct++;
                        previous = prefix;
                        have_previous = true;
                    }
                }
                trie_nodes += distinct;
            }
            std::vector<uint16_t> high(suffixes.size());
            std::vector<uint16_t> low(suffixes.size());
            for (size_t index = 0; index < suffixes.size(); index++) {
                high[index] = uint16_t(suffixes[index] >> 16);
                low[index] = uint16_t(suffixes[index]);
            }
            std::sort(high.begin(), high.end());
            std::sort(low.begin(), low.end());
            uint64_t unique_high =
                std::unique(high.begin(), high.end()) - high.begin();
            uint64_t unique_low =
                std::unique(low.begin(), low.end()) - low.begin();
            uint64_t patricia_nodes =
                suffixes.empty() ? 0 : uint64_t(suffixes.size()) * 2 - 1;
            constexpr uint64_t sos_table_bytes = UINT64_C(65536) * 8;
            std::printf(
                "SUFFIX_STRUCTURE batch=%llu rank=%zu bucket=%u count=%u "
                "queries=%llu query_popcounts=%llu compatible_pairs=%llu "
                "comparisons=%llu unique_high16=%llu unique_low16=%llu "
                "full_trie_nodes=%llu patricia_nodes=%llu "
                "high_group_sos_gib=%.6f low_group_sos_gib=%.6f\n",
                (unsigned long long)(result.right_batches + 1), rank + 1,
                bucket_index, bucket.count,
                host_profile_queries[bucket_index],
                host_profile_popcounts[bucket_index],
                host_profile_pairs[bucket_index],
                host_profile_comparisons[bucket_index],
                (unsigned long long)unique_high,
                (unsigned long long)unique_low,
                (unsigned long long)trie_nodes,
                (unsigned long long)patricia_nodes,
                double(unique_high * sos_table_bytes) /
                    (1024.0 * 1024.0 * 1024.0),
                double(unique_low * sos_table_bytes) /
                    (1024.0 * 1024.0 * 1024.0));
            if (!result.right_batches && !rank) {
                if (left_layout.host_buckets.size() !=
                    left_layout.bucket_count) {
                    throw std::runtime_error(
                        "profile left host buckets unavailable");
                }
                std::vector<PrefixEntry> queries;
                for (const PrefixJoinDesc& join : joins) {
                    if (bucket_index < join.right_bucket_offset ||
                        bucket_index >=
                            uint64_t(join.right_bucket_offset) +
                                join.right_bucket_count) {
                        continue;
                    }
                    for (uint32_t left_index = 0;
                         left_index < join.left_bucket_count; left_index++) {
                        PrefixBucket left = left_layout.host_buckets[
                            join.left_bucket_offset + left_index];
                        if (left.prefix & bucket.prefix) continue;
                        size_t old_size = queries.size();
                        queries.resize(old_size + left.count);
                        CUDA_CHECK(cudaMemcpy(
                            queries.data() + old_size,
                            left_layout.entries + left.entry_offset,
                            size_t(left.count) * sizeof(PrefixEntry),
                            cudaMemcpyDeviceToHost));
                    }
                }
                if (queries.size() != host_profile_queries[bucket_index]) {
                    throw std::runtime_error(
                        "Patricia query extraction mismatch");
                }
                std::sort(entries.begin(), entries.end(),
                          [](const PrefixEntry& lhs, const PrefixEntry& rhs) {
                              return lhs.suffix < rhs.suffix;
                          });
                double patricia_build_start = seconds_now();
                std::vector<HostSuffixPatriciaNode> nodes;
                nodes.reserve(entries.size() * 2);
                uint32_t root = build_host_suffix_patricia(
                    entries, 0, entries.size(), nodes);
                double patricia_build_seconds =
                    seconds_now() - patricia_build_start;
                uint64_t visits = 0;
                uint64_t weighted_total = 0;
                double patricia_query_start = seconds_now();
                for (size_t query_index = 0; query_index < queries.size();
                     query_index++) {
                    uint64_t right_weight = query_host_suffix_patricia(
                        nodes, root, queries[query_index].suffix, visits);
                    if (query_index < 4) {
                        uint64_t expected = 0;
                        for (const PrefixEntry& right : entries) {
                            if (!(queries[query_index].suffix & right.suffix)) {
                                expected += right.weight;
                            }
                        }
                        if (right_weight != expected) {
                            throw std::runtime_error(
                                "Patricia exact query mismatch");
                        }
                    }
                    weighted_total +=
                        uint64_t(queries[query_index].weight) * right_weight;
                }
                double patricia_query_seconds =
                    seconds_now() - patricia_query_start;
                uint64_t cartesian =
                    uint64_t(entries.size()) * uint64_t(queries.size());
                std::printf(
                    "SUFFIX_PATRICIA_CPU bucket=%u entries=%zu queries=%zu "
                    "nodes=%zu node_bytes=%zu visits=%llu "
                    "cartesian=%llu comparison_visit_ratio=%.6f "
                    "weighted_total=%llu build_seconds=%.6f "
                    "query_seconds=%.6f exact=OK\n",
                    bucket_index, entries.size(), queries.size(), nodes.size(),
                    nodes.size() * sizeof(HostSuffixPatriciaNode),
                    (unsigned long long)visits,
                    (unsigned long long)cartesian,
                    double(cartesian) / double(visits),
                    (unsigned long long)weighted_total,
                    patricia_build_seconds, patricia_query_seconds);

                double zdd_build_start = seconds_now();
                std::vector<HostSuffixZddNode> zdd_nodes;
                zdd_nodes.reserve(entries.size() * 4);
                zdd_nodes.emplace_back();
                std::unordered_map<uint64_t, uint32_t> zdd_terminals;
                std::unordered_map<HostSuffixZddKey, uint32_t,
                                   HostSuffixZddKeyHash>
                    zdd_unique;
                zdd_unique.reserve(entries.size() * 4);
                uint32_t zdd_root = build_host_suffix_zdd(
                    entries, 0, entries.size(), 31, zdd_nodes,
                    zdd_terminals, zdd_unique);
                double zdd_build_seconds = seconds_now() - zdd_build_start;
                uint64_t zdd_visits = 0;
                uint64_t zdd_weighted_total = 0;
                double zdd_query_start = seconds_now();
                for (size_t query_index = 0; query_index < queries.size();
                     query_index++) {
                    uint64_t right_weight = query_host_suffix_zdd(
                        zdd_nodes, zdd_root, queries[query_index].suffix,
                        zdd_visits);
                    if (query_index < 4) {
                        uint64_t expected = 0;
                        for (const PrefixEntry& right : entries) {
                            if (!(queries[query_index].suffix & right.suffix)) {
                                expected += right.weight;
                            }
                        }
                        if (right_weight != expected) {
                            throw std::runtime_error(
                                "weighted ZDD exact query mismatch");
                        }
                    }
                    zdd_weighted_total +=
                        uint64_t(queries[query_index].weight) * right_weight;
                }
                double zdd_query_seconds = seconds_now() - zdd_query_start;
                if (zdd_weighted_total != weighted_total) {
                    throw std::runtime_error(
                        "weighted ZDD aggregate mismatch");
                }
                std::printf(
                    "SUFFIX_ZDD_CPU bucket=%u entries=%zu queries=%zu "
                    "nodes=%zu terminals=%zu node_bytes=%zu visits=%llu "
                    "cartesian=%llu comparison_visit_ratio=%.6f "
                    "weighted_total=%llu build_seconds=%.6f "
                    "query_seconds=%.6f exact=OK\n",
                    bucket_index, entries.size(), queries.size(),
                    zdd_nodes.size(), zdd_terminals.size(),
                    zdd_nodes.size() * sizeof(HostSuffixZddNode),
                    (unsigned long long)zdd_visits,
                    (unsigned long long)cartesian,
                    double(cartesian) / double(zdd_visits),
                    (unsigned long long)zdd_weighted_total,
                    zdd_build_seconds, zdd_query_seconds);
            }
        }
