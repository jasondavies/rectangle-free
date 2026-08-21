#pragma once

// Rejected exact Patricia/ZDD suffix-query prototypes retained for
// reproducibility.  Included only when PROFILE_SUFFIX_STRUCTURES is enabled.
struct HostSuffixPatriciaNode {
    uint32_t child_zero = UINT32_MAX;
    uint32_t child_one = UINT32_MAX;
    uint32_t suffix_or = 0;
    uint32_t suffix_and = 0;
    uint64_t weight = 0;
};

struct HostSuffixZddNode {
    uint32_t bit = UINT32_MAX;
    uint32_t zero = UINT32_MAX;
    uint32_t one = UINT32_MAX;
    uint32_t suffix_or = 0;
    uint64_t weight = 0;
};

struct HostSuffixZddKey {
    uint32_t bit;
    uint32_t zero;
    uint32_t one;

    bool operator==(const HostSuffixZddKey& other) const {
        return bit == other.bit && zero == other.zero && one == other.one;
    }
};

struct HostSuffixZddKeyHash {
    size_t operator()(const HostSuffixZddKey& key) const {
        return size_t(mix64(uint64_t(key.bit) << 56 ^
                            uint64_t(key.zero) << 28 ^ key.one));
    }
};

static uint32_t build_host_suffix_patricia(
    const std::vector<PrefixEntry>& entries, size_t begin, size_t end,
    std::vector<HostSuffixPatriciaNode>& nodes) {
    if (begin >= end) throw std::runtime_error("empty Patricia range");
    uint32_t node_index = uint32_t(nodes.size());
    nodes.emplace_back();
    if (end - begin == 1) {
        nodes[node_index].suffix_or = entries[begin].suffix;
        nodes[node_index].suffix_and = entries[begin].suffix;
        nodes[node_index].weight = entries[begin].weight;
        return node_index;
    }
    uint32_t differing = entries[begin].suffix ^ entries[end - 1].suffix;
    if (!differing) {
        throw std::runtime_error("duplicate suffix in Patricia input");
    }
    unsigned bit = 31U - unsigned(__builtin_clz(differing));
    size_t middle = begin;
    while (middle < end && !(entries[middle].suffix & (UINT32_C(1) << bit))) {
        middle++;
    }
    if (middle == begin || middle == end) {
        throw std::runtime_error("invalid Patricia split");
    }
    uint32_t zero =
        build_host_suffix_patricia(entries, begin, middle, nodes);
    uint32_t one = build_host_suffix_patricia(entries, middle, end, nodes);
    HostSuffixPatriciaNode& node = nodes[node_index];
    node.child_zero = zero;
    node.child_one = one;
    node.suffix_or = nodes[zero].suffix_or | nodes[one].suffix_or;
    node.suffix_and = nodes[zero].suffix_and & nodes[one].suffix_and;
    node.weight = nodes[zero].weight + nodes[one].weight;
    return node_index;
}

static uint64_t query_host_suffix_patricia(
    const std::vector<HostSuffixPatriciaNode>& nodes, uint32_t root,
    uint32_t suffix, uint64_t& visits) {
    std::array<uint32_t, 64> stack{};
    size_t stack_size = 0;
    stack[stack_size++] = root;
    uint64_t result = 0;
    while (stack_size) {
        const HostSuffixPatriciaNode& node = nodes[stack[--stack_size]];
        visits++;
        if (node.suffix_and & suffix) continue;
        if (!(node.suffix_or & suffix)) {
            result += node.weight;
            continue;
        }
        if (node.child_zero == UINT32_MAX) continue;
        if (stack_size + 2 > stack.size()) {
            throw std::runtime_error("Patricia query stack overflow");
        }
        stack[stack_size++] = node.child_zero;
        stack[stack_size++] = node.child_one;
    }
    return result;
}

static uint32_t build_host_suffix_zdd(
    const std::vector<PrefixEntry>& entries, size_t begin, size_t end, int bit,
    std::vector<HostSuffixZddNode>& nodes,
    std::unordered_map<uint64_t, uint32_t>& terminals,
    std::unordered_map<HostSuffixZddKey, uint32_t, HostSuffixZddKeyHash>&
        unique_nodes) {
    if (begin == end) return 0;
    if (bit < 0) {
        if (end - begin != 1) {
            throw std::runtime_error("duplicate weighted ZDD suffix");
        }
        uint64_t weight = entries[begin].weight;
        auto found = terminals.find(weight);
        if (found != terminals.end()) return found->second;
        uint32_t result = uint32_t(nodes.size());
        nodes.push_back(HostSuffixZddNode{UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                          0, weight});
        terminals.emplace(weight, result);
        return result;
    }
    size_t middle = begin;
    uint32_t bit_mask = UINT32_C(1) << unsigned(bit);
    while (middle < end && !(entries[middle].suffix & bit_mask)) middle++;
    uint32_t zero = build_host_suffix_zdd(
        entries, begin, middle, bit - 1, nodes, terminals, unique_nodes);
    uint32_t one = build_host_suffix_zdd(
        entries, middle, end, bit - 1, nodes, terminals, unique_nodes);
    if (!one) return zero;
    HostSuffixZddKey key{uint32_t(bit), zero, one};
    auto found = unique_nodes.find(key);
    if (found != unique_nodes.end()) return found->second;
    uint32_t result = uint32_t(nodes.size());
    uint32_t suffix_or = nodes[zero].suffix_or | nodes[one].suffix_or | bit_mask;
    uint64_t weight = nodes[zero].weight + nodes[one].weight;
    nodes.push_back(HostSuffixZddNode{uint32_t(bit), zero, one, suffix_or,
                                      weight});
    unique_nodes.emplace(key, result);
    return result;
}

static uint64_t query_host_suffix_zdd(
    const std::vector<HostSuffixZddNode>& nodes, uint32_t root,
    uint32_t suffix, uint64_t& visits) {
    std::array<uint32_t, 64> stack{};
    size_t stack_size = 0;
    stack[stack_size++] = root;
    uint64_t result = 0;
    while (stack_size) {
        const HostSuffixZddNode& node = nodes[stack[--stack_size]];
        visits++;
        if (node.bit == UINT32_MAX || !(node.suffix_or & suffix)) {
            result += node.weight;
            continue;
        }
        if (suffix & (UINT32_C(1) << node.bit)) {
            stack[stack_size++] = node.zero;
        } else {
            if (stack_size + 2 > stack.size()) {
                throw std::runtime_error("weighted ZDD query stack overflow");
            }
            stack[stack_size++] = node.zero;
            stack[stack_size++] = node.one;
        }
    }
    return result;
}
