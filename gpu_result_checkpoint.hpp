#pragma once

// Geometry-neutral, self-identifying checkpoint support for exact GPU solves.
// Result format version 3 is line-oriented so campaign tooling can inspect it,
// but every payload is checksummed and bound to the executable, configuration,
// canonical source, orbit corpus, and exact range/filter identity.

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "sha256.hpp"

namespace gpu_checkpoint {

namespace fs = std::filesystem;

struct WorkItem {
    std::string id;
    std::string path;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t filter_mod = 0;
    uint64_t filter_id = 0;
};

struct RunProvenance {
    std::string result_magic;
    std::string geometry;
    std::string solver_binary_sha256;
    std::string solver_configuration_sha256;
    std::string canonical_cache_sha256;
};

struct WorkProvenance {
    RunProvenance run;
    std::string orbit_corpus_sha256;
};

inline bool valid_work_id(const std::string& id) {
    if (id.empty()) return false;
    for (unsigned char character : id) {
        if (!std::isalnum(character) && character != '-' && character != '_' &&
            character != '.') {
            return false;
        }
    }
    return true;
}

inline std::vector<WorkItem> read_work_manifest(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open work manifest " + path);
    std::vector<WorkItem> items;
    std::unordered_set<std::string> ids;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        line_number++;
        size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream fields(line);
        WorkItem item;
        if (!(fields >> item.id >> item.path >> item.start >> item.end)) {
            throw std::runtime_error("invalid manifest line " +
                                     std::to_string(line_number));
        }
        if (fields >> item.filter_mod) {
            if (!(fields >> item.filter_id) || !item.filter_mod ||
                item.filter_id >= item.filter_mod) {
                throw std::runtime_error("invalid manifest filter on line " +
                                         std::to_string(line_number));
            }
        }
        std::string trailing;
        if (fields >> trailing || !valid_work_id(item.id) ||
            !ids.insert(item.id).second) {
            throw std::runtime_error("invalid or duplicate work id on line " +
                                     std::to_string(line_number));
        }
        items.push_back(std::move(item));
    }
    if (items.empty()) throw std::runtime_error("work manifest is empty");
    return items;
}

inline fs::path result_path(const fs::path& directory, const WorkItem& item) {
    return directory / (item.id + ".result");
}

class WorkClaim {
  public:
    explicit WorkClaim(const fs::path& result) {
        path_ = result;
        path_ += ".lock";
        descriptor_ = open(path_.c_str(), O_CREAT | O_RDWR, 0666);
        if (descriptor_ < 0 || flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
            if (descriptor_ >= 0) close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("work item is already claimed: " +
                                     result.string());
        }
    }
    WorkClaim(const WorkClaim&) = delete;
    WorkClaim& operator=(const WorkClaim&) = delete;
    WorkClaim(WorkClaim&& other) noexcept
        : descriptor_(std::exchange(other.descriptor_, -1)),
          path_(std::move(other.path_)) {}
    ~WorkClaim() {
        if (descriptor_ >= 0) {
            flock(descriptor_, LOCK_UN);
            close(descriptor_);
        }
    }

  private:
    int descriptor_ = -1;
    fs::path path_;
};

inline RunProvenance run_provenance(
    const std::string& result_magic, const std::string& geometry,
    const std::string& executable, const std::string& configuration,
    const std::string& canonical_cache) {
    return RunProvenance{result_magic, geometry, sha256_file(executable),
                         sha256_string(configuration),
                         sha256_file(canonical_cache)};
}

inline WorkProvenance work_provenance(const RunProvenance& run,
                                      const WorkItem& item) {
    // A manifest commonly contains many ranges of one immutable corpus.
    static std::unordered_map<std::string, std::string> corpus_digests;
    auto found = corpus_digests.find(item.path);
    if (found == corpus_digests.end()) {
        found = corpus_digests.emplace(item.path, sha256_file(item.path)).first;
    }
    return WorkProvenance{run, found->second};
}

inline bool validated_result_exists(
    const fs::path& directory, const WorkItem& item,
    const WorkProvenance& provenance,
    const std::vector<std::string>& required_result_fields) {
    fs::path path = result_path(directory, item);
    if (!fs::exists(path)) return false;
    std::ifstream input(path);
    std::string header;
    if (!std::getline(input, header) ||
        header != provenance.run.result_magic + " 3") {
        throw std::runtime_error("invalid existing result header: " +
                                 path.string());
    }
    std::unordered_map<std::string, std::string> fields;
    Sha256 payload_hash;
    std::string expected_payload_hash;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream values(line);
        std::string key;
        std::string value;
        std::string trailing;
        if (!(values >> key >> value) || values >> trailing) {
            throw std::runtime_error("invalid existing result field: " +
                                     path.string());
        }
        if (key == "result_payload_sha256") {
            if (!expected_payload_hash.empty()) {
                throw std::runtime_error("duplicate result checksum: " +
                                         path.string());
            }
            expected_payload_hash = value;
            continue;
        }
        payload_hash.update(line);
        payload_hash.update("\n");
        if (!fields.emplace(key, value).second) {
            throw std::runtime_error("duplicate existing result field: " +
                                     path.string());
        }
    }
    if (expected_payload_hash.empty() ||
        expected_payload_hash != payload_hash.finish_hex()) {
        throw std::runtime_error("existing result checksum mismatch: " +
                                 path.string());
    }
    auto require = [&](const std::string& name) -> const std::string& {
        auto found = fields.find(name);
        if (found == fields.end()) {
            throw std::runtime_error("missing existing result field: " +
                                     path.string());
        }
        return found->second;
    };
    bool identity_matches =
        require("id") == item.id && require("path") == item.path &&
        std::stoull(require("start")) == item.start &&
        std::stoull(require("end")) == item.end &&
        std::stoull(require("filter_mod")) == item.filter_mod &&
        std::stoull(require("filter_id")) == item.filter_id;
    bool provenance_matches =
        require("geometry") == provenance.run.geometry &&
        require("token_plane_quotient") == "1" &&
        require("solver_binary_sha256") ==
            provenance.run.solver_binary_sha256 &&
        require("solver_configuration_sha256") ==
            provenance.run.solver_configuration_sha256 &&
        require("canonical_cache_sha256") ==
            provenance.run.canonical_cache_sha256 &&
        require("orbit_corpus_sha256") == provenance.orbit_corpus_sha256;
    for (const std::string& field : required_result_fields) require(field);
    if (!identity_matches || !provenance_matches) {
        throw std::runtime_error(
            "existing result identity/provenance mismatch: " + path.string());
    }
    return true;
}

inline void write_result(const fs::path& directory, const WorkItem& item,
                         const WorkProvenance& provenance,
                         const std::string& result_fields) {
    fs::path final_path = result_path(directory, item);
    fs::path temporary_path = final_path;
    temporary_path += ".tmp." + std::to_string(getpid());
    std::ostringstream payload;
    payload << "id " << item.id << "\n"
            << "path " << item.path << "\n"
            << "start " << item.start << "\n"
            << "end " << item.end << "\n"
            << "filter_mod " << item.filter_mod << "\n"
            << "filter_id " << item.filter_id << "\n"
            << "geometry " << provenance.run.geometry << "\n"
            << "token_plane_quotient 1\n"
            << "solver_binary_sha256 "
            << provenance.run.solver_binary_sha256 << "\n"
            << "solver_configuration_sha256 "
            << provenance.run.solver_configuration_sha256 << "\n"
            << "canonical_cache_sha256 "
            << provenance.run.canonical_cache_sha256 << "\n"
            << "orbit_corpus_sha256 " << provenance.orbit_corpus_sha256
            << "\n"
            << result_fields;
    std::string payload_text = payload.str();
    if (payload_text.empty() || payload_text.back() != '\n') {
        payload_text.push_back('\n');
    }
    std::ofstream output(temporary_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create " + temporary_path.string());
    }
    output << provenance.run.result_magic << " 3\n"
           << payload_text << "result_payload_sha256 "
           << sha256_string(payload_text) << "\n";
    output.close();
    if (!output) {
        throw std::runtime_error("failed writing " + temporary_path.string());
    }
    std::error_code publication_error;
    fs::create_hard_link(temporary_path, final_path, publication_error);
    fs::remove(temporary_path);
    if (publication_error) {
        throw std::runtime_error("result publication refused for " +
                                 final_path.string() + ": " +
                                 publication_error.message());
    }
}

}  // namespace gpu_checkpoint
