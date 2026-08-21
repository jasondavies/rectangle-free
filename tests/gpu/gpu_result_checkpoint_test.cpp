#include "gpu_result_checkpoint.hpp"

#include <iostream>

namespace fs = std::filesystem;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
static void require_failure(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    if (argc != 1) return 2;
    fs::path root = fs::temp_directory_path() /
                    ("rectangle-free-checkpoint-" +
                     std::to_string(uint64_t(getpid())));
    fs::create_directories(root);
    try {
        fs::path canonical = root / "canonical.orbits";
        fs::path corpus = root / "owner.orbits";
        fs::path manifest = root / "work.tsv";
        fs::path results = root / "results";
        fs::create_directories(results);
        {
            std::ofstream output(canonical);
            output << "canonical-content\n";
        }
        {
            std::ofstream output(corpus);
            output << "orbit-content\n";
        }
        {
            std::ofstream output(manifest);
            output << "# exact range and filter intersection\n"
                   << "s0001 " << corpus.string() << " 10 20 8 3\n";
        }

        std::vector<gpu_checkpoint::WorkItem> items =
            gpu_checkpoint::read_work_manifest(manifest.string());
        require(items.size() == 1 && items[0].id == "s0001" &&
                    items[0].start == 10 && items[0].end == 20 &&
                    items[0].filter_mod == 8 && items[0].filter_id == 3,
                "manifest identity changed");
        gpu_checkpoint::RunProvenance run = gpu_checkpoint::run_provenance(
            "RECT_TEST_RESULT", "8x8", argv[0], "configuration=v1",
            canonical.string());
        gpu_checkpoint::WorkProvenance provenance =
            gpu_checkpoint::work_provenance(run, items[0]);
        const std::vector<std::string> required = {
            "records", "contribution", "total_seconds"};

        {
            gpu_checkpoint::WorkClaim first(
                gpu_checkpoint::result_path(results, items[0]));
            require_failure(
                [&] {
                    gpu_checkpoint::WorkClaim second(
                        gpu_checkpoint::result_path(results, items[0]));
                },
                "duplicate work claim was accepted");
            gpu_checkpoint::write_result(
                results, items[0], provenance,
                "records 1\ncontribution 42\ntotal_seconds 0.5\n");
        }
        require(gpu_checkpoint::validated_result_exists(
                    results, items[0], provenance, required),
                "fresh result did not validate");
        require_failure(
            [&] {
                gpu_checkpoint::write_result(
                    results, items[0], provenance,
                    "records 1\ncontribution 42\ntotal_seconds 0.5\n");
            },
            "immutable result publication was overwritten");

        gpu_checkpoint::RunProvenance changed_run =
            gpu_checkpoint::run_provenance(
                "RECT_TEST_RESULT", "8x8", argv[0], "configuration=v2",
                canonical.string());
        gpu_checkpoint::WorkProvenance changed =
            gpu_checkpoint::work_provenance(changed_run, items[0]);
        require_failure(
            [&] {
                gpu_checkpoint::validated_result_exists(
                    results, items[0], changed, required);
            },
            "configuration mismatch was accepted");

        fs::path result = gpu_checkpoint::result_path(results, items[0]);
        {
            std::ofstream output(result, std::ios::app);
            output << "tampered 1\n";
        }
        require_failure(
            [&] {
                gpu_checkpoint::validated_result_exists(
                    results, items[0], provenance, required);
            },
            "checksum corruption was accepted");

        fs::remove_all(root);
        std::cout << "CHECKPOINT_TEST exact=OK\n";
        return 0;
    } catch (...) {
        fs::remove_all(root);
        throw;
    }
}
