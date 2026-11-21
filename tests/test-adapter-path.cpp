#include "adapter-resolve.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

int main() {
    const fs::path temp = fs::temp_directory_path();
    const std::wstring native = temp.native();
    const std::wstring verbatim = native.compare(0, 4, L"\\\\?\\") == 0 ? native : L"\\\\?\\" + native;
    const fs::path fixture = fs::path(verbatim) / ("ace-adapter-path-" + std::to_string(GetCurrentProcessId()));
    struct Cleanup {
        const fs::path & path;
        ~Cleanup() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } cleanup{ fixture };

    try {
        const fs::path dir = fixture / std::string(125, 'a') / std::string(125, 'b') / fs::u8path(u8"\u0413\u043e\u043b\u043e\u0441");
        const fs::path weights = dir / fs::u8path(u8"\u97f3\u8272.safetensors");
        fs::create_directories(dir);
        const std::string header = R"({"decoder.cross_attn.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}})";
        {
            std::ofstream file(weights, std::ios::binary);
            for (unsigned shift = 0; shift < 64; shift += 8) {
                file.put(static_cast<char>(static_cast<uint64_t>(header.size()) >> shift));
            }
            file << header;
            file.write("\0\0\0\0", 4);
            file.close();
            if (!file) {
                throw std::runtime_error("could not write the adapter weights fixture");
            }
        }
        std::ofstream(dir / "adapter_config.json") << R"({"lora_alpha":16})";

        const std::string path = dir.u8string();
        if (path.size() <= MAX_PATH || adapter_single_weights(path.c_str()) != weights.u8string()) {
            throw std::runtime_error("single adapter lookup lost the native separator or Unicode");
        }
        if (adapter_read_alpha(path.c_str()) != 16) {
            throw std::runtime_error("could not read adapter_config.json");
        }
        bool dit = false;
        bool lm = false;
        adapter_classify(path, &dit, &lm);
        if (!dit || lm) {
            throw std::runtime_error("could not classify the adapter through its directory");
        }
        puts("Long Unicode Windows adapter path passed");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
