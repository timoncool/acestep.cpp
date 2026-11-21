#include "adapter-resolve.h"
#include "lm-adapter.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

static void write_weights(const fs::path & path, const std::string & tensor) {
    const std::string header = "{\"" + tensor + "\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}";
    std::ofstream file(path, std::ios::binary);
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

int main() {
    const fs::path temp = fs::temp_directory_path();
    const std::wstring native = temp.native();
    const std::wstring verbatim = native.compare(0, 4, L"\\\\?\\") == 0 ? native : L"\\\\?\\" + native;
    const fs::path unique = "ace-adapter-path-" + std::to_string(GetCurrentProcessId());
    const fs::path fixture = fs::path(verbatim) / unique;
    struct Cleanup {
        const fs::path & path;
        ~Cleanup() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } cleanup{ fixture };

    try {
        const fs::path nested = fs::path(std::string(125, 'a')) / std::string(125, 'b');
        const fs::path root = fixture / nested;
        const fs::path normal = temp / unique / nested;
        const std::string voice = u8"\u0413\u043e\u043b\u043e\u0441";
        const fs::path dir = root / fs::u8path(voice);
        const fs::path weights = dir / fs::u8path(u8"\u97f3\u8272.safetensors");
        fs::create_directories(dir);
        write_weights(weights, "decoder.cross_attn.weight");
        std::ofstream(dir / "adapter_config.json") << R"({"lora_alpha":16,"r":8})";

        const fs::path peft = root / "peft";
        const fs::path lokr = root / "lokr";
        fs::create_directories(peft);
        fs::create_directories(lokr);
        write_weights(peft / "adapter_model.safetensors", "model.layers.0.self_attn.q_proj.lora_A.weight");
        write_weights(peft / "other.safetensors", "decoder.cross_attn.weight");
        write_weights(lokr / "lokr_weights.safetensors", "decoder.cross_attn.weight");
        write_weights(lokr / "other.safetensors", "model.layers.0.self_attn.q_proj.lora_A.weight");

        ModelRegistry registry;
        if (normal.u8string().size() <= MAX_PATH || !registry_scan_adapters(&registry, normal.u8string().c_str())) {
            throw std::runtime_error("could not scan an ordinary Windows adapter path beyond MAX_PATH");
        }
        ModelRegistry verbatim_registry;
        if (!registry_scan_adapters(&verbatim_registry, root.u8string().c_str()) ||
            verbatim_registry.adapters.size() != registry.adapters.size()) {
            throw std::runtime_error("could not scan the equivalent verbatim adapter path");
        }
        const auto * custom = registry_find_adapter(registry, voice.c_str());
        const auto * peft_entry = registry_find_adapter(registry, "peft");
        const auto * lokr_entry = registry_find_adapter(registry, "lokr");
        if (!custom || !peft_entry || !lokr_entry ||
            adapter_single_weights(custom->path.c_str()) != weights.u8string()) {
            throw std::runtime_error("single adapter lookup lost the native separator or Unicode");
        }
        if (adapter_read_alpha(custom->path.c_str()) != 16 || lm_adapter_read_alpha_ratio(custom->path) != 2.0f) {
            throw std::runtime_error("could not read adapter_config.json");
        }
        bool dit = false;
        bool lm = false;
        adapter_classify(custom->path, &dit, &lm);
        if (!dit || lm) {
            throw std::runtime_error("could not classify the adapter through its directory");
        }
        adapter_classify(peft_entry->path, &dit, &lm);
        if (dit || !lm) {
            throw std::runtime_error("could not classify the PEFT adapter");
        }
        adapter_classify(lokr_entry->path, &dit, &lm);
        if (!dit || lm) {
            throw std::runtime_error("could not classify the LoKr adapter");
        }
        puts("Long Unicode Windows adapter path passed");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
