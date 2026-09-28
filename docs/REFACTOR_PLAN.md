# 重构计划 (Refactoring plan)

Auto-generated. Review and refine by hand.

## Largest files (split candidates)

- `projects/llama.cpp/tools/ui/.svelte-kit/output/client/_app/immutable/bundle.bFnAB0IV.js` — 132157 LOC
- `projects/llama.cpp/vendor/miniaudio/miniaudio.h` — 102704 LOC
- `projects/llama.cpp/vendor/nlohmann/json.hpp` — 23835 LOC
- `projects/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` — 23065 LOC
- `projects/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp` — 20009 LOC
- `projects/llama.cpp/benches/dgx-spark/aime25_openai__gpt-oss-120b-high_temp1.0_20251109_094547.html` — 19380 LOC
- `projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` — 16625 LOC
- `projects/llama.cpp/vendor/cpp-httplib/httplib.cpp` — 13516 LOC
- `projects/llama.cpp/tests/test-backend-ops.cpp` — 10148 LOC
- `projects/llama.cpp/ggml/src/ggml-cpu/ops.cpp` — 9734 LOC
- `projects/llama.cpp/ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp` — 7293 LOC
- `projects/llama.cpp/vendor/stb/stb_image.h` — 7075 LOC
- `projects/llama.cpp/examples/gguf-hash/deps/xxhash/xxhash.h` — 6618 LOC
- `projects/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` — 6389 LOC
- `projects/llama.cpp/tools/ui/.svelte-kit/output/server/entries/pages/(chat)/_layout.svelte.js` — 6302 LOC

## TODO / FIXME / HACK density

- `TODO`: 359
- `FIXME`: 24
- `HACK`: 7
- `XXX`: 0
- `console.log`: 737

## Suggested next steps

- Pick the largest module and extract pure helpers into `lib/`.
- Convert the highest-density TODO cluster into issues.
- Add tests for any module with 0% test/sLOC ratio before further changes.
