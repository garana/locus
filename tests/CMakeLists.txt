# Catch2 v3.7.1, vendored amalgamated build (see docs/DESIGN.md
# section 6 for the dependency policy).
add_library(catch2 STATIC
    ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
)
target_include_directories(catch2 SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/catch2
)

add_executable(locus_tests
    test_sanity.cpp
    test_block_allocator.cpp
    test_features.cpp
    test_arch_registry.cpp
    test_backend_registry.cpp
    test_bench.cpp
    test_engine_loop.cpp
    test_bpe_tokenizer.cpp
    test_chat_template.cpp
    test_cpu_ops.cpp
    test_deepseek.cpp
    test_engine.cpp
    test_engine_metrics.cpp
    test_gguf.cpp
    test_paged_cache.cpp
    test_kv_quant.cpp
    test_token_cache.cpp
    test_helper_message.cpp
    test_helper_connection.cpp
    test_auth_client.cpp
    test_helper_spawn.cpp
    test_prefix_cache.cpp
    test_poller.cpp
    test_meminfo.cpp
    test_mmap_file.cpp
    test_pipeline.cpp
    test_pipeline_stage.cpp
    test_forward_batch_layers.cpp
    test_logit_rows.cpp
    test_outw_batch.cpp
    test_batch_embd.cpp
    test_pipeline_mux.cpp
    test_stage_executor.cpp
    test_stage_batch.cpp
    test_stage_cuda.cpp
    test_resolver.cpp
    test_stage_config.cpp
    test_pipeline_driver.cpp
    test_grammar.cpp
    test_http.cpp
    test_http_server.cpp
    test_sampling.cpp
    test_speculative.cpp
    test_responses.cpp
    test_server.cpp
    test_server_errors.cpp
    test_tools.cpp
    test_vulkan.cpp
    test_vulkan_attn.cpp
    test_vulkan_moe.cpp
    test_vulkan_rope.cpp
    test_vulkan_conv1d.cpp
    test_vulkan_fused_matvec.cpp
    test_vulkan_small_batch.cpp
    test_vulkan_prefill.cpp
    test_vulkan_mem_fit.cpp
    test_vulkan_limits.cpp
    test_vulkan_oob.cpp
    test_q8_layout.cpp
    test_vulkan_reset.cpp
    test_tokenizer.cpp
    test_t5_tokenizer.cpp
    test_t5_encoder.cpp
    test_pixart_dit.cpp
    test_pixart_sampler.cpp
    test_pixart_vae.cpp
    test_pixart_pipeline.cpp
    test_ltxv_dit.cpp
    test_model_e2e.cpp
    test_moe.cpp
    test_dsa.cpp
    test_muse_spark.cpp
    test_codec.cpp
    test_snac.cpp
    test_vq16.cpp
    test_janus.cpp
    test_janus_vision.cpp
    test_png.cpp
    test_image_input.cpp
    test_png_decode.cpp
    test_jpeg_decode.cpp
    test_jpeg_speed.cpp
    test_base64.cpp
    test_deepseek41.cpp
    test_threads.cpp
    test_perf_counters.cpp
    test_cli_args.cpp
    test_vulkan_pool.cpp
)
target_link_libraries(locus_tests PRIVATE locus catch2)
target_include_directories(locus_tests SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/httplib
    ${CMAKE_SOURCE_DIR}/third_party/json
)
target_include_directories(locus_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tools
)
target_compile_definitions(locus_tests PRIVATE
    LOCUS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    LOCUS_AUTH_HELPER_BIN="$<TARGET_FILE:locus-auth-helper>"
    LOCUS_STAGE_BIN="$<TARGET_FILE:locus-stage>"
    LOCUS_RUN_BIN="$<TARGET_FILE:locus-run>"
    LOCUS_SERVER_BIN="$<TARGET_FILE:locus-server>"
)
add_dependencies(locus_tests locus-auth-helper locus-stage locus-run
                 locus-server)

add_test(NAME unit COMMAND locus_tests)

# On-device out-of-bounds proof for the Q8_0 upload (F2). Runs the
# hidden [validation] cases with the Khronos validation layer enabled
# (GPU-assisted). Needs vulkan-validationlayers installed and a device;
# the cases skip when LOCUS_VK_VALIDATION is unset, so they never run in
# the plain "unit" test.
add_test(NAME vulkan_oob_validation
         COMMAND locus_tests "[validation]")
set_tests_properties(vulkan_oob_validation PROPERTIES
    ENVIRONMENT "LOCUS_VK_VALIDATION=1"
    # Catch2 exits 4 when every selected case skips (no validation
    # layer or no device); SKIP_RETURN_CODE makes ctest read that as
    # skipped, not failed, so a layerless host (CI or a peer without
    # the Khronos layer) does not report a spurious failure.
    SKIP_RETURN_CODE 4)

# i#103: force the smaller-tile (BK=16) matmul variant and run the
# Vulkan parity cases against the CPU. On a tight device (e.g. V3DV,
# 16384) the backend already picks this variant, so this target lets a
# roomy device (e.g. MoltenVK, 32768) exercise it too. The filter drops
# the slow model e2e (~[e2e]) and the hidden throughput benches
# (~[.]); selecting a tag force-includes hidden cases carrying it, and
# the benches allocate buffers that OOM a 4 GB device. Cases skip
# without a device; SKIP_RETURN_CODE reads that as skipped, not failed.
add_test(NAME vulkan_matmul_small
         COMMAND locus_tests "[vulkan]~[e2e]~[.]")
set_tests_properties(vulkan_matmul_small PROPERTIES
    ENVIRONMENT "LOCUS_VK_MATMUL_SMALL=1"
    SKIP_RETURN_CODE 4)
