#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace locus::backend::metal {

/**
 * The native Metal backend's device state (i#865, stage 1 of #10): the
 * default MTLDevice, one command queue, buffers in shared storage (the
 * CPU and the GPU see the same memory on Apple silicon, so there is no
 * staging copy), and the kernels of kernels.metal, compiled from source
 * when the context opens. Plain C++ here; the Objective-C++ stays in
 * context.mm. Builds without Metal (not Apple) get a stub whose
 * usable() is false and whose open() throws.
 *
 * Not thread-safe: one thread drives a context, as with the Vulkan
 * backend.
 */
class Context {
  public:
    /**
     * A device buffer in shared storage. `data` is its CPU address:
     * writes there are seen by the next dispatch, and a dispatch's
     * writes are seen there once it returns.
     */
    struct Buffer {
        void* handle = nullptr;  ///< the MTLBuffer (retained)
        void* data = nullptr;
        std::size_t bytes = 0;
    };

    /** One kernel argument: a buffer at a byte offset. */
    struct Arg {
        const Buffer* buf = nullptr;
        std::size_t offset = 0;
    };

    /**
     * @returns Whether this build has Metal and the machine a Metal
     *     device. Opens no context.
     */
    static bool usable();

    /**
     * @returns The process's context, opened on first use (device,
     *     queue, kernel library).
     * @throws std::runtime_error without a device, or when the kernels
     *     fail to compile.
     */
    static Context& get();

    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /** @returns The device's name, such as "Apple M4 Max". */
    const std::string& device_name() const;

    /**
     * @param bytes The size (0 gives a 1-byte buffer).
     * @returns A zeroed shared-storage buffer.
     * @throws std::runtime_error when the device cannot allocate it.
     */
    Buffer alloc(std::size_t bytes);

    /**
     * @returns A shared-storage buffer holding a copy of `src` (one
     *     memcpy into the buffer, no staging).
     */
    Buffer upload(std::span<const std::byte> src);

    /** Releases b (no-op on an empty one) and empties it. */
    void free(Buffer& b);

    /** @returns Whether kernel `name` is in the compiled library. */
    bool has_kernel(std::string_view name) const;

    /**
     * Runs kernel `name` over `threads` threads (a 1-D grid, in
     * threadgroups of `group` threads) and waits for it. Buffers bind
     * at indices 0 .. args.size() - 1, then `params` (if any) as a
     * constant block at index args.size().
     * @throws std::invalid_argument for an unknown kernel or a group
     *     size past the kernel's limit; std::runtime_error when the
     *     command buffer fails.
     */
    void dispatch(std::string_view name, std::span<const Arg> args,
                  std::span<const std::byte> params, std::uint32_t threads,
                  std::uint32_t group);

  private:
    struct Impl;
    explicit Context(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace locus::backend::metal
