// The Metal backend's device state (i#865); see context.hpp.
#include "locus/backend/metal/context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

// kKernelSource: kernels.metal, embedded by CMake.
#include "locus/backend/metal/kernels_src.hpp"

namespace locus::backend::metal {

namespace {

std::string ns_error(NSError* err) {
    return err != nil ? std::string([[err localizedDescription] UTF8String])
                      : std::string("unknown error");
}

}  // namespace

struct Context::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::string name;
    std::unordered_map<std::string, id<MTLComputePipelineState>> kernels;
};

bool Context::usable() {
    static const bool ok = [] {
        @autoreleasepool {
            id<MTLDevice> d = MTLCreateSystemDefaultDevice();
            return d != nil;
        }
    }();
    return ok;
}

Context& Context::get() {
    static Context* ctx = [] {
        @autoreleasepool {
            auto impl = std::make_unique<Impl>();
            impl->device = MTLCreateSystemDefaultDevice();
            if (impl->device == nil) {
                throw std::runtime_error("metal: no Metal device");
            }
            impl->name = [[impl->device name] UTF8String];
            impl->queue = [impl->device newCommandQueue];
            if (impl->queue == nil) {
                throw std::runtime_error("metal: no command queue");
            }
            MTLCompileOptions* opt = [MTLCompileOptions new];
            opt.mathMode = MTLMathModeSafe;  // no fast-math reassociation
            NSError* err = nil;
            id<MTLLibrary> lib = [impl->device
                newLibraryWithSource:[NSString
                                         stringWithUTF8String:kKernelSource]
                             options:opt
                               error:&err];
            if (lib == nil) {
                throw std::runtime_error("metal: kernels: " + ns_error(err));
            }
            for (NSString* fname in [lib functionNames]) {
                id<MTLFunction> fn = [lib newFunctionWithName:fname];
                id<MTLComputePipelineState> pso = [impl->device
                    newComputePipelineStateWithFunction:fn
                                                  error:&err];
                if (pso == nil) {
                    throw std::runtime_error(
                        "metal: pipeline " + std::string([fname UTF8String]) +
                        ": " + ns_error(err));
                }
                impl->kernels.emplace([fname UTF8String], pso);
            }
            // Leaked on purpose: the context lives for the whole
            // process, and no static destructor runs after Metal's own
            // teardown.
            return new Context(std::move(impl));
        }
    }();
    return *ctx;
}

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Context::~Context() = default;

const std::string& Context::device_name() const { return impl_->name; }

Context::Buffer Context::alloc(std::size_t bytes) {
    @autoreleasepool {
        const std::size_t n = bytes == 0 ? 1 : bytes;
        id<MTLBuffer> b =
            [impl_->device newBufferWithLength:n
                                       options:MTLResourceStorageModeShared];
        if (b == nil) {
            throw std::runtime_error("metal: cannot allocate " +
                                     std::to_string(n) + " bytes");
        }
        // Owned by the Buffer until free(): +1 retained across ARC.
        Buffer out;
        out.handle = (__bridge_retained void*)b;
        out.data = [b contents];
        out.bytes = bytes;
        std::memset(out.data, 0, n);
        return out;
    }
}

Context::Buffer Context::upload(std::span<const std::byte> src) {
    Buffer b = alloc(src.size());
    if (!src.empty()) {
        std::memcpy(b.data, src.data(), src.size());
    }
    return b;
}

void Context::free(Buffer& b) {
    if (b.handle != nullptr) {
        @autoreleasepool {
            id<MTLBuffer> mb = (__bridge_transfer id<MTLBuffer>)b.handle;
            (void)mb;  // released at the end of this scope
        }
    }
    b = Buffer{};
}

bool Context::has_kernel(std::string_view name) const {
    return impl_->kernels.count(std::string(name)) != 0;
}

void Context::dispatch(std::string_view name, std::span<const Arg> args,
                       std::span<const std::byte> params,
                       std::uint32_t threads, std::uint32_t group) {
    const auto it = impl_->kernels.find(std::string(name));
    if (it == impl_->kernels.end()) {
        throw std::invalid_argument("metal: no kernel " + std::string(name));
    }
    id<MTLComputePipelineState> pso = it->second;
    if (group == 0 || group > [pso maxTotalThreadsPerThreadgroup]) {
        throw std::invalid_argument(
            "metal: group of " + std::to_string(group) + " for " +
            std::string(name) + " (max " +
            std::to_string([pso maxTotalThreadsPerThreadgroup]) + ")");
    }
    if (threads == 0) {
        return;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        NSUInteger i = 0;
        for (const Arg& a : args) {
            [enc setBuffer:(__bridge id<MTLBuffer>)a.buf->handle
                    offset:a.offset
                   atIndex:i++];
        }
        if (!params.empty()) {
            [enc setBytes:params.data() length:params.size() atIndex:i];
        }
        [enc dispatchThreadgroups:MTLSizeMake((threads + group - 1) / group,
                                              1, 1)
            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("metal: " + std::string(name) + ": " +
                                     ns_error([cb error]));
        }
    }
}

}  // namespace locus::backend::metal
