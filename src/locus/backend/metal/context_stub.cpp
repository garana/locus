// The Metal backend's context on builds without Metal (not Apple):
// never usable, and opening it throws.
#include <stdexcept>

#include "locus/backend/metal/context.hpp"

namespace locus::backend::metal {

struct Context::Impl {};

bool Context::usable() { return false; }

Context& Context::get() {
    throw std::runtime_error("metal: not built with Metal");
}

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Context::~Context() = default;

const std::string& Context::device_name() const {
    static const std::string none;
    return none;
}

Context::Buffer Context::alloc(std::size_t) {
    throw std::runtime_error("metal: not built with Metal");
}

Context::Buffer Context::upload(std::span<const std::byte>) {
    throw std::runtime_error("metal: not built with Metal");
}

void Context::free(Buffer& b) { b = Buffer{}; }

bool Context::has_kernel(std::string_view) const { return false; }

void Context::begin_batch() {
    throw std::runtime_error("metal: not built with Metal");
}

bool Context::batch_open() const { return false; }

void Context::end_batch() {
    throw std::runtime_error("metal: not built with Metal");
}

void Context::dispatch(std::string_view, std::span<const Arg>,
                       std::span<const std::byte>, std::uint32_t,
                       std::uint32_t) {
    throw std::runtime_error("metal: not built with Metal");
}

}  // namespace locus::backend::metal
