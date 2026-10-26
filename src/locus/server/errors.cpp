#include "locus/server/errors.hpp"

#include <stdexcept>

#include "json.hpp"

namespace locus::server {

int http_status_for(const std::exception& e) {
    const bool client =
        dynamic_cast<const std::invalid_argument*>(&e) != nullptr ||
        dynamic_cast<const nlohmann::json::exception*>(&e) != nullptr;
    return client ? 400 : 500;
}

}  // namespace locus::server
