#pragma once

#include <exception>

namespace locus::server {

/**
 * The HTTP status for an exception a request handler caught. A client
 * error is 400: std::invalid_argument (the request checks throw it) or
 * a JSON exception (a malformed body, or a field of the wrong type).
 * Anything else failed on the server side, such as a full KV cache or
 * a decoder error, and is 500, so a client can tell the two apart.
 *
 * @param e The caught exception.
 * @returns 400 or 500.
 */
int http_status_for(const std::exception& e);

}  // namespace locus::server
