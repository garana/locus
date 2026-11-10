#include "router_cli.hpp"

/**
 * locus-router (#388): one URL in front of locus instances, proxying
 * each request to an instance that serves its kind.
 *   locus-router [--listen HOST:PORT] INSTANCE...
 */
int main(int argc, char** argv) {
    return locus_tools::run_router(argc, argv, 1);
}
