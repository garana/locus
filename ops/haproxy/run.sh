#!/bin/bash
# Runs HAProxy (natively, in the foreground) as an HTTPS front for
# locus-server, with ops/haproxy/locus.cfg. Start locus-server first, on
# loopback:
#
#   locus-server model.gguf 8080 &        # on 127.0.0.1, its default
#   LOCUS_TLS_CRT=server.crt LOCUS_TLS_KEY=server.key \
#       LOCUS_TLS_BIND=192.168.1.10:8443 ops/haproxy/run.sh
#
# Variables (defaults in brackets):
#   LOCUS_TLS_CRT   the server certificate, PEM (required)
#   LOCUS_TLS_KEY   its private key, PEM (required)
#   LOCUS_TLS_BIND  "addr:port[,addr:port...]" [127.0.0.1:8443]
#   LOCUS_UPSTREAM  locus-server [127.0.0.1:8080]
#   HAPROXY         the haproxy binary [haproxy on PATH]
# Clients must trust the CA that signed the certificate (curl --cacert).
set -euo pipefail

: "${LOCUS_TLS_CRT:?set LOCUS_TLS_CRT to the server certificate}"
: "${LOCUS_TLS_KEY:?set LOCUS_TLS_KEY to its private key}"
export LOCUS_TLS_CRT LOCUS_TLS_KEY
export LOCUS_TLS_BIND=${LOCUS_TLS_BIND:-127.0.0.1:8443}
export LOCUS_UPSTREAM=${LOCUS_UPSTREAM:-127.0.0.1:8080}
HAPROXY=${HAPROXY:-haproxy}

cfg="$(cd "$(dirname "$0")" && pwd)/locus.cfg"
"$HAPROXY" -c -q -f "$cfg"
exec "$HAPROXY" -db -f "$cfg"
