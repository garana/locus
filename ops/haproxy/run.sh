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
#   LOCUS_TLS_CLIENT_CA   a CA certificate, PEM: when set, every client
#                   must present a certificate it signed (mutual TLS);
#                   unset, no client certificate is asked for [unset]
#   LOCUS_TLS_CLIENT_CRL  that CA's revocation list, PEM [unset]. Renew
#                   it before its nextUpdate: an expired list makes
#                   OpenSSL refuse every client, the valid ones too.
#                   HAProxy reads it at start, so restart after a new one.
# Clients must trust the CA that signed the certificate (curl --cacert).
# Behind the proxy every client reaches locus-server from 127.0.0.1, so
# leave its --max-conns-per-ip at 0 (off, the default): a cap there
# would count all clients together.
#
# HAProxy 3.0 and later read the key in place. Older ones (Debian 12's
# 2.6, Ubuntu 24.04's 2.8) need the certificate and key in one file:
# for them this script writes it to a private temp dir (mode 0700, file
# 0600) and deletes it when HAProxy exits. LOCUS_TLS_FORCE_PEM=1 takes
# that path on any version (to test it).
set -euo pipefail

: "${LOCUS_TLS_CRT:?set LOCUS_TLS_CRT to the server certificate}"
: "${LOCUS_TLS_KEY:?set LOCUS_TLS_KEY to its private key}"
export LOCUS_TLS_CRT LOCUS_TLS_KEY
export LOCUS_TLS_BIND=${LOCUS_TLS_BIND:-127.0.0.1:8443}
export LOCUS_UPSTREAM=${LOCUS_UPSTREAM:-127.0.0.1:8080}
# Client certificates: HAProxy needs only the CA's certificate (never
# its key, nor each client's certificate), and accepts any certificate
# that CA signed. The options reach the bind line as separate words, so
# these paths may not contain whitespace.
# locus speaks HTTP/1.1.
LOCUS_TLS_OPTS="alpn http/1.1"
if [ -n "${LOCUS_TLS_CLIENT_CA:-}" ]; then
    LOCUS_TLS_OPTS+=" ca-file $LOCUS_TLS_CLIENT_CA verify required"
    if [ -n "${LOCUS_TLS_CLIENT_CRL:-}" ]; then
        LOCUS_TLS_OPTS+=" crl-file $LOCUS_TLS_CLIENT_CRL"
    fi
elif [ -n "${LOCUS_TLS_CLIENT_CRL:-}" ]; then
    echo "LOCUS_TLS_CLIENT_CRL needs LOCUS_TLS_CLIENT_CA" >&2
    exit 2
fi
case "${LOCUS_TLS_CLIENT_CA:-}${LOCUS_TLS_CLIENT_CRL:-}" in
    *[[:space:]]*)
        echo "client CA / CRL paths may not contain whitespace" >&2
        exit 2 ;;
esac
export LOCUS_TLS_OPTS
HAPROXY=${HAPROXY:-haproxy}
cfg="$(cd "$(dirname "$0")" && pwd)/locus.cfg"

# "HAProxy version 2.8.5-..." -> 2
major=$("$HAPROXY" -v | sed -n 's/^HA-*Proxy version \([0-9]*\)\..*/\1/p')
if [ "${LOCUS_TLS_FORCE_PEM:-0}" = 1 ] || [ "${major:-0}" -lt 3 ]; then
    dir=$(mktemp -d)
    trap 'rm -rf "$dir"' EXIT
    (umask 077 && cat "$LOCUS_TLS_CRT" "$LOCUS_TLS_KEY" > "$dir/server.pem")
    export LOCUS_TLS_PEM="$dir/server.pem"
    "$HAPROXY" -c -q -f "$cfg"
    # Not exec: the trap removes the file when HAProxy exits.
    "$HAPROXY" -db -f "$cfg" &
    pid=$!
    trap 'kill "$pid" 2>/dev/null' INT TERM
    rc=0
    wait "$pid" || rc=$?
    # A signal ends the first wait early: let HAProxy finish stopping.
    wait "$pid" 2>/dev/null || true
    exit "$rc"
fi
"$HAPROXY" -c -q -f "$cfg"
exec "$HAPROXY" -db -f "$cfg"
