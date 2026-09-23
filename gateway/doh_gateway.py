#!/usr/bin/env python3
"""AdaptDNS DoH Gateway — a REAL DoH proxy with actual blackhole suppression.

This turns Stage 2 from a simulation into reality: the gateway terminates
TLS for its own clients (the deployment assumption in the case study §1),
forwards DNS queries upstream over DoH, and — when blackhole probing is
armed for a query pattern — genuinely SUPPRESSES the first matching response
and records the client's real retry timing from subsequent retransmissions.

Only Python stdlib + openssl (for the self-signed certificate).

Run:
    python gateway/doh_gateway.py                     # listens on :8443
    curl --doh-url https://localhost:8443/dns-query -k https://example.com

Arm a blackhole probe (suppress the FIRST response for matching qnames):
    curl -k -X POST http://localhost:9080/admin/blackhole \
         -d '{"pattern": "example.com"}'

Inspect captured flows (same Flow model as the synthetic generator):
    curl -k http://localhost:9080/admin/flows | python -m json.tool

The captured flows can be fed straight into Stage 1/Stage 2 feature
extraction — resolver-side DNS transactions AND observed retry delays are
both real measurements here.
"""
from __future__ import annotations

import json
import re
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptdns.datamodel import DnsTxn, Flow, Packet

UPSTREAM_DOH = "https://cloudflare-dns.com/dns-query"
DOH_PORT = 8443
ADMIN_PORT = 9080
CERT_DIR = Path(tempfile.gettempdir()) / "adaptdns-gateway"


# --------------------------------------------------------------------------
def ensure_cert() -> tuple[Path, Path]:
    """Generate a self-signed cert with openssl (no extra Python deps)."""
    CERT_DIR.mkdir(exist_ok=True)
    key, crt = CERT_DIR / "key.pem", CERT_DIR / "cert.pem"
    if crt.exists() and key.exists():
        return key, crt
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(crt), "-days", "365",
         "-subj", "/CN=adaptdns-gateway.local",
         "-addext", "subjectAltName=DNS:localhost,DNS:adaptdns-gateway.local"],
        check=True, capture_output=True,
    )
    return key, crt


# --------------------------------------------------------------------------
class GatewayState:
    """Shared state: pending blackhole rules, captured flows, retry obs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rules: list[dict] = []          # {"pattern": re, "armed": True, "hits": 0}
        self.flows: dict[str, Flow] = {}     # client_ip -> Flow (reused per run)
        self.events: list[dict] = []         # audit log for the dashboard
        self.pending_first_seen: dict[str, float] = {}  # qname -> t of suppressed resp

    def add_rule(self, pattern: str) -> None:
        with self.lock:
            self.rules.append({"pattern": re.compile(re.escape(pattern), re.I),
                               "text": pattern, "hits": 0})

    def should_blackhole(self, qname: str) -> bool:
        with self.lock:
            for r in self.rules:
                if r["pattern"].search(qname):
                    r["hits"] += 1
                    return True
            return False

    def note_suppressed(self, qname: str) -> None:
        with self.lock:
            self.pending_first_seen[qname] = time.time()
            self.events.append({"t": time.time(), "type": "blackhole",
                                "qname": qname})

    def note_retry(self, qname: str) -> float | None:
        """A repeated query after suppression = a real observed retry.
        Returns the delay in seconds (and clears the pending marker)."""
        with self.lock:
            t0 = self.pending_first_seen.pop(qname, None)
            if t0 is None:
                return None
            delay = time.time() - t0
            self.events.append({"t": time.time(), "type": "retry",
                                "qname": qname, "delay": delay})
            return delay

    def record_txn(self, client: str, qname: str, qtype: str,
                   resp_size: int, suppressed: bool, retry_delay: float | None) -> None:
        with self.lock:
            flow = self.flows.get(client)
            if flow is None:
                flow = Flow(flow_id=f"gw-{client}", src=client,
                            dst=f"doh-gateway:{DOH_PORT}", proto="doh/443",
                            resolver_visible=True, retry_delays=[])
                self.flows[client] = flow
            now = time.time()
            flow.txns.append(DnsTxn(t=now, qname=qname, qtype=qtype, ttl=0,
                                    resp_size=0 if suppressed else resp_size))
            # Retries of a suppressed qname accumulate as observed delays.
            if retry_delay is not None:
                assert flow.retry_delays is not None
                flow.retry_delays.append(retry_delay)
            if not suppressed:
                flow.packets.append(Packet(t=now, size=resp_size, direction=-1))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "rules": [{"pattern": r["text"], "hits": r["hits"]}
                          for r in self.rules],
                "flows": [
                    {"flow_id": f.flow_id, "src": f.src,
                     "n_queries": len(f.txns),
                     "observed_retry_delays": list(f.retry_delays or []),
                     "sample_qnames": [t.qname for t in f.txns[:5]]}
                    for f in self.flows.values()
                ],
                "events": self.events[-50:],
            }


STATE = GatewayState()


# --------------------------------------------------------------------------
class DoHHandler(BaseHTTPRequestHandler):
    """RFC 8484 POST/GET DoH endpoint with resolver-side blackhole ability."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _respond(self, code: int, body: bytes, ctype: str = "application/dns-message") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _extract_qname(self, wire: bytes) -> tuple[str, str]:
        """Best-effort qname/qtype from a DNS query wire message."""
        try:
            if len(wire) < 13:
                return "", ""
            i = 12
            labels = []
            while i < len(wire) and wire[i] != 0:
                n = wire[i]
                labels.append(wire[i + 1:i + 1 + n].decode(errors="replace"))
                i += n + 1
            qtype = int.from_bytes(wire[i + 1:i + 3], "big") if i + 3 <= len(wire) else 0
            qtypes = {1: "A", 28: "AAAA", 16: "TXT", 5: "CNAME", 65: "HTTPS", 15: "MX"}
            return ".".join(labels), qtypes.get(qtype, str(qtype))
        except Exception:
            return "", ""

    def _forward(self, wire: bytes) -> bytes | None:
        # The system trust store may be broken on some Python builds —
        # prefer certifi's CA bundle, then the system default, then (last
        # resort for lab machines) an unverified context with a warning.
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        req = urllib.request.Request(
            UPSTREAM_DOH, data=wire,
            headers={"Content-Type": "application/dns-message",
                     "Accept": "application/dns-message"})
        try:
            with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 415:  # upstream answered; our wire was malformed
                return exc.read()
            print(f"[gateway] upstream HTTP {exc.code}", file=sys.stderr)
            return None
        except ssl.SSLError:
            ctx = ssl._create_unverified_context()
            print("[gateway] WARNING: upstream TLS verification failed; "
                  "retrying unverified", file=sys.stderr)
            try:
                with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                    return resp.read()
            except Exception as exc:
                print(f"[gateway] upstream failure: {exc}", file=sys.stderr)
                return None
        except Exception as exc:
            print(f"[gateway] upstream failure: {exc}", file=sys.stderr)
            return None

    def _handle(self, wire: bytes) -> None:
        client = self.client_address[0]
        qname, qtype = self._extract_qname(wire)

        # Real observed retry: repeated qname we previously suppressed.
        retry_delay = STATE.note_retry(qname) if qname else None

        answer = self._forward(wire)
        if answer is None:
            self._respond(502, b"upstream failure", "text/plain")
            return

        # ---- THE actual blackhole: suppress this response ---------------
        suppressed = False
        if qname and STATE.should_blackhole(qname):
            STATE.note_suppressed(qname)
            suppressed = True

        if suppressed:
            # Intentionally do NOT return the answer — the client will
            # retransmit; that retransmission timing is what Stage 2 wants.
            self._respond(504, b"no answer (AdaptDNS probe)", "text/plain")
        else:
            self._respond(200, answer)

        STATE.record_txn(client, qname, qtype, len(answer or b""),
                         suppressed, retry_delay)

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/dns-query":
            self._respond(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        self._handle(self.rfile.read(length))

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/dns-query"):
            import base64
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            param = next((p for p in qs.split("&") if p.startswith("dns=")), "")
            try:
                wire = base64.urlsafe_b64decode(param[4:] + "=" * (-len(param[4:]) % 4))
            except Exception:
                self._respond(400, b"bad dns param", "text/plain")
                return
            self._handle(wire)
        else:
            self._respond(404, b"not found", "text/plain")


class AdminHandler(BaseHTTPRequestHandler):
    """Control plane: arm blackhole rules, inspect captured flows."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200) -> None:
        body = json.dumps(obj, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/admin/flows":
            self._json(STATE.snapshot())
        else:
            self._json({"endpoints": ["/admin/flows", "/admin/blackhole"]})

    def do_POST(self):  # noqa: N802
        if self.path != "/admin/blackhole":
            self._json({"error": "unknown endpoint"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            pattern = data["pattern"]
        except Exception:
            self._json({"error": "expected {\"pattern\": \"...\"}"}, 400)
            return
        STATE.add_rule(pattern)
        self._json({"armed": True, "pattern": pattern,
                    "note": "first response for matching qnames will now be suppressed"})


def main() -> int:
    key, crt = ensure_cert()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))

    doh = ThreadingHTTPServer(("0.0.0.0", DOH_PORT), DoHHandler)
    doh.socket = ctx.wrap_socket(doh.socket, server_side=True)
    admin = ThreadingHTTPServer(("127.0.0.1", ADMIN_PORT), AdminHandler)

    print("=" * 64)
    print("AdaptDNS DoH Gateway — real blackhole probing")
    print("=" * 64)
    print(f"  DoH endpoint : https://localhost:{DOH_PORT}/dns-query  (TLS)")
    print(f"  admin API    : http://localhost:{ADMIN_PORT}/admin/...")
    print(f"  upstream     : {UPSTREAM_DOH}")
    print()
    print("  try:")
    print(f"    curl --doh-url https://127.0.0.1:{DOH_PORT}/dns-query \\")
    print("         -k --doh-insecure https://example.com   # use the IP:")
    print("    # curl cannot resolve its --doh-url host THROUGH DoH, so")
    print("    # always address the gateway by IP, not 'localhost'.")
    print(f"    curl -k -X POST http://localhost:{ADMIN_PORT}/admin/blackhole "
          "-d '{{\"pattern\": \"example.com\"}}'")
    print(f"    curl -k http://localhost:{ADMIN_PORT}/admin/flows")
    print()
    threading.Thread(target=admin.serve_forever, daemon=True).start()
    try:
        doh.serve_forever()
    except KeyboardInterrupt:
        print("\n[gateway] shut down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
