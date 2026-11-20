import os
import socket
import json
import base64
import time
import ipaddress
from collections import deque

from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP

from detector import AnomalyDetector
from database import DatabaseManager
from utils import log_message, log_success, log_warning, log_error


def _infer_service(dst_port: int, proto: str) -> str:
    """
    Very small heuristic to approximate UNSW-NB15 'service' field.
    This is NOT perfect, but gives stable categories for common ports.
    """
    if proto.upper() not in ("TCP", "UDP"):
        return "other"

    port_map = {
        80: "http",
        443: "http",
        21: "ftp",
        20: "ftp-data",
        22: "ssh",
        25: "smtp",
        53: "dns",
        110: "pop3",
        143: "imap",
        123: "ntp",
        161: "snmp",
        389: "ldap",
        445: "smb",
        3306: "mysql",
        3389: "rdp",
        8080: "http",
    }
    return port_map.get(int(dst_port or 0), "other")


def _infer_state(proto: str, pkt) -> str:
    """
    Approximate UNSW-NB15 'state'.
    If you have real TCP flags/handshake tracking later, swap this out.
    """
    p = (proto or "").upper()
    if p == "TCP":
        try:
            if pkt and pkt.haslayer(TCP):
                flags = int(pkt[TCP].flags)
                if flags == 0x02:
                    return "SYN"
                if flags == 0x12:
                    return "SYNACK"
                if flags & 0x04:
                    return "RST"
                if flags & 0x01:
                    return "FIN"
        except Exception:
            pass
        return "CON"
    if p == "UDP":
        return "INT"
    if p == "ICMP":
        return "ECO"
    return "UNK"


class FlowWindow:
    """
    Small sliding window counters to approximate some ct_* features.
    This is an approximation (not an exact UNSW-NB15 generator),
    but it's consistent and avoids "missing feature" chaos.
    """

    def __init__(self, maxlen=None):
        self.maxlen = int(maxlen or os.getenv("FLOW_WINDOW_SIZE", "2000"))
        self.recent_src = deque(maxlen=maxlen)
        self.recent_dst = deque(maxlen=maxlen)
        self.recent_srv = deque(maxlen=maxlen)
        self.recent_state = deque(maxlen=maxlen)
        self.recent_dst_sport = deque(maxlen=maxlen)
        self.recent_src_dport = deque(maxlen=maxlen)
        self.recent_dst_src = deque(maxlen=maxlen)

    def push(self, src, dst, srv, state, src_port, dst_port):
        self.recent_src.append(src)
        self.recent_dst.append(dst)
        self.recent_srv.append(srv)
        self.recent_state.append(state)
        self.recent_dst_sport.append((dst, src_port))
        self.recent_src_dport.append((src, dst_port))
        self.recent_dst_src.append((dst, src))

    def ct_src_ltm(self, src):
        return sum(1 for x in self.recent_src if x == src)

    def ct_dst_ltm(self, dst):
        return sum(1 for x in self.recent_dst if x == dst)

    def ct_srv_dst(self, srv):
        return sum(1 for x in self.recent_srv if x == srv)

    def ct_state_ttl(self, state):
        return sum(1 for x in self.recent_state if x == state)

    def ct_dst_sport_ltm(self, dst, src_port):
        return sum(1 for x in self.recent_dst_sport if x == (dst, src_port))

    def ct_src_dport_ltm(self, src, dst_port):
        return sum(1 for x in self.recent_src_dport if x == (src, dst_port))

    def ct_dst_src_ltm(self, dst, src):
        return sum(1 for x in self.recent_dst_src if x == (dst, src))


def _norm_ip(ip_str: str) -> str:
    """Normalize IP address string for consistent comparison (IPv4 and IPv6)."""
    try:
        return str(ipaddress.ip_address(ip_str))
    except ValueError:
        return ip_str


class NIDSReceiver:
    def __init__(self, host="127.0.0.1", port=8080, alert_threshold: float = 0.66):
        self.host = host
        self.port = port

        # FIX 1: Configurable alert threshold (was hardcoded to 1.0)
        # 0.66 means at least 2 out of 3 models must agree to fire an alert
        self.alert_threshold = alert_threshold

        self.db = DatabaseManager()
        self.detector = AnomalyDetector()
        self.flow_window = FlowWindow(maxlen=2000)

        self.packet_count = 0
        self.last_log_time = time.time()

        self.dropped_count = 0
        self.dropped_reasons = {}
        self.alert_dedup = {}
        self.alert_count = 0

    def _decode_packet_bytes(self, packet_b64: str):
        try:
            return base64.b64decode(packet_b64)
        except Exception as e:
            log_error(f"Failed to decode base64 payload: {e}")
            return None

    def _validate_schema(self, msg: dict) -> tuple:
        """
        Validate required fields EXIST (not that they're truthy).
        Returns (is_valid, reason_if_invalid)
        """
        core_required = ["src_ip", "dst_ip", "protocol", "src_port", "dst_port"]
        missing = [k for k in core_required if k not in msg or msg[k] is None]

        if missing:
            return False, f"missing_fields:{','.join(missing)}"

        payload_keys = ["raw_data", "data", "packet", "payload"]
        has_payload = any(k in msg and msg[k] is not None for k in payload_keys)

        if not has_payload:
            return False, "missing_payload:no_raw_data_or_fallbacks"

        if msg.get("src_ip") == "0.0.0.0" or msg.get("dst_ip") == "0.0.0.0":
            return False, "invalid_ip:0.0.0.0"

        try:
            int(msg.get("src_port", 0))
            int(msg.get("dst_port", 0))
        except (ValueError, TypeError):
            return False, "invalid_port_type"

        return True, ""

    def _check_feature_completeness(self, feats: dict) -> tuple:
        """
        FIX 2: Validate features against the ACTUAL training contract from feature_meta.json.
        Instead of silently filling zeros deep in the detector, we catch gaps here at the
        boundary, log them visibly, and fill defaults explicitly so you can see what's drifting.
        """
        if not self.detector.meta:
            # No meta loaded — can't validate, pass through with warning
            log_warning("Cannot validate feature completeness: feature_meta.json not loaded")
            return True, ""

        feature_cols = self.detector.meta.get("feature_cols", [])
        cat_cols = set(self.detector.meta.get("cat_cols", []))

        missing_numeric = []
        missing_cat = []

        for col in feature_cols:
            if col not in feats:
                if col in cat_cols:
                    missing_cat.append(col)
                else:
                    missing_numeric.append(col)

        if missing_numeric or missing_cat:
            # Log so you know your _build_features() is drifting from the training contract
            log_warning(
                f"Feature contract gap — numeric missing: {missing_numeric}, "
                f"categorical missing: {missing_cat}. Filling with safe defaults."
            )
            # Fill explicitly here rather than silently inside the detector
            for col in missing_numeric:
                feats[col] = 0.0
            for col in missing_cat:
                feats[col] = ""

        return True, ""

    def _should_alert(self, msg: dict, confidence: float) -> bool:
        """
        Rate-limit alerts to prevent spam.
        Returns True if should create new alert.
        """
        src_ip = _norm_ip(msg.get("src_ip") or "")
        dst_ip = _norm_ip(msg.get("dst_ip") or "")
        src_port = msg.get("src_port")
        dst_port = msg.get("dst_port")
        protocol = msg.get("protocol", "").upper()

        # Use tuple comparison after IP normalization (correct for IPv4 and IPv6)
        if (src_ip, src_port) < (dst_ip, dst_port):
            flow_key = (src_ip, src_port, dst_ip, dst_port)
        else:
            flow_key = (dst_ip, dst_port, src_ip, src_port)

        if confidence <= 0.4:
            conf_bucket = 0
        elif confidence <= 0.7:
            conf_bucket = 1
        else:
            conf_bucket = 2

        dedup_key = (flow_key, protocol, conf_bucket)

        now = time.time()
        cooldown = 30  # seconds

        if dedup_key in self.alert_dedup:
            last_seen = self.alert_dedup[dedup_key]
            if now - last_seen < cooldown:
                return False

        self.alert_dedup[dedup_key] = now
        return True

    def _log_stats(self):
        """Print debug stats every N packets."""
        if self.packet_count % 1000 == 0 and self.packet_count > 0:
            total = self.packet_count + self.dropped_count
            drop_rate = (self.dropped_count / total * 100) if total > 0 else 0

            log_message(f"Stats: received={self.packet_count}, dropped={self.dropped_count} ({drop_rate:.1f}%), alerts={self.alert_count}")

            if self.dropped_reasons:
                top_reasons = sorted(self.dropped_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                log_message(f"Top drop reasons: {top_reasons}")

    def _build_features(self, msg: dict, pkt):
        """
        Build a feature dict that tries to align with UNSW-NB15-style features
        (and your feature_meta.json contract).
        Missing fields are filled with safe defaults.
        """
        src_ip = msg.get("src_ip") or ""
        dst_ip = msg.get("dst_ip") or ""
        proto = msg.get("protocol") or ""
        src_port = int(msg.get("src_port") or 0)
        dst_port = int(msg.get("dst_port") or 0)
        length = float(msg.get("length") or 0)

        dur = float(msg.get("duration") or 0.0)
        sbytes = float(msg.get("src_bytes") or msg.get("sbytes") or length or 0.0)
        dbytes = float(msg.get("dst_bytes") or msg.get("dbytes") or 0.0)

        sttl = 0.0
        if pkt and pkt.haslayer(IP):
            try:
                sttl = float(pkt[IP].ttl)
            except Exception:
                sttl = 0.0

        state = _infer_state(proto, pkt)
        service = _infer_service(dst_port, proto)

        srv = (dst_ip, dst_port)
        self.flow_window.push(src_ip, dst_ip, srv, state, src_port, dst_port)

        feats = {
            "proto": str(proto).lower(),
            "protocol": str(proto),
            "state": str(state).lower(),
            "service": str(service).lower(),

            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": float(src_port),
            "dst_port": float(dst_port),

            "dur": float(dur),
            "length": float(length),

            "sbytes": float(sbytes),
            "dbytes": float(dbytes),
            "sttl": float(sttl),

            "ct_src_ltm": float(self.flow_window.ct_src_ltm(src_ip)),
            "ct_dst_ltm": float(self.flow_window.ct_dst_ltm(dst_ip)),
            "ct_srv_dst": float(self.flow_window.ct_srv_dst(srv)),
            "ct_state_ttl": float(self.flow_window.ct_state_ttl(state)),
            "ct_dst_sport_ltm": float(self.flow_window.ct_dst_sport_ltm(dst_ip, src_port)),
            "ct_src_dport_ltm": float(self.flow_window.ct_src_dport_ltm(src_ip, dst_port)),
            "ct_dst_src_ltm": float(self.flow_window.ct_dst_src_ltm(dst_ip, src_ip)),
        }

        feats.setdefault("Sload", 0.0)
        feats.setdefault("Dload", 0.0)
        feats.setdefault("Sintpkt", 0.0)
        feats.setdefault("tcprtt", 0.0)
        feats.setdefault("smeansz", 0.0)
        feats.setdefault("dmeansz", 0.0)

        return feats

    def process_line(self, line: str):
        """
        Process one JSON line from Rust with robustness guardrails.
        """
        try:
            msg = json.loads(line)
        except Exception as e:
            log_error(f"Invalid JSON from Rust: {e} | line={line[:200]}")
            return

        # GUARDRAIL 1: Schema validation
        is_valid, reason = self._validate_schema(msg)
        if not is_valid:
            self.dropped_count += 1
            self.dropped_reasons[reason] = self.dropped_reasons.get(reason, 0) + 1
            return

        data_b64 = (
            msg.get("raw_data")
            or msg.get("data")
            or msg.get("packet")
            or msg.get("payload")
        )

        if not data_b64:
            self.dropped_count += 1
            self.dropped_reasons["no_raw_data"] = self.dropped_reasons.get("no_raw_data", 0) + 1
            return

        raw = self._decode_packet_bytes(str(data_b64))
        if raw is None:
            self.dropped_count += 1
            self.dropped_reasons["decode_failed"] = self.dropped_reasons.get("decode_failed", 0) + 1
            return

        self.packet_count += 1
        now = time.time()

        if now - self.last_log_time >= 2:
            log_message(f"[PY] receiving... packet_count={self.packet_count}")
            self.last_log_time = now

        self._log_stats()

        pkt = None
        try:
            pkt = Ether(raw)
        except Exception:
            try:
                pkt = IP(raw)   # Fallback: some interfaces send raw IP frames
            except Exception:
                pkt = None

        # 1) Log packet to MongoDB
        packet_id = None
        try:
            packet_doc = {
                "timestamp": msg.get("timestamp"),
                "protocol": msg.get("protocol"),
                "src_ip": msg.get("src_ip"),
                "dst_ip": msg.get("dst_ip"),
                "src_port": msg.get("src_port"),
                "dst_port": msg.get("dst_port"),
                "length": msg.get("length"),
                "raw_data": data_b64,
            }
            packet_id = self.db.log_packet(packet_doc)
        except Exception as e:
            log_error(f"Failed to log packet to DB: {e}")

        # GUARDRAIL 2: Feature extraction with contract-aware completeness check
        try:
            feats = self._build_features(msg, pkt)

            # FIX 2: This now validates against the full 38-feature training contract
            is_complete, reason = self._check_feature_completeness(feats)
            if not is_complete:
                self.dropped_count += 1
                self.dropped_reasons[reason] = self.dropped_reasons.get(reason, 0) + 1
                return

            is_attack, confidence, votes = self.detector.detect(feats)
        except Exception as e:
            log_error(f"Detection failed: {e}")
            self.dropped_count += 1
            self.dropped_reasons["detection_error"] = self.dropped_reasons.get("detection_error", 0) + 1
            return

        # FIX 4: Detection logging and alert escalation are now SEPARATE try/except blocks.
        # A DB write failure on the detection log will no longer suppress the alert.

        # Step 1: Detection logging (non-critical — failure is acceptable)
        try:
            self.db.log_detection(
                packet_data=msg,
                is_attack=is_attack,
                confidence=confidence,
                votes=votes,
                packet_id=packet_id,
            )
        except Exception as e:
            log_error(f"Detection logging failed (non-fatal): {e}")

        # Step 2: Alert escalation (independent of logging success)
        # FIX 1: Uses self.alert_threshold instead of hardcoded 1.0
        if is_attack and confidence >= self.alert_threshold:
            try:
                if self._should_alert(msg, confidence):
                    self.db.log_alert(
                        packet_data=msg,
                        confidence=confidence,
                        votes=votes,
                        packet_id=packet_id,
                    )
                    self.alert_count += 1
            except Exception as e:
                log_error(f"Alert escalation failed: {e}")

    def start_listening(self):
        log_message(f"Starting receiver on {self.host}:{self.port} ...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # FIX 3: Explicit bind error handling — crashes loudly instead of silently
        try:
            server_socket.bind((self.host, self.port))
        except OSError as e:
            log_error(f"Failed to bind to {self.host}:{self.port} — {e}")
            raise

        server_socket.listen(1)
        log_success(f"Listening on {self.host}:{self.port}")

        while True:
            # FIX 3: Accept errors retry instead of crashing the whole receiver
            try:
                conn, addr = server_socket.accept()
            except Exception as e:
                log_error(f"Accept failed: {e} — retrying in 2s")
                time.sleep(2)
                continue

            print("[PY] accepted:", addr, flush=True)
            log_success(f"Connection from {addr}")

            buffer = b""
            last_heartbeat = time.time()
            received_lines = 0

            with conn:
                conn.settimeout(1.0)

                while True:
                    try:
                        chunk = conn.recv(65535)
                    except socket.timeout:
                        if time.time() - last_heartbeat >= 5:
                            log_message(f"[PY] heartbeat: connected, received_lines={received_lines}")
                            last_heartbeat = time.time()
                        continue
                    except Exception as e:
                        log_error(f"Connection error: {e}")
                        break

                    if not chunk:
                        log_warning("Rust sniffer disconnected.")
                        break

                    buffer += chunk

                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        received_lines += 1
                        self.process_line(line)

            # FIX 3: Log warning if Rust disconnected mid-line (incomplete JSON)
            tail = buffer.decode("utf-8", errors="ignore").strip()
            if tail:
                log_warning(f"Processing incomplete trailing line ({len(tail)} bytes) after disconnect")
                received_lines += 1
                self.process_line(tail)

            log_message("Waiting for Rust to reconnect...")