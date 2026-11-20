import os
import sys
import signal
from receiver import NIDSReceiver
from utils import log_message, log_success, log_error

# handles Ctrl+C so the engine exits cleanly instead of crashing out
def signal_handler(sig, frame):
    print("\n")
    log_message("Shutting down NIDS Detection Engine...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    print("NIDS Python Detection Engine v1.0")
    print("Starting detection engine...\n")

    try:
        log_message("Initializing NIDS Detection Engine...")

        # threshold is configurable at runtime via env var, default is 2/3 model agreement
        # e.g. on Windows: set NIDS_ALERT_THRESHOLD=0.5
        threshold = float(os.environ.get("NIDS_ALERT_THRESHOLD", "0.66"))
        log_message(f"Alert confidence threshold: {threshold} (set NIDS_ALERT_THRESHOLD to override)")

        receiver = NIDSReceiver(host='127.0.0.1', port=8080, alert_threshold=threshold)

        log_success("Detection Engine initialized successfully!")
        log_message("Waiting for connection from Rust packet sniffer...")
        print()

        receiver.start_listening()

    except KeyboardInterrupt:
        log_message("\nReceived interrupt signal")
    except Exception as e:
        log_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()