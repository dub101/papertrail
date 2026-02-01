import os
import sys
import json
import httpx
import signal
from typing import Optional
from contextlib import contextmanager

# Optional: load .env if present (requires `pip install python-dotenv`)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

API_URL = os.getenv("PAPER_RESEARCHER_API", "http://localhost:8000").rstrip("/")
CHAT_ENDPOINT = f"{API_URL}/chat"


@contextmanager
def handle_interrupt():
    """Context manager to restore original Ctrl+C handler after streaming."""
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)


def parse_sse_event(lines: list[str]) -> tuple[Optional[str], str]:
    """
    Parse a single SSE event from accumulated lines.
    Expects lines like:
      event: delta
      data: {"text":"..."}
    Returns: (event_type, data_string)
    """
    event_type: Optional[str] = None
    data_lines: list[str] = []

    for line in lines:
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    data = "\n".join(data_lines)
    return event_type, data


def main():
    conversation_id: Optional[str] = None
    print("Welcome to the Paper Researcher CLI! Type /quit or /exit to leave.")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                print("Goodbye!")
                break

            payload = {
                "conversation_id": conversation_id,
                "message": user_input,
                "top_n": 5,
            }

            with handle_interrupt():
                with httpx.stream("POST", CHAT_ENDPOINT, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        try:
                            print(f"Error: {response.status_code} - {response.text}")
                        except Exception:
                            print(f"Error: {response.status_code}")
                        continue

                    buffer: list[str] = []
                    sources: list[dict] = []

                    for line in response.iter_lines():  # line is str
                        if line == "":
                            event_type, data = parse_sse_event(buffer)
                            buffer = []

                            if not event_type:
                                continue

                            if event_type == "status":
                                status_data = json.loads(data) if data else {}
                                print(f"[status] {status_data.get('message', '')}")

                            elif event_type == "delta":
                                delta_data = json.loads(data) if data else {}
                                print(delta_data.get("text", ""), end="", flush=True)

                            elif event_type == "sources":
                                sources_data = json.loads(data) if data else {}
                                sources.extend(sources_data.get("items", []))

                            elif event_type == "done":
                                done_data = json.loads(data) if data else {}
                                conversation_id = done_data.get("conversation_id", conversation_id)
                                print()  # newline after streamed text

                                if sources:
                                    print("Sources:")
                                    for src in sources:
                                        arxiv_id = src.get("arxiv_id", "")
                                        url = src.get("url", "")
                                        print(f"- {arxiv_id}: {url}")
                                break

                            elif event_type == "error":
                                error_data = json.loads(data) if data else {}
                                print(f"Error: {error_data.get('message', '')}")
                                break

                        else:
                            buffer.append(line)

        except KeyboardInterrupt:
            print("\n[Interrupted] Returning to prompt.")
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
