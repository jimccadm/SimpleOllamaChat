#!/usr/bin/env python3
"""
read_docs.py

Summarize a text document in French and English using Ollama + gpt-oss:20b.

Examples:
  python3 read_docs.py -f mytestdocument.txt
  python3 read_docs.py -f mytestdocument.txt --stream

Requirements:
  - Ollama installed and running (default: http://localhost:11434)
  - Model pulled: ollama pull gpt-oss:20b
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import textwrap
import urllib.request
import urllib.error
from typing import Iterator, Optional


DEFAULT_HOST = "http://localhost:11434"
MODEL_NAME = "gpt-oss:20b"


def read_text_file(path: str, max_bytes: int = 50 * 1024 * 1024) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"File too large ({size} bytes). Limit is {max_bytes} bytes.")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def post_json(url: str, payload: dict, timeout: int = 600) -> urllib.request.addinfourl:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def ollama_generate(
    host: str,
    prompt: str,
    stream: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_predict: int = 650,
    seed: Optional[int] = None,
) -> str:
    """
    Calls Ollama's /api/generate endpoint.

    If stream=False: returns full text.
    If stream=True : prints tokens as they arrive and returns the final text.
    """
    url = host.rstrip("/") + "/api/generate"

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": num_predict,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "options": options,
    }

    try:
        resp = post_json(url, payload)
        raw = resp.read().decode("utf-8", errors="replace")

        if not stream:
            # Non-stream response is a single JSON object
            data = json.loads(raw)
            return data.get("response", "").strip()

        # Stream response is newline-delimited JSON objects
        out_parts = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            chunk = obj.get("response", "")
            if chunk:
                print(chunk, end="", flush=True)
                out_parts.append(chunk)
            if obj.get("done", False):
                break
        print()  # newline after streaming
        return "".join(out_parts).strip()

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"Ollama HTTP error {e.code}: {e.reason}\n{body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to reach Ollama at {host}. Is it running?\n"
            f"Try: ollama serve\n"
            f"Original error: {e}"
        ) from e


def build_prompt(document_text: str) -> str:
    # Keep instructions very explicit so the output is predictable.
    # You can adjust the requested length here.
    return textwrap.dedent(
        f"""
        You are a careful summarizer.

        Summarize the following document in BOTH English and French.

        Requirements:
        - Output MUST be exactly two sections with these headings:
          ENGLISH SUMMARY:
          FRENCH SUMMARY:
        - Each summary should be concise but informative (about 8–14 bullet points).
        - Include key facts, names, dates, numbers if present.
        - If the document has a clear conclusion or action items, include them.
        - Do not invent anything not in the text.

        Document:
        \"\"\"{document_text}\"\"\"
        """
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a text document in French and English using Ollama (gpt-oss:20b)."
    )
    parser.add_argument("-f", "--file", required=True, help="Path to input text document (UTF-8).")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama host URL (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens as they arrive (prints as the model generates).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=120_000,
        help="Max characters of input to send to the model (default: 120000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for more reproducible outputs.",
    )
    args = parser.parse_args()

    try:
        doc = read_text_file(args.file)
        if not doc:
            die_msg = "Input file is empty."
            print(die_msg, file=sys.stderr)
            return 2

        # Trim very long inputs to reduce context overflow risk
        if len(doc) > args.max_chars:
            warn = (
                f"NOTE: Input is {len(doc)} chars; trimming to last {args.max_chars} chars "
                f"(use --max-chars to change).\n"
            )
            print(warn, file=sys.stderr)
            doc = doc[-args.max_chars :]

        prompt = build_prompt(doc)

        print(f"Using model: {MODEL_NAME}")
        print(f"Ollama host: {args.host}\n")
        if args.stream:
            print("(Streaming enabled)\n")

        result = ollama_generate(
            host=args.host,
            prompt=prompt,
            stream=args.stream,
            temperature=0.2,
            top_p=0.9,
            num_predict=900,
            seed=args.seed,
        )

        if not args.stream:
            print(result)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
