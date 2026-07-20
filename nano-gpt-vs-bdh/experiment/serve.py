"""Dual-model chat server for nano-bdh, using ONLY the Python standard library.

BEGINNER FIRST
--------------
This is a tiny web server (no Flask, no FastAPI, no uvicorn - just http.server and
json from the standard library) that hosts a side-by-side chat comparing our two
fine-tuned models:

  - out/gpt-chat.pt   the Transformer, post-trained for chat
  - out/bdh-chat.pt   the Dragon Hatchling, post-trained for chat

It serves the web UI at http://localhost:8000 and answers POST /chat requests by
building a prompt from the conversation history, generating a reply with the chosen
model, and returning the cleaned assistant text.

HONEST NOTE: these are ~1-2M param, character-level models fine-tuned on a few
thousand short dialogues. Replies will follow the chat FORMAT but read as broken
English. That is expected; this is a teaching demonstration of the SFT method.

THE CHAT FORMAT (must match prepare_chat.py and finetune.py exactly)
--------------------------------------------------------------------
A single turn is rendered as:

    User: <question>\nAssistant: <answer>\n\n

To answer, we render the whole history and leave a dangling "Assistant: " for the
model to complete:

    User: <q1>\nAssistant: <a1>\n\nUser: <q2>\nAssistant:<space>

We then generate characters until the model emits "\nUser:" (the start of the next
human turn) or we hit max_new_tokens. We strip any trailing "\nUser:" fragment so
the returned reply is only the assistant's text.

ROUTES
------
  GET  /            -> web/index.html
  GET  /style.css   -> web/style.css   (if present)
  GET  /app.js      -> web/app.js       (if present)
  POST /chat        -> {"reply": "..."} given {"model","history":[{role,content}...]}

Run:
    python serve.py
    # then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from nanobdh.tokenizer import CharTokenizer
from nanobdh.model_gpt import GPT, GPTConfig
from nanobdh.model_bdh import BDH, BDHConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_THIS_DIR, "out")
WEB_DIR = os.path.join(_THIS_DIR, "web")

HOST = "localhost"
PORT = 8000

# Generation defaults (per contract 7). Temperature ~0.8 keeps replies coherent-ish;
# top_k ~40 prunes the long tail of nonsense characters; 200 chars is plenty for a
# short turn given block_size 128.
TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 200

# The stop marker: once the model writes the start of the next human turn, the
# assistant's reply is over. We cut it there.
STOP_SEQUENCE = "\nUser:"


def pick_device() -> str:
    """MPS (Apple GPU) if available, else CUDA, else CPU. Matches the rest of the repo."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rebuild_model(model_name: str, config: dict):
    """Reconstruct the architecture from a saved config dict."""
    if model_name == "gpt":
        return GPT(GPTConfig(**config))
    if model_name == "bdh":
        return BDH(BDHConfig(**config))
    raise ValueError(f"unknown model {model_name!r}")


class ChatModel:
    """One loaded chat model plus everything needed to generate a reply from history.

    We keep the tokenizer, the model, its block_size, and its parameter count so the
    UI can show honest stats. All generation logic lives here so the request handler
    stays a thin wrapper.
    """

    def __init__(self, ckpt_path: str, device: str):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.name = ckpt["model"]                      # "gpt" or "bdh"
        self.tok = CharTokenizer(ckpt["vocab"])        # exact vocab from the checkpoint
        self.model = rebuild_model(self.name, ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.block_size = ckpt["config"]["block_size"]
        self.num_params = self.model.num_params()

    def build_prompt(self, history) -> str:
        """Render the conversation history into the exact chat text format.

        history is a list of {"role": "user"|"assistant", "content": str}. We emit a
        completed "User:/Assistant:" turn for each answered exchange, then leave a
        dangling "Assistant: " for the FINAL user message so the model completes it.

        We are permissive about shape: we walk the history and, whenever we see a user
        message, we open a turn; if an assistant message follows, we close it with the
        turn terminator "\n\n". The last user message is left open (no answer yet).
        """
        parts = []
        i = 0
        n = len(history)
        while i < n:
            msg = history[i]
            role = msg.get("role")
            content = (msg.get("content") or "")
            if role == "user":
                # Does an assistant reply follow this user message?
                if i + 1 < n and history[i + 1].get("role") == "assistant":
                    ans = history[i + 1].get("content") or ""
                    parts.append(f"User: {content}\nAssistant: {ans}\n\n")
                    i += 2
                else:
                    # Final unanswered user turn: leave "Assistant: " dangling.
                    parts.append(f"User: {content}\nAssistant: ")
                    i += 1
            else:
                # A stray assistant message with no preceding user; render it plainly
                # so context is preserved, then move on.
                parts.append(f"Assistant: {content}\n\n")
                i += 1
        return "".join(parts)

    def clean_to_vocab(self, text: str) -> str:
        """Drop any characters not in this model's vocab so encode() never crashes.

        User input from the browser can contain arbitrary characters (emoji, tabs,
        curly quotes). The base vocab is only ~65 chars, so we filter defensively:
        map a couple of common cases, drop the rest. This mirrors prepare_chat.py's
        spirit but is intentionally minimal (we only need encode() to be safe).
        """
        vocab = set(self.tok.chars)
        # Keys for the fancy unicode cases are written via chr(codepoint) so this
        # source file contains only plain ASCII punctuation.
        simple_map = {
            "\t": " ",
            "\r": "",
            chr(0x2019): "'",   # right single quote -> apostrophe
            chr(0x2018): "'",   # left single quote  -> apostrophe
            chr(0x201C): "",    # left double quote  -> drop
            chr(0x201D): "",    # right double quote -> drop
            '"': "",            # straight double quote -> drop
            chr(0x2014): "-",   # em dash -> hyphen
            chr(0x2013): "-",   # en dash -> hyphen
        }
        out = []
        for ch in text:
            if ch in vocab:
                out.append(ch)
            elif ch in simple_map:
                for rc in simple_map[ch]:
                    if rc in vocab:
                        out.append(rc)
            # else: silently drop the character
        return "".join(out)

    @torch.no_grad()
    def generate_reply(self, history) -> str:
        """Build the prompt, generate char-by-char, stop at the next user turn.

        We do NOT reuse model.generate() because we need to STOP EARLY the moment the
        model starts a new "\nUser:" turn. So we run the same crop -> logits ->
        temperature -> top_k -> multinomial sampling loop here, appending one
        character at a time and checking a small rolling tail for the stop sequence.
        """
        # Sanitize each message so encoding is always safe.
        safe_history = [
            {"role": m.get("role"), "content": self.clean_to_vocab(m.get("content") or "")}
            for m in history
        ]
        prompt = self.build_prompt(safe_history)

        # Encode and seed the running context. We keep the freshly generated
        # characters separately so we can watch for the stop sequence and return
        # only the assistant's own text.
        idx = torch.tensor(
            [self.tok.encode(prompt)], dtype=torch.long, device=self.device
        )
        generated_chars = []

        for _ in range(MAX_NEW_TOKENS):
            idx_cond = idx[:, -self.block_size:]           # crop to the model's window
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / TEMPERATURE        # next-char scores
            if TOP_K is not None:
                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            ch = self.tok.decode([int(idx_next.item())])
            generated_chars.append(ch)

            # Check whether the accumulated text now contains the stop sequence.
            # We only need to test the tail, but joining is cheap for <=200 chars.
            text_so_far = "".join(generated_chars)
            if STOP_SEQUENCE in text_so_far:
                text_so_far = text_so_far.split(STOP_SEQUENCE, 1)[0]
                return self._postprocess(text_so_far)

        return self._postprocess("".join(generated_chars))

    @staticmethod
    def _postprocess(reply: str) -> str:
        """Trim whitespace and any dangling fragment of the stop marker.

        The turn ends with "\n\n"; we strip trailing whitespace, and also guard
        against a partial "\nUser" fragment that could appear if generation stopped
        just before the full stop sequence formed.
        """
        reply = reply.rstrip()
        # Remove any trailing partial start-of-next-turn fragment, longest first.
        for frag in ("\nUser:", "\nUser", "\nUse", "\nUs", "\nU", "\n"):
            if reply.endswith(frag):
                reply = reply[: -len(frag)].rstrip()
                break
        return reply.strip()


# --------------------------------------------------------------------------------
# Load both models once at startup (module-level so every request reuses them).
# --------------------------------------------------------------------------------
DEVICE = pick_device()
MODELS: dict[str, ChatModel] = {}


def load_models(use_base: bool = False):
    """Load the two models into MODELS. Fail loudly with a helpful hint.

    Default: the SFT chat checkpoints (out/gpt-chat.pt, out/bdh-chat.pt) - these
    were post-trained to follow the User/Assistant format.

    With use_base=True: load the RAW base checkpoints (out/gpt.pt, out/bdh.pt)
    instead. Those are pure next-character Shakespeare models that never saw the
    chat format, so they will CONTINUE your prompt in Shakespeare style rather
    than answer it. Handy for a base-vs-chat comparison in the same UI.
    """
    suffix = "" if use_base else "-chat"
    kind = "BASE (pre-chat, Shakespeare completion)" if use_base else "CHAT (SFT)"
    print(f"device: {DEVICE}  |  serving {kind} models")
    for name in ("gpt", "bdh"):
        path = os.path.join(OUT_DIR, f"{name}{suffix}.pt")
        if not os.path.exists(path):
            hint = (f"  python -m nanobdh.train --model {name}" if use_base else
                    f"  python data/prepare_chat.py\n"
                    f"  python -m nanobdh.finetune --model {name} --max_iters 800")
            raise FileNotFoundError(f"missing {path}. Train it first:\n{hint}")
        MODELS[name] = ChatModel(path, DEVICE)
        print(f"loaded {name}{suffix}.pt  | params {MODELS[name].num_params:,}")


class Handler(BaseHTTPRequestHandler):
    """Routes GET (static files) and POST /chat (generation)."""

    # Quieter logging: one line per request is enough for a teaching server.
    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        print("[serve] " + (fmt % args))

    # ---- small helpers ------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Same-origin app, but a permissive CORS header keeps local tinkering easy.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _send_file(self, rel_path: str, content_type: str):
        full = os.path.join(WEB_DIR, rel_path)
        if not os.path.exists(full):
            self._send_json(404, {"error": f"{rel_path} not found"})
            return
        with open(full, "rb") as f:
            self._send(200, f.read(), content_type)

    # ---- GET: static files --------------------------------------------------
    def do_GET(self):  # noqa: N802 - stdlib signature
        if self.path in ("/", "/index.html"):
            self._send_file("index.html", "text/html; charset=utf-8")
        elif self.path == "/style.css":
            self._send_file("style.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file("app.js", "application/javascript; charset=utf-8")
        else:
            self._send_json(404, {"error": "not found"})

    # ---- CORS preflight (harmless for same-origin, handy if opened elsewhere) --
    def do_OPTIONS(self):  # noqa: N802 - stdlib signature
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---- POST /chat: generate a reply --------------------------------------
    def do_POST(self):  # noqa: N802 - stdlib signature
        if self.path != "/chat":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        model_name = payload.get("model")
        history = payload.get("history", [])
        if model_name not in MODELS:
            self._send_json(
                400, {"error": f"unknown model {model_name!r}; expected 'gpt' or 'bdh'"}
            )
            return
        if not isinstance(history, list):
            self._send_json(400, {"error": "history must be a list of messages"})
            return

        try:
            reply = MODELS[model_name].generate_reply(history)
        except Exception as e:  # noqa: BLE001 - surface generation errors as JSON
            self._send_json(500, {"error": f"generation failed: {e}"})
            return

        # If the model produced nothing usable (all stripped), say so honestly rather
        # than returning an empty bubble.
        if not reply:
            reply = "(the tiny model produced no readable reply this time)"
        self._send_json(200, {"reply": reply})


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Serve the nano-bdh dual chat UI.")
    ap.add_argument("--base", action="store_true",
                    help="serve the raw base (pre-chat) models instead of the SFT chat models")
    args = ap.parse_args()
    load_models(use_base=args.base)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"serving nano-bdh chat at http://{HOST}:{PORT}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
