Diagram name: 11-post-training. Type: left-to-right pipeline flow with a highlighted callout on the masked-loss step. Muted palette: ink #1A1D23 for text and outlines, blue #2E86AB for the GPT path, amber #E9A12E for the BDH path and the key callout, grey #5A6472 for secondary labels, background #F7F8FA. No em dashes anywhere in rendered text.

Overall left-to-right flow, five stages:

STAGE 1 (grey box) "BASE MODELS (pretraining, Ch.6)": two stacked file chips, out/gpt.pt (blue, label "GPT 802K params, val 1.76") and out/bdh.pt (amber, label "BDH 1.59M params, val 1.53"). Caption under the box: "next-char predictor: great at continuation, no conversation manners". Arrow to Stage 2.

STAGE 2 (grey box) "CHAT DATA (data/prepare_chat.py)": show one rendered example in a monospace card:
"User: <q>\nAssistant: <a>\n\n"
Below it, the paired tensors as two aligned rows:
  tokens x: [U s e r : ... A s s i s t a n t : ... a n s w e r \n \n]
  target y: [-1 -1 ............................ -1 | real-next-char ...]
Annotate the user + "Assistant:" prefix region with a grey bracket labeled "target = -1 (IGNORED)" and the answer region (including \n\n) with an amber bracket labeled "GRADED". Small note: "cleaned to the ~65-char base vocab; no special tokens". Arrow to Stage 3.

STAGE 3 (AMBER-HIGHLIGHTED box, the visual centerpiece) "SFT with MASKED LOSS (nanobdh/finetune.py)": inside, a mini loop:
  load base out/gpt.pt / out/bdh.pt (dashed arrow coming down from Stage 1 to show weights are reused, same architecture)
  -> model(idx) -> logits (B,T,V)
  -> loss = F.cross_entropy(logits.view(-1,V), targets.view(-1), ignore_index=-1)
  -> AdamW step (small lr, MPS/CPU)
Callout bubble in amber: "KEY IDEA: only the assistant's reply is graded; ignore_index=-1 zeroes the rest". Arrow to Stage 4.

STAGE 4 (grey box) "CHAT MODELS": two file chips out/gpt-chat.pt (blue) and out/bdh-chat.pt (amber), same shape/format as base, label "same weights, new turn-taking behavior". A small red-tinted honesty tag pinned to this stage: "HARD REALITY: learns format, not competence". Arrow to Stage 5.

STAGE 5 (ink-outlined box) "SERVED IN DUAL UI (serve.py + web/index.html)": a browser frame split into two panels. LEFT panel titled "GPT (Transformer)" in blue with stats line "802K / val 1.76". RIGHT panel titled "BDH (Dragon Hatchling)" in amber with stats line "1.59M / val 1.53". One shared input box + Send button at the bottom spanning both panels. Dashed arrows from the shared input fan out to both panels labeled "POST /chat (twice, once per model)". Small note: "stdlib http.server only; generate temp 0.8, top_k 40, stop at \nUser:".

Bottom banner across full width (ink text on bg): "Post-training installs behavior (turn-taking); it cannot add knowledge pretraining never provided."