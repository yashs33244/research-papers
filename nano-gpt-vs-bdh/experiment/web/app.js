/*
  nano-bdh chat front-end logic (vanilla JS, no dependencies).

  The idea:
    - We keep ONE shared conversation history in memory: a list of
      {role: "user"|"assistant", content: string}. But note the two models can
      diverge (they produce different assistant replies), so we actually keep a
      SEPARATE history per model. The USER messages are identical across both;
      only the assistant replies differ.
    - On Send: append the user's message to both panels, then fire two POST /chat
      requests in parallel (one per model). Each panel shows a "thinking..."
      placeholder, replaced by the reply (or an error) when the response lands.

  The server does the prompt building and generation; we just pass the running
  history so it can render the multi-turn format correctly.
*/

(function () {
  "use strict";

  // Per-model conversation state. Same user turns, different assistant turns.
  const histories = {
    gpt: [],
    bdh: [],
  };

  const MODELS = ["gpt", "bdh"];

  const form = document.getElementById("chat-form");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const logs = {
    gpt: document.getElementById("log-gpt"),
    bdh: document.getElementById("log-bdh"),
  };

  // Show a friendly empty-state hint in each panel until the first message.
  MODELS.forEach(function (m) {
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.textContent = "No messages yet. Send something below.";
    hint.dataset.hint = "1";
    logs[m].appendChild(hint);
  });

  function clearEmptyHint(model) {
    const hint = logs[model].querySelector('[data-hint="1"]');
    if (hint) hint.remove();
  }

  // Create and append a message bubble. Returns the element so callers can update
  // it in place (used for the "thinking..." -> reply swap).
  function addMessage(model, role, text) {
    clearEmptyHint(model);
    const el = document.createElement("div");
    el.className = "msg " + role;

    const roleLabel = document.createElement("span");
    roleLabel.className = "role";
    roleLabel.textContent = role === "user" ? "You" : model.toUpperCase();
    el.appendChild(roleLabel);

    const body = document.createElement("span");
    body.className = "body";
    body.textContent = text;
    el.appendChild(body);

    logs[model].appendChild(el);
    scrollToBottom(model);
    return el;
  }

  // Replace a bubble's body text (and optionally its style class).
  function setMessage(el, text, extraClass) {
    el.className = "msg " + (extraClass || "assistant");
    // Keep the role label, replace the body.
    const body = el.querySelector(".body");
    if (body) {
      body.textContent = text;
    } else {
      el.textContent = text;
    }
  }

  // A "thinking..." bubble with animated dots.
  function addThinking(model) {
    clearEmptyHint(model);
    const el = document.createElement("div");
    el.className = "msg thinking";
    el.innerHTML =
      '<span class="role">' +
      model.toUpperCase() +
      '</span><span class="body">thinking<span class="dots"></span></span>';
    logs[model].appendChild(el);
    scrollToBottom(model);
    return el;
  }

  function scrollToBottom(model) {
    const log = logs[model];
    log.scrollTop = log.scrollHeight;
  }

  // Call the server for one model. Returns the reply string or throws.
  async function requestReply(model, history) {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model, history: history }),
    });
    let data;
    try {
      data = await resp.json();
    } catch (e) {
      throw new Error("server returned non-JSON (status " + resp.status + ")");
    }
    if (!resp.ok) {
      throw new Error(data && data.error ? data.error : "request failed");
    }
    return data.reply || "";
  }

  // Handle one model's full round-trip: append user turn, show thinking, fetch,
  // then record the assistant reply into that model's history.
  async function runModel(model, userText) {
    // Append the user's message to this model's history and its panel.
    addMessage(model, "user", userText);
    histories[model].push({ role: "user", content: userText });

    const thinkingEl = addThinking(model);
    try {
      const reply = await requestReply(model, histories[model]);
      setMessage(thinkingEl, reply, "assistant");
      // Re-add the role label the thinking bubble had, since setMessage kept it.
      histories[model].push({ role: "assistant", content: reply });
    } catch (err) {
      setMessage(thinkingEl, "error: " + err.message, "error");
      // Do NOT push a bad assistant turn into history; keep it clean so the next
      // prompt is still well-formed. Drop the just-added user turn's expectation by
      // leaving history as-is (user turn stays; the model simply failed this round).
    }
    scrollToBottom(model);
  }

  async function onSubmit(evt) {
    evt.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    // Lock the composer while both models generate.
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;

    // Fire both models in parallel; each updates its own panel independently.
    await Promise.all(MODELS.map(function (m) { return runModel(m, text); }));

    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }

  form.addEventListener("submit", onSubmit);
  input.focus();
})();
