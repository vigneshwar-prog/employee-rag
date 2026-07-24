# Verifying the Local Claude Gate (localhost:8080)

How we confirm the gate is actually working — not just that the port is open.
Run these anytime the app can't reach the LLM, or at the start of a session.

**Principle:** go cheapest → most conclusive. A port can be open but not serving the API,
so don't stop at "it responds" — finish with a real model reply.

## The 4 checks

### 1. Is anything on the port?
```bash
curl -sS -m 5 http://localhost:8080/ -o /dev/null -w "HTTP %{http_code}\n"
```
Expect `HTTP 200`. (`-m 5` timeout, `-w` prints status, `-o /dev/null` discards body.)

### 2. Is it healthy?
```bash
curl -sS -m 5 http://localhost:8080/health
```
Expect `{"status":"ok",...}`.

### 3. API shape + available models (confirms OpenAI-compatible)
```bash
curl -sS -m 5 http://localhost:8080/v1/models
```
Expect `{"object":"list","data":[{"id":"claude-..."}, ...]}`.
The `/v1/models` path + this JSON = OpenAI-compatible → LangChain `ChatOpenAI` works.

### 4. Real inference (THE conclusive test)
```bash
curl -sS -m 30 http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-5","messages":[{"role":"user","content":"Reply with exactly: RAG_TEST_OK"}],"max_tokens":20}'
```
Expect the reply to contain `RAG_TEST_OK`. Use a **unique sentinel** (not "hello") so a
canned/cached response can't fool you. This proves the full path: HTTP → model → parsing.

## Why the order matters
Each step costs more but proves more. 1–2 rule out "port dead" instantly. 3 confirms the
contract. **Only 4 proves inference actually works.**

## One-shot script
Run: `bash scripts/check_gate.sh`
