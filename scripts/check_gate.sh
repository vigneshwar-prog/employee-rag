#!/usr/bin/env bash
# Verify the local Claude gate at localhost:8080 is fully working.
# Usage: bash scripts/check_gate.sh
set -u
BASE="${GATE_BASE:-http://localhost:8080}"
MODEL="${GATE_MODEL:-claude-sonnet-5}"
pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; }

echo "Checking gate at $BASE ..."

# 1. port alive
code=$(curl -sS -m 5 "$BASE/" -o /dev/null -w "%{http_code}" 2>/dev/null)
[ "$code" = "200" ] && pass "root responds (HTTP $code)" || { fail "root not reachable (HTTP ${code:-none})"; exit 1; }

# 2. health
health=$(curl -sS -m 5 "$BASE/health" 2>/dev/null)
echo "$health" | grep -q '"status":"ok"' && pass "health ok — $health" || fail "health not ok — ${health:-none}"

# 3. models / API shape
models=$(curl -sS -m 5 "$BASE/v1/models" 2>/dev/null)
echo "$models" | grep -q '"object":"list"' && pass "OpenAI-compatible /v1/models" || fail "unexpected /v1/models"

# 4. real inference (conclusive)
reply=$(curl -sS -m 30 "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: RAG_TEST_OK\"}],\"max_tokens\":20}" \
  2>/dev/null)
if echo "$reply" | grep -q "RAG_TEST_OK"; then
  pass "inference works ($MODEL replied with sentinel)"
  echo; echo "Gate is fully operational. ✅"
else
  fail "inference failed — response: ${reply:-none}"; exit 1
fi
