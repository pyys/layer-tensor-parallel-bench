#!/usr/bin/env python3
"""레이어 병렬 vs 텐서 병렬 12셀 매트릭스 측정 하네스.

셀 = {layer, tensor} x {MTP off, on} x {GPU 2, 3, 4장}

각 셀마다 llama-server 를 띄우고, 실제 대화에서 잘라낸 프롬프트로
콜드 1회 + 웜 5회를 던진 뒤 결과를 jsonl 로 남긴다.

프롬프트는 12셀 모두 동일하다. 웜 턴에서 모델의 자기 출력이 아니라
기록된 응답을 이어붙이므로 셀 간 발산이 없고, ignore_eos + 고정 n_predict
로 생성 토큰 수까지 같아진다.

중단 후 다시 실행하면 이미 끝난 셀은 건너뛴다.

사용법:
    python3 server_bench.py
    python3 server_bench.py --dry-run          # 프롬프트 길이만 확인
    python3 server_bench.py --only tensor-mtp-4   # 셀 하나만
"""
import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ===== 설정 =====
BIN = "/root/llama.cpp-nccl/build/bin/llama-server"
MODEL = "/root/models/qwen38-heretic-Q3KM-mtp.gguf"
CHAT = "/root/fixtures/mujeong.json"   # make_fixture.py 산출물
OUTDIR = "/root/bench"
PORT = 8099

N_CTX = 27000        # 2장에서 NCCL P2P 버퍼까지 확보되는 상한 (실측)
N_PREDICT = 400
SEED = 1234

# 보정: 이 대화는 1.871 자/토큰 (메시지 99까지 67,292자 = 실측 35,962토큰)
# 정확한 값은 --check 로 확인할 것. 마지막 웜 턴 + N_PREDICT 가 N_CTX 안에 들어가야 한다.
COLD_END = 66        # 메시지 0..65 = 43,204자 = 약 23,100토큰
N_WARM = 4           # 그 뒤 user 메시지 4개. 턴당 약 780토큰 증가
CHARS_PER_TOKEN = 1.871

HEALTH_TIMEOUT = 600     # 모델 로드 대기 (초)
REQUEST_TIMEOUT = 3600   # 콜드 프리필이 길어질 수 있다

SPLIT_MODES = ["layer", "tensor"]
MTP_MODES = [False, True]
GPU_COUNTS = [4, 3, 2]   # 빠른 조합부터 (실패해도 앞쪽 결과는 남는다)


# ===== 프롬프트 조립 =====
# persona_chat.html 이 브라우저에서 만드는 형태를 따른다.
def build_prompts():
    with open(CHAT, encoding="utf-8") as f:
        data = json.load(f)

    msgs = data["messages"]
    head = "<|im_start|>system\n" + data.get("personaContent", "") + "<|im_end|>\n"

    def block(m):
        role = m.get("role")
        if role not in ("user", "assistant"):
            return ""     # image 등은 건너뛴다 (합쳐야 78자)
        return "<|im_start|>" + role + "\n" + str(m.get("content", "")) + "<|im_end|>\n"

    tail = "<|im_start|>assistant\n<think>\n</think>\n"

    # 콜드: 0..COLD_END-1
    body = head + "".join(block(m) for m in msgs[:COLD_END])
    prompts = [body + tail]

    # 웜: COLD_END 이후의 user 메시지를 순서대로. 그 사이 assistant 응답은
    # 기록된 것을 그대로 이어붙인다 (셀 간 동일한 입력을 보장하기 위해).
    i = COLD_END
    while len(prompts) <= N_WARM and i < len(msgs):
        body += block(msgs[i])
        if msgs[i].get("role") == "user":
            prompts.append(body + tail)
        i += 1

    if len(prompts) != N_WARM + 1:
        sys.exit("프롬프트를 %d개만 만들었다 (기대 %d개). COLD_END 를 조정할 것."
                 % (len(prompts), N_WARM + 1))
    return prompts


# ===== 서버 제어 =====
def server_args(split_mode, mtp, ngpu):
    args = [BIN, "-m", MODEL, "-ngl", "99", "-c", str(N_CTX), "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT),
            "--split-mode", split_mode]
    if mtp:
        args += ["--spec-type", "draft-mtp"]
    return args


def wait_health():
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def count_tokens(prompt):
    payload = json.dumps({"content": prompt}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % PORT, payload,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.loads(r.read())["tokens"])


def post_completion(prompt):
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "ignore_eos": True,
        "cache_prompt": True,
        "seed": SEED,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, payload,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        return json.loads(r.read())


# ===== 셀 실행 =====
def run_cell(name, split_mode, mtp, ngpu, prompts):
    os.makedirs(OUTDIR, exist_ok=True)
    log_path = os.path.join(OUTDIR, name + ".server.log")
    smi_path = os.path.join(OUTDIR, name + ".smi.csv")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(ngpu))

    log = open(log_path, "w", encoding="utf-8")
    srv = subprocess.Popen(server_args(split_mode, mtp, ngpu),
                           stdout=log, stderr=subprocess.STDOUT, env=env)

    smi_f = open(smi_path, "w", encoding="utf-8")
    smi = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,utilization.gpu,power.draw,temperature.gpu,memory.used",
         "--format=csv,noheader", "-l", "1"],
        stdout=smi_f, stderr=subprocess.DEVNULL)

    result = {"cell": name, "split_mode": split_mode, "mtp": mtp, "n_gpu": ngpu,
              "n_ctx": N_CTX, "n_predict": N_PREDICT, "turns": []}
    try:
        if not wait_health():
            result["error"] = "health timeout (로드 실패 또는 OOM). %s 확인" % log_path
            return result

        for idx, p in enumerate(prompts):
            t0 = time.time()
            try:
                r = post_completion(p)
            except Exception as e:                    # noqa: BLE001
                result["error"] = "turn %d 실패: %r" % (idx, e)
                return result
            wall = time.time() - t0
            t = r.get("timings", {})
            result["turns"].append({
                "turn": idx,
                "kind": "cold" if idx == 0 else "warm",
                "wall_s": round(wall, 2),
                "prompt_n": t.get("prompt_n"),
                "prompt_ms": t.get("prompt_ms"),
                "prompt_tps": t.get("prompt_per_second"),
                "cache_n": t.get("cache_n"),
                "predicted_n": t.get("predicted_n"),
                "predicted_ms": t.get("predicted_ms"),
                "predicted_tps": t.get("predicted_per_second"),
                "draft_n": t.get("draft_n"),
                "draft_n_accepted": t.get("draft_n_accepted"),
            })
            print("    turn %d  pp %s tok @ %.1f t/s | tg %s tok @ %.2f t/s | %.1fs"
                  % (idx, t.get("prompt_n"), t.get("prompt_per_second") or 0,
                     t.get("predicted_n"), t.get("predicted_per_second") or 0, wall),
                  flush=True)
        return result
    finally:
        smi.send_signal(signal.SIGTERM)
        smi.wait(timeout=10)
        smi_f.close()
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="이미 떠 있는 서버의 /tokenize 로 정확한 토큰 수를 잰다")
    ap.add_argument("--only", default=None, help="셀 이름 하나만 실행")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "results.jsonl"))
    args = ap.parse_args()

    prompts = build_prompts()
    print("프롬프트 %d개 (콜드 1 + 웜 %d)" % (len(prompts), N_WARM))
    for i, p in enumerate(prompts):
        print("  [%d] %d 자 (추정 %d 토큰)" % (i, len(p), int(len(p) / CHARS_PER_TOKEN)))
    if args.check:
        print("--- /tokenize 실측 (포트 %d 서버 필요) ---" % PORT)
        worst = 0
        for i, p in enumerate(prompts):
            n = count_tokens(p)
            worst = max(worst, n)
            print("  [%d] %s: %d 토큰" % (i, "cold" if i == 0 else "warm", n))
        need = worst + N_PREDICT
        print("최대 프롬프트 %d + n_predict %d = %d / n_ctx %d (여유 %d)"
              % (worst, N_PREDICT, need, N_CTX, N_CTX - need))
        if need > N_CTX:
            print("!! N_CTX 초과. COLD_END 를 낮출 것 (프롬프트가 잘려 측정이 오염된다)")
        return

    if args.dry_run:
        return

    os.makedirs(OUTDIR, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("error"):
                    done.add(row["cell"])

    cells = []
    for sm, mtp, ngpu in itertools.product(SPLIT_MODES, MTP_MODES, GPU_COUNTS):
        name = "%s-%s-%d" % (sm, "mtp" if mtp else "nomtp", ngpu)
        cells.append((name, sm, mtp, ngpu))

    for name, sm, mtp, ngpu in cells:
        if args.only and name != args.only:
            continue
        if name in done:
            print("[skip] %s (완료됨)" % name, flush=True)
            continue
        print("[run ] %s" % name, flush=True)
        t0 = time.time()
        row = run_cell(name, sm, mtp, ngpu, prompts)
        row["elapsed_s"] = round(time.time() - t0, 1)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row.get("error"):
            print("       ERROR: %s" % row["error"], flush=True)
        print("       %.0f초" % row["elapsed_s"], flush=True)


main()
