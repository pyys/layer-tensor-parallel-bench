#!/usr/bin/env python3
"""측정용 프롬프트 픽스처를 만든다.

이광수 「무정」(1917, 퍼블릭 도메인)을 한국어 위키문헌에서 받아
짧은 user 턴과 긴 assistant 턴이 번갈아 나오는 대화 형태로 재구성한다.

텍스트를 저장소에 넣지 않고 리비전 ID 를 고정해 받아오므로,
누가 언제 돌려도 같은 픽스처가 나온다.

토큰 수는 METHOD.md 1-3 의 목표값에 ±1% 로 맞춘다. 실측에는 이미 떠 있는
llama-server 의 /tokenize 를 쓴다 (측정에 쓸 모델과 같은 서버여야 한다).

사용법:
    # 1) 리비전 고정 (최초 1회, 또는 소스를 갱신할 때만)
    python3 make_fixture.py --pin

    # 2) 픽스처 생성 (llama-server 가 떠 있어야 함)
    python3 make_fixture.py --port 8099 --out /root/fixtures/mujeong.json
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

API = "https://ko.wikisource.org/w/api.php"
UA = "layer-tensor-parallel-bench/1.0 (fixture builder)"

PAGES = [
    "무정/1장~20장",
    "무정/21장~40장",
    "무정/41장~60장",
    "무정/61장~80장",
]

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture.lock.json")

# METHOD.md 1-3 의 목표 토큰 수 (콜드, 웜1~4)
TARGETS = [23079, 23177, 23890, 24688, 25470]
TOLERANCE = 0.01

# 턴 모양 — 실사용 대화의 분포를 모사한다 (짧은 질문 / 긴 응답)
USER_TOKENS = 95
ASST_TOKENS = 620

SYSTEM = (
    "다음은 이광수의 장편소설 「무정」(1917)을 발췌해 대화 형식으로 재구성한 것이다. "
    "사용자는 작품의 한 대목을 인용하거나 짧게 묻고, 상대는 이어지는 대목을 길게 서술한다. "
    "이 자료는 추론 성능 측정을 위한 고정 입력이며 내용의 해석이나 평가를 목적으로 하지 않는다. "
    "원문은 한국어 위키문헌에서 가져왔고 저작권이 만료된 퍼블릭 도메인 저작물이다."
)


# ===== 위키문헌 =====
def api_get(params):
    params = dict(params)
    params["format"] = "json"
    params["formatversion"] = "2"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def pin():
    lock = {}
    for title in PAGES:
        data = api_get({"action": "query", "prop": "revisions",
                        "titles": title, "rvprop": "ids", "rvlimit": 1})
        pages = data["query"]["pages"]
        if not pages or pages[0].get("missing"):
            sys.exit("문서를 찾을 수 없다: " + title)
        lock[title] = pages[0]["revisions"][0]["revid"]
        print("%s -> revid %d" % (title, lock[title]))
    with open(LOCK, "w", encoding="utf-8") as f:
        json.dump(lock, f, ensure_ascii=False, indent=2)
    print("기록: " + LOCK)


def fetch_wikitext(revid):
    data = api_get({"action": "query", "prop": "revisions", "revids": revid,
                    "rvprop": "content", "rvslots": "main"})
    return data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]


# ===== 위키텍스트 정리 =====
def clean(wikitext):
    t = wikitext
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    for _ in range(4):                       # 중첩 틀
        t = re.sub(r"\{\{[^{}]*\}\}", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:파일|File|Image|분류|Category):[^\]]*\]\]", "", t)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"^[=]{1,6}.*?[=]{1,6}\s*$", "", t, flags=re.M)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"^[*#:;]+\s*", "", t, flags=re.M)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def to_paragraphs(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 10:
            continue
        # 너무 긴 문단은 문장 단위로 쪼갠다
        if len(line) > 400:
            parts = re.split(r"(?<=[.!?…”\"])\s+", line)
            buf = ""
            for p in parts:
                if len(buf) + len(p) > 400 and buf:
                    out.append(buf.strip())
                    buf = p
                else:
                    buf = (buf + " " + p).strip()
            if buf:
                out.append(buf.strip())
        else:
            out.append(line)
    return out


# ===== 프롬프트 조립 =====
# ⚠️ server_bench.py 의 build_prompts() 와 같은 형태여야 한다.
#    어긋나면 토큰 수가 목표를 벗어나고, server_bench.py --check 가 잡아낸다.
def build_prompt(messages):
    head = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    body = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
                   for m in messages)
    return head + body + "<|im_start|>assistant\n<think>\n</think>\n"


def count_tokens(port, text):
    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % port, payload,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return len(json.loads(r.read())["tokens"])


def grow_to(port, messages, role, paras, cursor, target):
    """target 토큰에 닿을 때까지 문단을 붙인 메시지 하나를 추가한다.

    문단 단위로만 자르므로 문장이 중간에 끊기지 않는다.
    """
    msg = {"role": role, "content": ""}
    messages.append(msg)
    best = None
    while cursor < len(paras):
        candidate = (msg["content"] + "\n" + paras[cursor]).strip()
        msg["content"] = candidate
        n = count_tokens(port, build_prompt(messages))
        if n <= target:
            best = (n, cursor + 1)
            cursor += 1
            continue
        # 넘었다 — 직전 상태가 있으면 되돌리고, 없으면 이 문단 하나는 남긴다
        if best is None:
            return n, cursor + 1
        msg["content"] = msg["content"][:msg["content"].rfind(paras[cursor])].strip()
        return best
    return (best if best else (count_tokens(port, build_prompt(messages)), cursor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="store_true", help="리비전 ID 를 고정한다")
    ap.add_argument("--port", type=int, default=8099, help="llama-server 포트")
    ap.add_argument("--out", default="/root/fixtures/mujeong.json")
    args = ap.parse_args()

    if args.pin:
        pin()
        return

    if not os.path.exists(LOCK):
        sys.exit("리비전 잠금 파일이 없다. 먼저 --pin 을 실행할 것: " + LOCK)
    with open(LOCK, encoding="utf-8") as f:
        lock = json.load(f)

    paras = []
    for title in PAGES:
        if title not in lock:
            continue
        print("받는 중: %s (revid %d)" % (title, lock[title]))
        paras += to_paragraphs(clean(fetch_wikitext(lock[title])))
    print("문단 %d개, 총 %d자" % (len(paras), sum(len(p) for p in paras)))

    messages = []
    cursor = 0
    marks = []          # 각 목표 지점에서의 메시지 개수

    # 콜드 — assistant 로 끝나도록 user/assistant 를 번갈아 채운다
    while True:
        n = count_tokens(port=args.port, text=build_prompt(messages)) if messages else 0
        if n >= TARGETS[0] * (1 - TOLERANCE) and messages and messages[-1]["role"] == "assistant":
            break
        role = "user" if (len(messages) % 2 == 0) else "assistant"
        remain = TARGETS[0] - n
        want = USER_TOKENS if role == "user" else ASST_TOKENS
        n, cursor = grow_to(args.port, messages, role, paras, cursor,
                            n + min(want, max(remain, 40)))
        if cursor >= len(paras):
            sys.exit("원문이 부족하다. PAGES 에 장을 더 추가할 것.")
    marks.append(len(messages))
    print("콜드: %d 토큰 (목표 %d), 메시지 %d개" % (n, TARGETS[0], len(messages)))

    # 웜 — 목표마다 user 또는 assistant+user 를 덧붙인다
    for i, target in enumerate(TARGETS[1:], start=1):
        if i > 1:
            n, cursor = grow_to(args.port, messages, "assistant", paras, cursor,
                                target - USER_TOKENS)
        n, cursor = grow_to(args.port, messages, "user", paras, cursor, target)
        marks.append(len(messages))
        print("웜 %d: %d 토큰 (목표 %d, 오차 %+.2f%%)"
              % (i, n, target, (n - target) / target * 100))
        if cursor >= len(paras):
            sys.exit("원문이 부족하다. PAGES 에 장을 더 추가할 것.")

    out = {
        "id": "mujeong-fixture-v1",
        "persona": "무정 픽스처",
        "personaContent": SYSTEM,
        "source": {"work": "이광수 무정 (1917)", "site": "ko.wikisource.org",
                   "license": "public domain", "revisions": lock},
        "targets": TARGETS,
        "marks": marks,
        "messages": messages,
        "memory_chunks": [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("기록: %s (메시지 %d개)" % (args.out, len(messages)))
    print("marks =", marks, " -> server_bench.py 의 COLD_END 를", marks[0], "로 설정")


main()
