# METHOD — 측정 방법론 (동결 v1.2)

동결일 2026-09-15.

> ⚠️ **영문 [`METHOD.md`](METHOD.md) 이 정본이다.** 어긋나면 영문이 맞다.

이 문서는 **플랫폼 간 비교 가능성을 보장하기 위한 계약**이다.
여기 "고정"이라고 적힌 항목이 하나라도 바뀌면 **다른 플랫폼의 결과와 같은 표에
놓을 수 없다.** 바꿔야 한다면 버전을 올리고, 이전 버전으로 측정된 결과는
별도 표로 분리한다.

각 플랫폼 보고서(`docs/<platform>.md`)는 이 문서를 참조만 하고 방법을 다시
적지 않는다. 결과와 해석은 [`FINDINGS.ko.md`](FINDINGS.ko.md) 에 있다.

---

## 1. 고정 항목 — 변경 금지

### 1-1. 모델

| 항목 | 값 |
|---|---|
| 원본 | `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` 의 `RVN-BF16-mtp.gguf` [{ref 11.}](#참고문헌) |
| 양자화 | **Q3_K_M** |
| 크기 | 13,752,763,616 바이트 (13,115 MiB) |
| sha256 | `727e7d8b3ab3af1a6eeb166f3d7aac93d51ac0cdc06200e8f1cbac9074591ca4` |
| 아키텍처 | `qwen35`, 65블록(실층 64 + MTP 헤드 1), `full_attention_interval = 4` |

Q3_K_M 을 고른 이유는 품질이 아니라 **8GB 카드 2장(16GB)에 들어가는 최대 크기**
이기 때문이다. 이보다 크면 저사양 플랫폼에서 N=2 셀을 잴 수 없어 스케일링
곡선이 끊긴다.

양자화 재현:

```
llama-quantize --tensor-type blk.64=q8_0 RVN-BF16-mtp.gguf qwen38-heretic-Q3KM-mtp.gguf Q3_K_M
```

`--tensor-type blk.64=q8_0` 은 필수다. 원본의 MTP 헤드가 이미 q8_0 이라
그냥 돌리면 `requantizing from type q8_0 is disabled` 로 실패한다.

**측정 전 `sha256sum` 을 기록하고 보고서에 남긴다.** 플랫폼마다 다시 양자화할
경우 빌드 차이로 파일이 달라질 수 있다.

### 1-2. 빌드 — NCCL 이 없으면 측정이 무효다

```
cmake -S <소스> -B <소스>/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<아키텍처> -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NCCL=ON
```

- `CMAKE_CUDA_ARCHITECTURES` 는 플랫폼마다 다름 (허용 변동, 3절).
  여러 카드를 번갈아 쓸 계획이면 `61;70` 처럼 함께 넣는다 — **두 플랫폼을 같은
  바이너리로 재면 빌드 차이가 변수에서 빠진다**
- llama.cpp 커밋 해시를 보고서에 기록한다

> ## ⚠️ NCCL 라이브러리부터 설치할 것
>
> `GGML_CUDA_NCCL` 은 **기본값이 ON** 이다. 그런데 **NCCL 은 CUDA 에 포함돼
> 배포되지 않는다** [{ref 1.}](FINDINGS.ko.md#참고문헌). 직접 설치해야 한다.
> **플래그를 줬다는 사실만으로는 아무것도 보증되지 않는다** — cmake 는
> `Could NOT find NCCL` 경고만 내고 그대로 빌드한다.
>
> 라이브러리 없이 3장 이상으로 텐서 병렬을 띄우면 이렇게 된다.
>
> ```
> W NCCL not compiled in; falling back to internal AllReduce.
> W internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
> ```
>
> **내장 AllReduce 는 GPU 2장 전용이다.** 3장 이상이면 초기화에 실패하고 폴백을
> 탄다. 공식 안내서에도 이 제약은 적혀 있지 않다. **전부 그대로 돌아간다** —
> 텐서 경로만 3배 느린 채로.
>
> | 4장, Qwen3.6-27B Q6_K | NCCL 없음 | NCCL |
> |---|---|---|
> | tensor **pp 512** | 61.15 | **190.48** (+211%) |
> | tensor **tg 128** | 13.19 | **18.39** (+39%) |
> | layer (대조) | 변화 없음 | 변화 없음 |
>
> 폴백 상태에서 재면 *"텐서 병렬은 프리필을 팔아 디코드를 산다"* 는 결론이 나온다.
> **그건 측정 오류다.** 이 저장소의 첫 측정이 정확히 그 함정에 빠져
> **실험을 중단하고 처음부터 다시 측정했다.**
>
> *(위 llama-bench 수치는 동결 방법론 밖의 보조 측정이다. `pp 512` 는 512토큰
> 프롬프트 처리, `tg 128` 은 128토큰 생성을 뜻한다.)*

설치는 **CUDA 버전에 맞춰야 한다.** 기본 후보 패키지가 더 최신 CUDA 를 겨냥할 수
있다.

```
apt-get install -y libnccl2=<버전>+cuda<사용중인CUDA> libnccl-dev=<버전>+cuda<사용중인CUDA>
```

빌드 시점에 `Found NCCL` 이 찍히는지 확인하되, **최종 판정은 4절의 런타임
체크리스트에서 한다.**

### 1-3. 프롬프트 픽스처

이광수 「무정」(1917, **퍼블릭 도메인**)을 한국어 위키문헌에서 받아 조립한다
[{ref 12.}](#참고문헌). 짧은 `user` 턴과 긴 `assistant` 턴이 번갈아 나오고,
웜 턴마다 앞선 턴이 누적되는 **실사용 대화 구조**를 모사한다.

**텍스트는 저장소에 넣지 않는다.** `harness/make_fixture.py` 가
`harness/fixture.lock.json` 의 **리비전 ID 를 고정해** 받아오므로, 누가 언제
돌려도 바이트 단위로 같은 픽스처가 나온다.

**목표 토큰 수** (`/tokenize` 실측 기준, 허용 오차 ±1%):

| 프롬프트 | 토큰 |
|---|---|
| 콜드 | **23,079** |
| 웜 1 | 23,177 |
| 웜 2 | 23,890 |
| 웜 3 | 24,688 |
| 웜 4 | 25,470 |

이 숫자는 실제 대화 로그(비공개)에서 유래했다. 픽스처를 이 값에 맞추면
**텍스트 성격만 다른 통제된 대조**가 성립한다.

⚠️ **픽스처가 한국어이고 1917년 표기법이다.** 자당 토큰 수와 예측 가능성이
언어·시대에 따라 다르므로 **영어 벤치마크와 직접 비교하면 안 된다.** 특히 MTP
수락률이 픽스처에 민감하다 — P104 보고서 부록 A 에 같은 셀을 다른 픽스처로 돌린
결과가 있다.

### 1-4. 서버 인자

```
-ngl 99 -c 27000 -np 1 --split-mode <layer|tensor> [--spec-type draft-mtp]
```

- **`-c 27000`** — 8GB 카드 2장에서 NCCL P2P 버퍼까지 확보되는 상한.
  **여유가 있는 플랫폼에서도 늘리지 않는다.** 늘리면 KV 크기가 달라져 비교가 깨진다
- **`-np 1`** — 슬롯 1개. 다중 슬롯은 SSM 상태가 시퀀스마다 복제돼 메모리가 달라진다
- `-ts` 는 **주지 않는다.** 기본 균등 분할. 플랫폼별 튜닝은 비교를 깨뜨린다
- `--chat-template-kwargs` 등 템플릿 관련 인자는 주지 않는다.
  측정은 `/completion` 을 쓰므로 채팅 템플릿을 거치지 않는다

### 1-5. 요청 파라미터

```json
{"prompt": "...", "n_predict": 400, "ignore_eos": true, "cache_prompt": true, "seed": 1234, "stream": false}
```

- **`ignore_eos: true` + `n_predict 400`** — 12셀의 생성 작업량을 동일하게 만든다.
  ⚠️ 대신 모델이 끝내려는 지점을 넘어가므로 **MTP 수락률이 실사용보다 낮게 나온다.**
  이 한계는 보고서에 반드시 명시한다
- **샘플링 파라미터는 보내지 않는다.** GGUF 메타데이터의 기본값이 적용된다
  (`temp 1.0 / top_k 20 / top_p 0.95`). 모델이 고정이므로 결정적이다
- `cache_prompt: true` — 웜 턴의 프롬프트 캐시 적중이 측정 대상이다

### 1-6. 셀 정의

`{layer, tensor} × {MTP off, on} × {2, 3, 4장}` = **12셀**

- GPU 선택은 `CUDA_VISIBLE_DEVICES=0,1,...` 로 앞에서부터 N장
- 셀 이름은 `<split>-<nomtp|mtp>-<N>` (예: `tensor-nomtp-4`)
- 실행 순서는 4장 → 3장 → 2장. 실패하기 쉬운 조합을 뒤에 둬서 앞쪽 결과를 지킨다

**N=1 은 모델이 한 장에 들어가는 플랫폼에서 반드시 추가 측정한다.** 모델
13,115 MiB 는 16GB 카드에 들어간다 — V100 16GB 에서 실제로 측정됐다.
**이 셀이 있으면 파생 지표의 `tg_1` 이 환산값이 아니라 실측이 된다**(5절).
핵심 비교군은 어디까지나 2·3·4장이며, N=1 은 기준선 용도다.

### 1-7. 측정 소스

**서버 로그를 파싱하지 않는다.** `/completion` 응답의 `timings` 객체에서 취한다.

`prompt_n` / `prompt_ms` / `prompt_per_second` / `cache_n` /
`predicted_n` / `predicted_ms` / `predicted_per_second` /
`draft_n` / `draft_n_accepted`

벽시계 시간(`wall_s`)도 함께 기록한다.

### 1-8. 텔레메트리

셀마다 병행 기록. 1초 간격. **v1.2 에서 클럭과 스로틀 사유가 추가됐다.**

```
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active --format=csv,noheader -l 1
```

**사용률 합계가 100%를 넘는지**가 병렬화 성공의 1차 지표다.

> **`clocks_throttle_reasons.active` 는 비트마스크이고 `0x1` 은 GpuIdle 이다 —
> 스로틀링이 아니다.** 노는 카드가 전부 이 비트를 켜므로 "0이 아니면 스로틀"로
> 세면 크게 과대 보고된다. 특히 **레이어 분산은 한 시점에 한 장만 계산하므로
> 카드가 많을수록 유휴 샘플이 많아진다** — 그것을 스로틀로 세면 "카드를 더 쓸수록
> 스로틀이 줄어든다"는 무의미한 패턴이 나온다.
>
> 실제로 클럭을 누르는 비트는 이것들이다.
>
> ```
> therm = 0x08 HwSlowdown | 0x20 SwThermal | 0x40 HwThermal
> pcap  = 0x04 SwPowerCap | 0x80 HwPowerBrake
> ```
>
> **열과 전력을 분리해 집계해야 원인이 구분된다.** 집계가 옳은지는 원시 CSV 의
> 비트마스크 분포를 직접 확인하면 된다.
>
> ```
> awk -F, '{gsub(/ /,"",$8); c[$8]++} END {for (k in c) print k, c[k]}' <셀>.smi.csv
> ```

---

## 2. 재현 절차

```bash
# 0) NCCL 부터 설치한다. CUDA 에 포함돼 있지 않다 (1-2절).
apt-get install -y libnccl2=<버전>+cuda<사용중인CUDA> libnccl-dev=<버전>+cuda<사용중인CUDA>

# 1) 빌드. cmake 가 "Found NCCL" 을 찍는지 확인할 것
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<아키텍처> -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NCCL=ON
cmake --build llama.cpp/build --config Release -j 8

# 2) 모델 양자화 (원본은 HuggingFace, 1-1절)
llama-quantize --tensor-type blk.64=q8_0 RVN-BF16-mtp.gguf qwen38-heretic-Q3KM-mtp.gguf Q3_K_M
sha256sum qwen38-heretic-Q3KM-mtp.gguf        # 보고서에 기록

# 3) 픽스처 생성 — 위키문헌에서 고정 리비전을 받아 조립
python3 harness/make_fixture.py --pin          # 최초 1회
python3 harness/make_fixture.py --port 8099 --out fixtures/mujeong.json

# 4) 토큰 수 확인 (건너뛰지 말 것 — 초과하면 조용히 잘린다)
python3 harness/server_bench.py --check

# 5) 수동 프로브: 가장 무거운 셀(N=2, tensor, MTP on)을 띄워 로드 확인
# 6) 측정
python3 harness/server_bench.py

# 7) 결과 검증 (4절 체크리스트)
```

5번을 건너뛰지 않는다. 가장 빡빡한 셀이 안 뜨면 12셀 전체를 다시 돌려야 한다.

하네스는 중단 후 재실행 시 **완료된 셀을 건너뛴다.** 에러로 끝난 셀은 재시도한다.

---

## 3. 플랫폼마다 달라지는 것 (허용 변동)

| 항목 | 비고 |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | 카드에 맞춘다 |
| 드라이버 / CUDA Toolkit 버전 | 보고서에 기록 |
| NCCL 버전 | 기록. **없으면 측정 자체가 무효** |
| 사용 가능 GPU 수 | 4장 미만이면 해당 셀만 비움 |
| **전력 상한 (W)** | **카드 사양이므로 환경 절에 기록한다.** 측정 중 조정했다면 그 사실과 값을 명시한다 |
| CPU / RAM / PCIe 세대 / 토폴로지 | 보고서 환경 절에 기록 |

**이 목록에 없는 것을 바꾸면 v1.2 측정이 아니다.**

---

## 4. 결과 검증 체크리스트

측정 후 반드시 확인한다. 하나라도 어긋나면 그 셀은 무효다.

- [ ] 기동 로그에 `NCCL not compiled in` 이 **없다**
- [ ] 기동 로그에 `internal AllReduce init failed` 이 **없다**
- [ ] 모든 셀에서 `prompt_n` 이 1-3 의 목표값과 일치한다
- [ ] 콜드 턴의 `cache_n` 이 0 이다
- [ ] 웜 턴의 `cache_n` 이 22,000 이상이다 (캐시 적중)
- [ ] 모든 턴의 `predicted_n` 이 400 이다
- [ ] **MTP off 셀**의 웜 4턴 `predicted_per_second` 편차가 5% 이내다
- [ ] **열·전력 스로틀 비율을 집계해 보고서에 기록했다** (1-8절 기준)
- [ ] **측정 중 커널 로그에 Xid 가 없다** (`dmesg -T | grep -i xid`)

⚠️ **MTP on 셀에는 5% 기준을 적용하지 않는다.** 수락률이 생성 내용에 따라
크게 출렁이는 것은 투기적 디코딩의 내재적 성질이지 측정 오류가 아니다
(P104-100 측정에서 최대 67%). 대신 **편차를 기록해 보고한다** — 평균이 같아도
응답 시간이 들쭉날쭉하면 체감 품질이 다르므로, 편차 자체가 보고할 지표다.

⚠️ **크래시가 난 뒤 이어서 돌린 셀은 오염될 수 있다.** V100 측정에서
`tensor-mtp-4` 가 Xid 79 로 죽은 직후 실행된 두 셀의 프리필이 절반으로 떨어졌고,
서버 로그에 NCCL 메시지가 남아 있었다. **재부팅 후 다시 재고, 오염된 원본은
버리지 말고 보존한다** — 오염 판정의 근거다.

---

## 5. 파생 지표 계산식

보고서마다 같은 식을 쓴다.

**웜 턴 소요** = 웜 4턴 `wall_s` 의 평균

**세션 소요** = `콜드 wall_s + 웜턴소요 × N`

**세션 손익분기 턴 수** — 구성 A 가 B 를 이기기 시작하는 지점

```
N = (콜드_A - 콜드_B) / (웜_B - 웜_A)
```

**텐서 스케일링 효율**

```
효율(N) = (tg_tensor(N) / tg_1) / N
```

`tg_1` 은 **실측값을 쓴다**(1-6절). 모델이 한 장에 들어가지 않는 플랫폼에서는
**레이어 분산의 tg** 를 대용하고 **환산값임을 명시**한다 — 레이어 분산은 한 시점에
한 장만 계산하므로 그 값이 곧 1장 속도라는 전제다. **V100 에서 이 전제가 1% 안에서
맞음이 확인됐다**(1장 실측 27.71, 레이어 2·3·4장 27.97 / 27.78 / 27.29).

**MTP 배치 비용 배수** — 투기적 디코딩의 손익을 가르는 값

```
스텝 수      = n_predict - draft_n_accepted
스텝당 비용  = predicted_ms(MTP) / 스텝 수
토큰당 비용  = predicted_ms(no MTP) / n_predict
비용 배수    = 스텝당 비용 / 토큰당 비용
스텝당 산출  = n_predict / 스텝 수
```

**산출 < 비용 배수이면 MTP 는 손해다.**

**손익분기 수락률**

```
드래프트 길이 d = draft_n / 스텝 수
1 + d × acc = 비용 배수  ->  acc = (비용 배수 - 1) / d
```

**루프라인 실효 대역폭**

```
실효 대역폭 = (모델 크기 / N) / 카드가 실제로 쓰는 시간
```

레이어 분산에서는 카드가 전체 시간의 1/N 동안만 활성이므로 그 시간을 쓴다.
사양 대역폭 대비 비율이 낮으면 **대역폭 바운드가 아니다.**

---

## 6. 알려진 함정

| 증상 | 원인·대응 |
|---|---|
| `-sm row` 가 `does not support split buffers` | CUDA 백엔드에서 제거된 경로. **`-sm tensor` 를 쓸 것.** 같은 변경이 sd.cpp 의 row 분할도 끊었다 — 소스 수준 경위는 [`image-model-split-bench`](https://github.com/pyys/image-model-split-bench) 의 METHOD 에 있다 |
| 텐서 병렬 수치가 이상하게 낮음 | NCCL 미빌드. 4절 체크리스트로 걸러진다 |
| NCCL `Cuda failure 2 'out of memory'` (`transport/p2p.cc`) | 모델은 올라갔으나 **NCCL P2P 버퍼용 VRAM 부족.** 토폴로지 무관. `NCCL_P2P_DISABLE=1` 로 우회하면 호스트 경유가 되어 **비교가 깨진다.** 컨텍스트를 줄이는 것이 정답이지만 `-c 27000` 이 고정이므로 해당 셀을 실패로 기록한다 |
| `layer` + MTP + 소수 GPU 에서 연산 버퍼 OOM | MTP 드래프트 컨텍스트가 얹혀 안 들어감. 실패로 기록 |
| **`Xid 79 — GPU has fallen off the bus`** | **`tensor` + MTP + 4장에서 재현됐다**(V100, 2회). 해당 셀을 실패로 기록하고 **재시도하지 않는다.** 이어지는 셀의 오염은 4절 참조 |
| `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` | 경고일 뿐. `-ngl`·`-c` 를 직접 주므로 무관 |
| `backend sampling not supported with SPLIT_MODE_TENSOR; using CPU` | 텐서 병렬 내재 특성. 통제 변수로 둔다 |
| 프롬프트가 `n_ctx` 를 넘음 | llama.cpp 가 **조용히 잘라내** 측정이 오염된다. `--check` 를 건너뛰지 말 것 |

### 6-2. ⚠️ 텐서 병렬은 생성 결과 자체를 바꾼다

같은 seed·같은 프롬프트라도 **분할 방식과 GPU 수가 바뀌면 텐서 병렬에서는
출력 토큰이 달라진다.** all-reduce 의 연산 순서가 달라 부동소수점 결과가
미세하게 어긋나기 때문이다.

두 플랫폼에서 확인됐다.

| | 레이어 분산 | 텐서 병렬 |
|---|---|---|
| P104 `draft_n` | 4장·3장 **완전히 동일** | 675 / 642 / 660 … **전부 다름** |
| V100 `draft_n` / `accepted` | 4·3·2장 모두 **2440 / 780** | 2547/744, 2599/723 **제각각** |

따라서 **텐서 병렬 셀 사이의 MTP 수락률 비교는 생성 내용이 통제되지 않는다.**
수락률 차이를 GPU 수의 효과로 해석하지 말 것. MTP off 지표(pp·tg)는
생성 내용과 무관하므로 이 문제의 영향을 받지 않는다.

---

## 7. 보고할 수 없는 것

- **합성 벤치(`llama-bench` 의 `pp512` 등)를 실사용 프리필의 대표값으로 쓰지 않는다.**
  배치 크기가 작아 분할 방식의 우열이 뒤집힌다. 참고 수치로만 쓰고 그 사실을 명시한다
- **다른 방법론 버전으로 측정된 값을 같은 표에 넣지 않는다**

---

## 8. 버전 이력

| 버전 | 일자 | 내용 |
|---|---|---|
| v1.0 | 2026-09-10 | 최초 동결. P104-100 ×4 측정 기준 |
| v1.1 | 2026-09-10 | 4절 편차 검사를 **MTP off 셀에만** 적용(MTP 는 편차가 내재적). **6-2절 신설** — 텐서 병렬은 N 이 바뀌면 출력이 달라진다 |
| **v1.2** | **2026-09-15** | **1-8 텔레메트리에 `clocks.current.sm`·`clocks_throttle_reasons.active` 추가**(v1.1 하네스로는 스로틀 집계가 불가능했다) · **1-2 에 NCCL 경고 흡수** · **1-6 N=1 을 권고에서 요건으로** · 3절에 전력 상한(카드 사양) · 4절에 스로틀·Xid·크래시 오염 · **6절에 Xid 79** · 2절 재현 절차 신설 |

**v1.0 · v1.1 · v1.2 는 측정 조건이 같다.** 기록 항목과 검증·해석 규칙만 늘었으므로
**이전 버전으로 측정된 결과를 그대로 같은 표에 놓을 수 있다.** 다만 v1.1 이전
측정에는 스로틀 기록이 없다.

---

## 참고문헌

{ref 1.}~{ref 10.} 은 [`FINDINGS.ko.md`](FINDINGS.ko.md#참고문헌) 에 있다.

{ref 11.} *0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF*.
https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
(접근 2026-09-10)

{ref 12.} *이광수, 「무정」(1917)* — 한국어 위키문헌. 퍼블릭 도메인.
https://ko.wikisource.org/wiki/무정 (접근 2026-09-10)
