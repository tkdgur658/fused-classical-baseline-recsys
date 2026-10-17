# PREREG_R2NULL3 — PREREG_R2NULL2와 동일한 귀무 3종(N2·N3·W), 단 learned ranker L = MES (측정 전 고정)

작성 2026-10-15 21시대, PREREG-R2NULL3. 상태: **초안 = 동결 대기. 실행은 MES9 `QUEUE_DONE` 이후에만, 저자 승인 뒤 발사**(§6). 스크립트 sha256은 발사 시 기입. CPU ≤ 60분, GPU 없음, 서버에서는 읽기만 했다(`ls`/`cat`/`head`, 이 문서 작성 중 아무것도 실행하지 않음). R2NULL2 결과 파일(`results/r2null2.json|md`)은 이 문서를 쓰는 동안 열어 보지 않았다.

**효력**: 아래에서 "변경"으로 적은 항목을 제외하고 `PREREG_R2NULL2.md`(본문 + A1)의 모든 규칙·정의·상수를 그대로 가져온다. 충돌하면 이 문서의 "변경"이 우선한다.

## 0. 왜 R2NULL3인가
- 원고의 사실 N′(헤드룸이 잡음 수준)의 learned ranker는 MES가 아니라 **XRE**(= 이전 풀 설계 E + R + A, λ 0.1, K 40, 서버 설정 `<ds>_xres_xre_l010_k40`)로 측정되었다. 근거: `R2/results/r2s1.json`의 `per_ds.<ds>.Ltag = "xre"`(9 ds 모두), `R2/r2s1.py`의 `TAGS`, `study/results/r2null2.json`의 `Ltag`. 증거 정리는 `<REPO>/NAME_CHECK.md` §(2).
- MES = XRE에서 retention과 앵커항 A1을 뺀 entry-only 설계(패키지 `xres0`, 설정 `<ds>_xres_xeoa0_*`; `nc_parts: entry`, `nc_anchor: drop`). 본 논문의 제안법 이름이며 Table 1/2의 수치는 모두 MES다. 따라서 N′ 사실이 "MES로 학습한 ranker"에도 성립하는지는 아직 측정되지 않았다.
- R2NULL3 = 같은 프로토콜을 L = MES로 다시 측정한다. XRE 기반 R2NULL2 결과는 폐기하지 않고 그대로 남긴다.

## 1. 변경 요약 (R2NULL2 대비)
| 항목 | R2NULL2 | R2NULL3 |
|---|---|---|
| L | `r2s1.json`의 `Ltag`(= xre, 9 ds) | **`mes`**(MSCA + MES) 고정. 데이터셋별 native/MES 선택 없음(L은 항상 MES) |
| L 덤프 위치 | `ROOT/{dumps,W1/dumps,W2/dumps}/<ds>/xre__<ds>__s<s>[__valid].npz` | 기존 4 ds: `ROOT/dumps/<ds>/mes__<ds>__s<s>[__valid].npz` / 새 4 ds: `ROOT/study/MES9/dumps/<ds>/mes__<ds>__s<s>[__valid].npz` |
| 데이터셋 | 9 (microlens 포함) | **최대 8** (baby, pantry, scientific, sports, clothing, instruments, arts, office). microlens는 MES run 없음 → "—" |
| C | `r2s1.json`의 `Ctag` | **동일** (C 선택은 `alone_<c>` validation R@20으로 정해지며 L과 무관하다는 것을 `r2s1.py` 142행에서 확인) |
| 게이트 | R_L, R_C, R_O, g_O를 `r2s1.json`과 \|Δ\| ≤ 1e-6 | R_L이 달라지므로 §4의 새 게이트(R_C는 `r2s1.json`과 1e-6, R_L은 MES 덤프 sidecar json과 1e-4) |
| 전체 판정 임계 | 분모 9 | 분모 = 평가된 ds 수 n_eval(≤ 8), §3 |

**변하지 않는 것**: 규칙 `g_O ≤ hi(N2) ∧ g_O ≤ hi(N3) ∧ p_W > 0.0056`(α = 0.05/9 그대로, 8 ds로 줄어도 완화하지 않음), N2(B = 1000)·N3(B = 200)·W(순열 B = 1000), 7층(경계 [1],[2],[3],[4–5],[6–9],[10–19],[≥20]; 200명 미만 층 병합), 민감도(십분위·5분위, N2만 B = 200, 판정 제외), 난수 `SeedSequence([20261015 + 1000·ds번호 + seed, kind])` (ds번호는 **R2NULL2와 같은 `DSS` 목록 번호 0–8**, 즉 baby 0 … office 7, microlens 8은 비어 있을 뿐 번호를 당기지 않는다; kind N2 = 2, N3 = 3, W = 4, 민감도 20 + 분위수), N3 미끼 재추출 최대 30회와 `n3_unresolved_decoys_mean` 보고, 스레드 ≤ 8, `nice -n 15`, `CUDA_VISIBLE_DEVICES=`, `timeout 4200`, 스크립트 내부 60분 cap(ds 사이에서 확인), 해석 한계 문구, 진단 4종, 보고 형식, "수정 재실행은 새 번호, 시드로 판정을 고르지 않는다" 원칙.

## 2. 데이터와 정의
- 평가 사용자 = test 양성이 있는 사용자, 사용자 recall@20 r^L_u(L = MES), r^C_u(C = 기존 Ctag), r^O_u = max, g_O = mean_s R_O,s / mean_s R_L,s − 1 (seed 1–5 평균비). 모든 ds에서 **seed 1–5 test 덤프 5개가 전부 있어야** 평가한다(같은 seed 집합 규칙). 하나라도 없으면 그 ds는 "— (MES 미완)"이고 부분 seed로 대체하지 않는다.
- W는 같은 ds의 `__valid.npz` 5개가 전부 있을 때만 계산한다. 없으면 W "—", 판정은 N2·N3만으로 하고 "W 미평가"를 표기한다(R2NULL2 A1.1과 동일). valid 사용자 부분집합 정렬(R2NULL2 A1.4 수정)과 W 로더 오류는 FAIL로 전파하는 규칙도 그대로.
- **MES 선택의 비대칭(사전 명시)**: 기존 4 ds의 MES는 seed 1 validation으로 고른 (λ,K)(Baby .3/40, Pantry .1/20, Scientific .2/40, Sports .1/80)이고, 새 4 ds는 미튜닝 (λ 0.1, K 40)이다(`PREREG_MES9` A1.4). 이 차이는 보고서와 표 각주에 그대로 쓰며, 결과를 본 뒤 (λ,K)를 바꿔 재실행하지 않는다.

## 3. 판정 (R2NULL2 §4 동결 규칙, 분모만 일반화)
- 데이터셋 단위 "headroom = noise" ⇔ `[g_O ≤ hi(N2)] ∧ [g_O ≤ hi(N3)] ∧ [p_W > 0.0056]`. (W "—"인 ds는 N2·N3만.)
- 전체: N_noise = 위를 만족하는 ds 수, n_eval = 게이트 통과·평가된 ds 수. **N_noise = n_eval → "headroom = noise, n_eval/n_eval 유지"**; **N_noise ≥ n_eval − 2 → "대부분 유지(예외 ds 명시)"**; **N_noise ≤ n_eval − 3 → "유지되지 않음 — 구조적 여유 신호 있음"**. (n_eval = 9이면 R2NULL2의 9/9, ≥ 7, ≤ 6과 일치. 이 분모 일반화가 규칙의 유일한 변경이며 결과 전에 고정한다.) 게이트 실패·미완 ds는 N_noise에 넣지 않고 n_eval에서도 뺀다(실패 수를 따로 표기).
- 예상 n_eval: MES9 완료 상태에 따라 8 이하. office가 디스크·cap 사유로 skip되면 7.

## 4. 게이트 (변경)
1. **R_C 게이트**: 재계산한 R_C(seed별)가 `r2s1.json`의 `seeds[s].R_C`와 |Δ| ≤ 1e-6 (C·평가 사용자 집합이 R2NULL2와 같다는 확인).
2. **R_L 게이트**: 재계산한 R_L(seed별)이 같은 덤프의 sidecar json `recalc_r20`과 |Δ| ≤ 1e-4. 기존 4 ds는 추가로 `tables/mes_perseed.csv`(MSCA, MES 행) `r20`(소수 4자리)와 |Δ| ≤ 1e-4.
3. 평가 사용자 집합(`okp`)이 seed 5개에 걸쳐 동일하고 L 덤프의 users가 C 목록 users와 일치(R2NULL2와 동일 assert).
4. g_O에 대한 `r2s1.json` 일치 검사는 L이 다르므로 **하지 않는다**(`g_O_r2s1` 대신 XRE 기반 g_O를 참고 열로만 병기).
- 위반 → 그 ds는 `GATE_FAIL`, 판정 "—".

## 5. 덤프 경로 (2026-10-15 21:06 KST 서버 읽기 전용 확인; `ROOT=<RUN_ROOT>`)
**기존 4 ds (MSCA + MES 주표 run, 5 seed 모두 test·valid 쌍 존재 = 확인됨)**

| ds | 설정 (run_dir 확인) | test 덤프 (npz + json) | valid 덤프 |
|---|---|---|---|
| baby | `baby_xres_xeoa0_l030_k40` | `ROOT/dumps/baby/mes__baby__s{1..5}.npz` — 5/5 존재 | `…/mes__baby__s{1..5}__valid.npz` — 5/5 존재 |
| pantry | `pantry_xres_xeoa0_l010_k20` | `ROOT/dumps/pantry/mes__pantry__s{1..5}.npz` — 5/5 | `…__valid.npz` — 5/5 |
| scientific | `scientific_xres_xeoa0_l020_k40` | `ROOT/dumps/scientific/mes__scientific__s{1..5}.npz` — 5/5 | `…__valid.npz` — 5/5 |
| sports | `sports_xres_xeoa0_l010_k80` | `ROOT/dumps/sports/mes__sports__s{1..5}.npz` — 5/5 | `…__valid.npz` — 5/5 |

(같은 폴더의 `mes_c0base__*`는 같은 run_dir의 별칭이고 `mesc/mesf/mesm/mesn/mess__*`는 COHESION·FREEDOM·MMGCN·MENTOR·SMORE 백본용이라 **쓰지 않는다**. 사용 태그는 정확히 `mes`.)

**MES9 새 4 ds (설정 `<ds>_xres_xeoa0_l010_k40`, 미튜닝; `ROOT/study/MES9/dumps/<ds>/`)** — 21:06 시점

| ds | 기대 경로 | 상태 |
|---|---|---|
| instruments | `mes__instruments__s{1..5}.npz`, `…__valid.npz` | **확인됨** 5/5 test, 5/5 valid (markers DONE s1–s5, `BLOCK_instruments_DONE`) |
| clothing | `mes__clothing__s{1..5}.npz`, `…__valid.npz` | **부분**: s1–s3 test+valid 존재, s4·s5 학습 중(`RUNNING`) → 미완 |
| arts | `mes__arts__s{1..5}.npz`, `…__valid.npz` | **없음**(블록 미시작; 이후 `BLOCK_arts.ADMIT`) |
| office | `mes__office__s{1..5}.npz`, `…__valid.npz` | **없음**(블록 미시작; 디스크 60 GB 검사·cap 16.0 GPU-h 규칙으로 skip 가능, `BLOCK_office.*`/`SKIP_*` 마커) |

발사 시 재확인 규칙: `QUEUE_DONE` 존재, 각 ds의 `M9_<ds>_s{1..5}.DONE` 5개, 위 경로 10개 파일(5 test + 5 valid)을 확인한 뒤 그 ds만 평가 목록에 넣는다. `QUEUE_FAIL`이거나 어떤 ds의 마커가 `SKIP_*`/`FAIL`이면 그 ds는 "— (MES 미완)"으로 두고 나머지를 평가한다(재시도·부분 seed 대체 없음).

## 6. 실행 조건과 스크립트 변경 (최소 diff)
- **발사 조건**: (1) 서버 `<RUN_ROOT>/study/MES9/QUEUE_DONE` 생성, (2) 저자의 발사 승인(저자 결정 위임 범위 내), (3) `--selftest`, `--smoke`(baby seed 1, 이전과 같은 방식으로 수치 비출력) 통과. MES9가 끝나기 전에는 실행하지 않는다(GPU 서버 부하·덤프 미완 방지; R2NULL2의 CPU 잡이 MES9 GPU 학습과 공존했던 것과 달리 L 덤프 자체가 미완이므로).
- **재사용**: `study/R2NULL2/code/r2null2.py`(sha `2b3e11de1b4f6eaeab5f2d1b39e0713a2bafda5855dd486903de88483d1d3e50`)를 `study/R2NULL3/code/r2null3.py`로 복사하고 아래만 바꾼다. 다른 코드(N2/N3/W 루틴, 층화, 난수, 판정, selftest)는 한 줄도 바꾸지 않는다. `r2s1.py`는 계속 읽기 전용 import(L 로더 외의 `load_ds`, `classical_lists`, `user_recall`).
  1. `DSS` 중 평가 목록만 8 ds로 제한(번호 매기기는 원래 `DSS` 인덱스 유지).
  2. `Ltag = s1['Ltag']` → `Ltag = 'mes'` 고정; `Ctag`는 `s1['Ctag']` 그대로.
  3. `r2s1.find_dump(Ltag, ds, s)` 호출 두 곳(185, 192행) → 로컬 `find_dump_L(ds, s)`: 먼저 `ROOT/dumps/<ds>/mes__<ds>__s<s>.npz`, 없으면 `ROOT/study/MES9/dumps/<ds>/mes__<ds>__s<s>.npz`, 둘 다 없으면 `FileNotFoundError`(삼키지 않고 그 ds를 "— (MES 미완)"으로 기록). `find_valid`(174행)에도 같은 두 위치를 추가.
  4. 게이트(196–203행): §4로 교체(R_C는 `r2s1.json`, R_L은 sidecar `recalc_r20`·`mes_perseed.csv`; `g_O` 일치 검사 삭제, 대신 `g_O_xre = s1['g_O']`를 참고 필드로 저장).
  5. `write_out`의 결론 문구에서 "9"를 `n_eval`로 일반화(§3).
  6. 출력 폴더 `<RUN_ROOT>/study/R2NULL3/{code,results,logs}`, 마커 `<RUN_ROOT>/study/R2NULL3/DONE|FAIL`. 래퍼 `r2null3_run.sh` = `r2null2_run.sh`에서 경로·로그 이름만 `R2NULL3`로 치환.
- 스크립트 sha256 (발사 시 기입):

| 파일 | 경로 | sha256 |
|---|---|---|
| `r2null3.py` | 서버 `<RUN_ROOT>/study/R2NULL3/code/` | `106df8e3ad91a4265398544d9e626f17e67d462a098eff7a435983a8bf122f6a` (2026-10-16 00:50 발사 직전; 최초본 a3237fe5…는 구문 오류·9a41f13e…는 find_dump_L 오타로 미실행, 규칙 변경 없음) |
| `r2null3_run.sh` | 〃 | `b3f443e66bc2cd738f6a21ad1762a05e012090aa3351c857a9bef61207700811` |
| `mes_perseed_msca_mes.csv` (게이트용, 로컬 tables/mes_perseed.csv의 MSCA,MES 20행 복사) | `code/` | `2c37443df083487970613ff46ac16244ba065e6670b500eb2e92a0ae7fc34fde` |
| `r2null2.py` (기반) | `<RUN_ROOT>/study/R2NULL2/code/` | `2b3e11de1b4f6eaeab5f2d1b39e0713a2bafda5855dd486903de88483d1d3e50` (21:06 확인) |
| `r2s1.py` (import) | `<RUN_ROOT>/R2/` | `1bd74ca9aa554b10aa0f17baaa7fed1e6a5fcadb5d9cf2e70bd97b6545d0891e` (R2NULL2 기재값) |
| `results/r2s1.json` (C·R_C 게이트) | `<RUN_ROOT>/R2/results/` | `a0a48d97c1d041611245f22f3a817643529de35a263797e437652e1b10d5f836` |
| 합성 검증 | 새로 쓰지 않음: `study/synth_r2null2.py`(sha `cc6de5f0…`, 출력 `synth_r2null2.out`)가 규칙 전체를 이미 검증했고 판정 규칙·B·난수가 불변이므로 재사용 | — |

- 비용: R2NULL2와 같은 자릿수(8 ds, 추정 15–25분), cap 60분, CPU 1 프로세스. 실패 처리와 새 번호 재실행 원칙은 R2NULL2 §5와 동일(다음은 R2NULL4).

## 7. 사전 설정 보고 계획 (결과 전 고정)
- **원고 4f (`4_ex/4f_cold_neardup.tex`)**: XRE 기반 결과(표 tab:headroom, g_O 14.9–36.4%, 현재 N′ 문구)는 **그대로 둔다**. 추가는 두 가지뿐이다.
  1. tab:headroom에 열 하나: "g_O (MES)" — R2NULL3이 평가한 ds의 MES 기준 g_O, 평가 안 된 ds(microlens, 미완·skip된 ds)는 "—". 캡션에 "MES on Baby, Pantry, Scientific, and Sports uses validation-selected λ and K; the other datasets use λ 0.1 and K 40 untuned"를 한 문장으로 쓴다(숫자 0–10은 단어로).
  2. 본문에 **"MES-based" 문장 하나**: "With MES as the learned ranker on [n_eval] datasets, the per-user oracle raises Recall@20 by [a] to [b]%, and the [규칙 결론: 유지 / 대부분 유지(예외 ds) / 유지되지 않음]." 결과가 어느 쪽이든 이 문장은 넣는다.
- 결론이 XRE 결과와 다르면(예: MES에서 "유지되지 않음") 두 결과를 함께 쓰고 어느 쪽도 지우지 않는다. 그 경우 N′ 문구를 ds 범위로 제한할지는 저자 결정으로 올린다(R2NULL2 §4의 "삭제 또는 해당 ds 범위로 제한" 규칙은 해당 ranker 기준으로 각각 적용).
- 보고서(`results/r2null3.md`)는 R2NULL2 §8 형식, 맨 위에 N_noise/n_eval, §3 결론 문구 한 줄, 진단 4종, 그리고 **XRE 대 MES 병기 열**(`g_O_xre`)과 "MES 선택 비대칭"(§2) 한 줄을 넣고 해석 한계를 생략하지 않는다.
- 본문의 방법명 표현은 NAME_CHECK 교정문 방침을 따른다: XRE는 "the earlier full design E + R + A", MES는 제안법.
