# PREREG_MES10 — 보고된 MES 설정에서 마스크 뷰 메커니즘 측정 (실험 ⑤, M/F/S 3팔)

작성 2026-10-16 (서버 시계 18–19시대), MES10 작업자. 근거: 저자 결정(저자가 thesis A "method paper" 선택 후; 저자 결정으로 취급). 상태: **동결** — 이 파일의 sha256을 `LAUNCH_LOG.md`와 서버 `study/MES10/prereg/PREREG_MES10.sha256`에 기록한 뒤 발사. 이후 변경은 번호 붙은 AMENDMENT로만, 결과 전/후 표시.
규칙: 내부 규칙(seed당 1 run, 반복·best-of 금지), 내부 규칙(서버 스크립트+마커, 자동 폴링 없음), 내부 규칙(발사 직후 1회 확인), 원고 수정 없음, <DISK>·/tmp 쓰기 없음, 삭제 없음(mv만), pkill/pgrep -f 금지.

## 1. 목적 (가설 없음)
보고된 MES 설정(MSCA, MES = xres0 xeoa0 entry-only, ds별 validation 선택 λ/K: Baby λ.3 K40, Pantry .1/20, Scientific .2/40, Sports .1/80)에서 마스크 뷰가 하는 일을 효과 크기와 유의 개수로 잰다.
- **M−F**: 같은 엔트리 쌍을 마스크 그래프(M) vs 전체 그래프(F)에서 채점. 10 seed.
- **S−F**: 전체 그래프 점수에서 직접 자기 합산항(self summand)을 대수적으로 뺀 것(S) vs F. 5 seed.
- **S−M**: 대수적 제거(S) vs 실제 간선 제거(M). 5 seed.
- **귀결(저자, 사전 고정)**: M−F의 R@20 차이가 양수이고 Holm 유의인 ds가 **4개 중 2개 이상이면 masked-view 주장 부활, 아니면 최종 삭제.** S−F, S−M은 메커니즘 서술용(귀결 규칙 없음).

## 2. 실행 목록 (80 run, seed당 1 run, 복제 없음)
| 팔 | 패키지 | config (ds별) | seed | run 수 |
|---|---|---|---|---|
| M (마스크 뷰 = 보고된 MES) | `xres0` | `<ds>_xres_xeoa0_<sel>` (보고 run과 같은 파일) | 6–10 (1–5는 `tables/mes_perseed.csv` MSCA/MES 행을 그대로 사용, 재실행 안 함) | 20 |
| F (전체 뷰, 튜닝된 λ/K) | `xres0f` | `<ds>_xres_xeoa0f_<sel>` (신규) | 1–10 | 40 |
| S (대수적 자기항 제거) | `xres0s` (신규) | `<ds>_xres_xeoa0s_<sel>` (신규) | 1–5 | 20 |

`<sel>` = baby `l030_k40`, pantry `l010_k20`, scientific `l020_k40`, sports `l010_k80`. 기존 MES_fv run(λ.1 K40)은 재사용하지 않음.
순서(`jobs_mes10.tsv`): 라운드 r=1..5, 각 라운드에서 ds 순서 (sports, pantry, baby, scientific), ds 안에서 F s_r, S s_r, M s_{r+5}, F s_{r+5}. (cap·GPU0 예약으로 중단돼도 세 팔이 균형 있게 진행되도록.)
각 run: `main.py --models <pkg> --configs <cfg> --datasets <ds>@<copy> --seeds <s> --stage train --execute --gpu <g> --allow-busy-gpu --data-copy <copy> --test-eval best_primed --save-ckpt --runs-dir study/MES10/runs/<id>` (cwd `study/MES10/wt`, nice 15). 덤프 없음(판정은 result.json `best_test`).

## 3. 코드·데이터 (보고된 MES run과 동일함을 확인)
- 스냅샷 `study/MES10/wt` = `study/MES9/wt` 복사(.git·__pycache__ 제외) + `models/xres0/run.py`, `models/xres0/manifest.json`을 P04_novctrl_wt@dba2ba3(보고 run의 worktree, clean) 바이트로 복원(MES9가 바꾼 것은 등록 줄 1개뿐). 복원 후 core·configs·main.py·xres0·xres0f·xres·r12·r9·r8·r7·r6·objx_s·objx_m·msca 파일이 P04_novctrl_wt와 전부 동일(차이 0, `mes10_setup.py` 단언).
- **지문 검증(`fp_check.py`, CPU)**: MES10/wt의 `main.cell_fingerprint`가 보고 run의 run_id 지문을 그대로 재현: M pantry `90054d0b7a`, scientific `f8a16fda4c`, sports `9ffac46833`, baby(canonical) `aa540a46dc`; xres0f(원 manifest) MES_fv pantry `cce33785d3`, scientific `e8bff8ea29`, sports `da23e7422c`, baby `88b20e6696` — 8/8 OK. 즉 M seed 6–10은 보고 seed 1–5와 같은 코드·config·core·데이터 manifest·평가기(best_primed)로 돈다. `verify_manifest` xres0/xres0f/xres0s OK.
- **데이터 사본 (지시와 다른 점, 사유 기록)**: 지시는 canonical_v2였으나 보고된 **Baby** MES seed 1–5는 `canonical` 사본(→ `<HOME>/MMRec/data/baby` 심볼릭)에서 돌았다. 두 사본의 baby.inter·image_feat·text_feat는 바이트 동일하지만 kNN 캐시가 다르다(image_adj_10_True: 70,500 간선 중 1,365개 다름, 838행; text_adj: 14개 다름; `tests/adj_check.out`). M 팔의 seed 1–10을 한 데이터 설정으로 유지하고 Baby의 M/F/S를 같은 데이터에서 짝짓기 위해 **Baby는 세 팔 모두 `canonical`, 나머지 3 ds는 `canonical_v2`**(보고 run과 같은 사본)를 쓴다. 두 사본 모두 <DISK> 읽기 전용(캐시 존재, 새로 쓰지 않음).
- F config = 보고 MES config + 기존 xres0f 두 줄(주석 + `nc_view: full`). 같은 생성 함수가 커밋된 4개 `xeoa0f_l010_k40` config를 바이트 단위로 재현함을 확인. F의 정의 = 기존 MES_fv: 엔트리 쌍을 전체 그래프 점수로 고르고 채점, 배치마다 마스크 뷰의 계수 질량에 맞춰 스케일(마스크 뷰는 질량 계산용 no-grad forward), A1 미가산.
- S config = F config + 주석 + `ss_mode: self`.

## 4. Arm S 정의 (측정 전 고정)
Proposition prop:history(e_u = Σ_{i∈H_u} R̃(u,i) m(u,i))의 직접 자기 합산항:
S(u,h) = R̃(u,h) · m(u,h)ᵀ e_h,  m(u,h) = ḡ_h + κ Σ_ν (1 − π^ν_u) h^ν_h
- R̃(u,h) = d_u^{-1/2} d_h^{-1/2} (전체 그래프 정규화), ḡ_h = (E_h + (ÂE)_h)/2 (MSCA n_layers 2에서 collab_u = Σ_i R̃ ḡ_i), h^ν_h = item–item 뷰 ν∈{image, text, struct}의 아이템 행, π^ν_u = 사용자 u의 뷰 주의 가중치(MSCA.forward의 query_common softmax), κ = fusion_coeff, e_h = 전체 그래프 최종 아이템 임베딩.
- **채널: 협업 부분 S_C = R̃ ḡ_hᵀ e_h 와 item-view 부분 S_V = R̃ κ Σ_ν(1−π^ν_u) h^ν_hᵀ e_h 를 모두 뺀다** (`ss_channels both`).
- 들어가는 곳 (전체 뷰 exchange 호출, 기울기 있는 호출 하나): (1) 목록 선택 — 배치 사용자의 fold-f 보류 양성 h 전부의 (detached) 전체 뷰 점수를 Sv − S(u,h)로 바꾼 뒤 L_u, D_u, C_u, 엔트리 행을 고름; (2) 엔트리 손실 — s_up → s_up − S(u,p) (두 항 모두로 기울기), s_uD는 그대로. 다른 아이템 점수, 사용자 임베딩, 정규화 계수는 바꾸지 않는다(= 직접 합산항만).
- 점수용 두 번째 forward 없음. F와 같은 단일 요인 대비를 위해 **F의 질량 맞춤(no-grad 마스크 forward로 얻은 스칼라 질량)은 그대로 유지**(own mass는 뺀 뒤 간격으로 계산). 따라서 S−F = "자기 합산항 빼기" 한 요인, S−M = "대수적 제거 vs 실제 간선 제거".
- 파라미터·버퍼 추가 0 (state_dict = MSCA). 코드 `models/xres0s/src/xres0s.py`(class XRES0S(XRES0F)).

### 4.1 CPU 점검 결과 (발사 전, 학습 아님)
- **(a) 끔 = F 비트 동일** (`tests/unit.out`, Baby l030_k40 seed 1 첫 배치): `ss_mode none`(F로 위임)과 `ss_mode zero`(빼기 코드 경로 전체 실행, S×0) 모두 F와 손실 비트 동일(13.691112518310547), 15개 파라미터 기울기 max abs diff 0.0, 같은 배치 → `BITWISE_OK true`. `ss_mode self`: 손실 13.69111443, 기울기 차 4.4e-6, 4스텝 손실·기울기 유한.
- **(b) tiny 그래프** (`tests/tiny.out`, 14 사용자 × 18 아이템, 79 간선 전부, d 6): 역사 분해 항등식 |e_u − Σ R̃ m| ≤ 1.2e-7, self_summand = R̃ m(u,p)ᵀe_p ≤ 1.2e-7. 간선 하나를 실제로 제거한 점수(norm_R·이분 인접 재구성, item–item 그래프 유지 = 마스크 뷰와 같음)와의 잔차 |s_rm − (s − S)| 평균 0.123 = 평균 |S|의 49%; 사용자 측(다른 계수의 차수 재정규화 + 주의 이동) 0.070, 아이템 측(e_p가 u의 메시지를 잃음) 0.181; corr(S, 정확한 LOO 하락) .80. (임베딩 ×3에서도 비율 49%, corr .82.) 작은 그래프는 차수가 작아 생략항이 크다.
- **(c) 실제 학습 체크포인트** (`tests/real.out`, 보고 MES Baby seed 1, 무작위 40 학습 간선): |s| 평균 2.281, S 평균 0.756 (S_C .518, S_V .237), 정확한 LOO 하락 |·| 0.744, 잔차 평균 0.096 (최대 0.439) = **평균 |S|의 12.8%**, 사용자 측 0.116, 아이템 측 0.103 (부분적으로 상쇄), corr(S, LOO 하락) **.884**. 즉 대수적 S는 실제 간선 제거 효과의 대부분을 담고, 생략항(재정규화·아이템 측·주의 이동)이 약 13%.
- **(d) GPU 스모크** (`tests/gpu.out`, GPU1, AMP fp16+GradScaler, Baby 30스텝, ≈1분): F·S 모두 손실 유한, peak 793/803 MiB, S 오버헤드 무시 가능.

## 5. 판정 (동결)
- 지표: test Recall@20 (result.json `best_test.recall@20`, mes_perseed.csv와 같은 값·같은 반올림), NDCG@20은 서술만.
- 검정: `tables/src/sigtest.py`(바이트 동일 사본 `code/sigtest.py`) 의미 — 같은 seed 짝 양측 paired t(`ttest_rel`, df = n−1), 95% CI = t_{.975,n−1}·sd/√n, 한 대비의 4 ds에 Holm, α .05, p_holm ≤ .05면 유의. M−F는 seed 1–10 (n=10), S−F·S−M은 seed 1–5 (n=5). M seed 1–5 = 보고 run.
- 한 대비에서 어느 팔이 그 대비의 seed를 하나라도 빠뜨린 ds는 "— (incomplete)"로 표기하고 유의 아님으로 센다; Holm m = 완전한 ds 수.
- 보고: 대비×ds별 평균 A·B, 차 ×1e-3, CI, t, p, p_holm, wins A/n, 유의 표시; 대비별 양(+)·음(−) 유의 개수.
- **귀결**: n(M−F R@20 차>0 ∧ p_holm ≤ .05) ≥ 2 → masked-view 주장 REVIVED, 아니면 DELETED. 음의 방향 유의는 부활 근거가 아님.
- 판정 스크립트 `code/mes10_verdict.py` (collect: 서버에서 `summary/mes10_perseed.csv`; verdict: `summary/mes10_verdict.{json,md}`), 큐 종료 시 자동 실행(CPU).

### 5.1 합성 검증 (`synth_mes10_verdict.py`, 로컬·서버 동일 출력 `SYNTH_MES10 OK`)
```
scenario A: M>F sig 4/4 (neg 0), incomplete [] ... -> REVIVED (expected REVIVED) OK
scenario B: M>F sig 0/4 (neg 0), incomplete [] ... -> DELETED (expected DELETED) OK
scenario C: M>F sig 1/3 (neg 0), incomplete ['sports'] ... -> DELETED (expected DELETED) OK
scenario D: M>F sig 0/4 (neg 3), incomplete [] ... -> DELETED (expected DELETED) OK
scenario E: M>F sig 2/4 (neg 0), incomplete [] ... -> REVIVED (expected REVIVED) OK
SYNTH_MES10 OK
```
(A 모든 ds +3e-3 → 부활; B 차 0 → 삭제; C Baby만 강함 + sports F seed 7 누락 → 삭제·incomplete 처리; D 반대 방향 유의 3개 → 삭제; E 정확히 2개 → 부활 경계.) 전문 `study/synth_mes10_verdict.out`.

## 6. 비용과 cap
- 단일 job 추정(보고 MES seed 1–5 평균 elapsed): baby .113, pantry .144, scientific .075, sports .462 h (S ×1.1). 80 run 단일 job 환산 ≈ **16.3 GPU-h**.
- 동시 실행: GPU당 3 slot(현재 두 GPU 유휴, MES run 1.3–2 GB·사용률 ~45%). cap 회계 = 각 job 벽시계 elapsed 합(동시 실행이면 점유를 과대 계산하는 보수적 정의). job 추정 = 2 × 단일 추정 → 합 32.6.
- **cap 55 GPU-h, 하드 가드**: claim 시 committed(끝난 job elapsed 합 + 실행 중 job est 합) + est ≥ 55 → `SKIP_CAP`. 예상 사용 ≈ 25–35.
- 예상 벽시계: 두 GPU 6 slot이면 ≈ 4–6 h; GPU0 예약으로 GPU1만 남으면 더 길어짐(≈ 8–10 h).

## 7. 큐·마커 (자동 폴링 없음)
- `nohup bash code/run_mes10.sh > logs/run_mes10.out 2>&1 &` 한 번. slot 6개(`mes10_lane.py <gpu> <slot>`, GPU1 ×3, GPU0 ×3), 공용 락 `state.lock`, 파일 순서로 동적 선점, nice 15, OMP 8.
- **GPU0 예약**: GPU0 slot은 claim 직전(락 안)과 대기 중에 `study/GPU0_RESERVED`를 확인하고, 있으면 새 job 없이 종료(실행 중 job은 끝까지). GPU1 slot은 영향 없음.
- 대기 조건: <DISK> 여유 ≥ 40 GB, 해당 GPU 여유 메모리 ≥ 5000 MiB.
- 마커 `study/MES10/markers/{M,F,S}/<job>.RUNNING(→.RUNNING.closed)|DONE|FAIL|SKIP_CAP`, 종료 `study/MES10/QUEUE_DONE|QUEUE_FAIL`(모든 slot rc 0이고 80 job 전부 DONE일 때만 DONE), 로그 `logs/mes10.log`, job 로그 `logs/<id>.log`, run `runs/<id>/`, 판정 `summary/mes10_verdict.{json,md}`, `logs/verdict.out`.

## 8. 실패 처리
- OOM → FAIL(사유 OOM), 재시도·설정 변경 없음. 그 외 rc≠0/발산 → FAIL, 재시도 없음. 결과가 있는 seed는 절대 재실행하지 않음(lane이 runs/<id>에 result.json이 있으면 실행 거부). FAIL·SKIP_CAP seed가 생긴 ds는 해당 대비에서 incomplete(§5).
- 코드 버그로 결과 없이 FAIL한 job을 고쳐 다시 돌려야 하면: 이 문서에 AMENDMENT로 원인·수정 sha를 적고, 결과 없는 job만 같은 seed로 1회 실행(결과 있는 seed는 그대로).

## 9. 파일 sha256 (서버 `study/MES10/`)
| 파일 | sha256 |
|---|---|
| `code/xres0s.py` = `wt/models/xres0s/src/xres0s.py` | `373ac1ca227fefd7e3ec1c97b53613a8fd777402f08c89bdbd61709397ab438e` |
| `code/mes10_setup.py` | `f4dd96b9a9ff0d39c001f59d670a6c745fcee318271df24559e8ac1fe6d3985d` |
| `code/fp_check.py` | `32cc1cf8657f0f676d01d607dad3bc9e49273dede80a05ea281877dde9ae71fa` |
| `code/adj_check.py` | `7dc3dc013d668fc4ad615d786e8888868738055dc94a6c0045054749c22156c0` |
| `code/mes10_tests.py` (unit·tiny·real는 gpu 모드 추가 전 판 `2896e24074eb…`로 실행; 두 판의 unit/tiny/real 코드는 동일) | `af61ddd73d25847d2a933aaa41b20377807589b25a8dd24688c578449a40e04d` |
| `code/mes10_gen_jobs.py` | `f0c1f8e5770a620d4c83f21d4980c7908f43f5d04d14b46e607867d31cbf3927` |
| `code/mes10_lane.py` | `a106ef21cdd82659a84ffa4a45fe7eb1d380e16a388ead79f6829bfce5cafb15` |
| `code/run_mes10.sh` | `0e647aaa37d0232b5fc541fc9aec296cd3ef2bd0194c4fdbd82ee1847659309d` |
| `code/mes10_verdict.py` | `4374adc9ecd93781db2403da0173f76056d0ba2e0b61d8fb88b42aa5fb8e2997` |
| `code/sigtest.py` (= 원고 `tables/src/sigtest.py`) | `a3621bb34739294cd9045588b74b2556604feab37f4b7f455921d34af73adb47` |
| `code/synth_mes10_verdict.py` (로컬 `study/`에 동일 사본) | `97b576cb7429a09e9cf81245bb64bf585970a91ba6d92b533813b0898127544c` |
| `prereg/mes_perseed.csv` (= 원고 `tables/mes_perseed.csv`) | `c458a2db0b6714c972a9990a9c18670fc4b5060737251819150fd0f8c4b812cb` |
| `budget.txt` (55.0) | `97fc8594e01fb3fa994062d4b72d1d9e31f8b1907917aeb9c4ee0d21eea98337` |
| `wt_manifest_orig.sha256` (복사+xres0 복원 직후, 1951 파일) | `057da3970413e0c45c21a07466f72e5b56b49dbe0aea263e8e61fc47e72b04a5` |
| `wt_manifest.sha256` (F config 4 + xres0s 패키지 추가 후, 1967 파일) | `0ea699adbc761e1bef2c09aacbab1093fba84007e71b116cc7c0192c7c1e5475` |
| 점검 출력 `tests/unit.out` / `tiny.out` / `real.out` / `gpu.out` / `adj_check.out` | `2dc2dc40…` / `b4f866b4…` / `407d2314…` / `a95b65e1…` / `8e8a1b80…` |

신규 run_id 지문(canonical_v2 기준 fp_check 출력; Baby는 canonical 사본이라 실제 지문은 실행 시 기록): xres0f pantry `84a45f7ceb`, scientific `90a5b7f399`, sports `f46f8b7151`; xres0s pantry `8da1a7c5ad`, scientific `a47eb30efd`, sports `02e7fea75d`. M baby(canonical)는 보고와 같은 `aa540a46dc`여야 한다.

## 10. 한계 (사전 명시)
- S는 직접 합산항만 뺀다: 차수 재정규화, e_p 쪽 경로, 주의 가중치 이동, 다른 보류 양성의 메시지가 사용자 임베딩에 남는 효과는 빼지 않는다(실제 체크포인트에서 생략항 ≈ |S|의 13%). 공기(co-occurrence) 구조 그래프는 M·S 모두 전체 학습 상호작용으로 만든 그대로(Lemma lem:masked의 범위와 같음).
- S의 질량 맞춤은 F와 같이 no-grad 마스크 forward를 쓴다(점수·기울기에는 쓰지 않음). "두 번째 forward가 전혀 없는" 변형은 질량 맞춤 요인이 섞이므로 이번 범위가 아니다.
- M seed 1–5는 10/09–10/10 run, 6–10은 이번 run: 같은 코드·지문이지만 GPU 비결정성(deterministic False)과 동시 실행 환경 차이는 남는다(F·S는 전부 이번 환경).
- n=5인 S 대비는 검정력이 낮다. 결과 후 seed 추가·λ/K 재튜닝 없음.

## AMENDMENT A1 (2026-10-16 18:4x 서버 시계, 발사 후·S Sports 결과 전; 동결 sha `d3b7c518…` 이후 첫 개정)
1. **사고**: 발사 직후 `S_sports_s1`이 12초 만에 FAIL(`AssertionError: gbar = (E + A E) / 2 holds for n_layers 2`). Sports의 MSCA config는 `n_layers 3`(fusion_coeff .3)인데 xres0s 코드가 ḡ를 L=2 식으로만 구현하고 L≠2를 단언으로 막았다. 결과(result.json) 없음. 다른 job 영향 없음.
2. **수정(정의는 §4 그대로, 일반 L로 확장)**: ḡ_h = L⁻¹ Σ_{l=0}^{L−1}(Â^l E)_h (원고 Prop. history의 식 그대로; MSCA collab_u = L⁻¹Σ_{l=1}^{L} (Â^l E)_u = Σ_i R̃(u,i) ḡ_i). `ss_layer1`이 Σ_{l=1}^{L−1}(Â^l E)의 아이템 행을 계산(L=2이면 이전과 같은 식, 같은 연산), `self_summand`는 L=2이면 이전 식 `0.5·(E+l1)`, 아니면 `(E+l1)/L`. 수정본 `code/xres0s_a1.py` sha `bf1056e83c85bc5120a309b0922083547b73be7b3f5a931d02e0b60dcab4a1b4` (발사판 `373ac1ca…`는 `code/xres0s_launch1.py`로 mv 보존).
3. **수정본 점검(CPU, 테스트용 wt 사본 `tests/wt_a1`)**: (a) Baby 첫 배치 unit 재실행(`tests/a1_unit.out`) — S none/zero = F 비트 동일, S self·collab·view 손실과 기울기 차가 발사판 결과와 자릿수까지 동일(13.69111442565918, 4.369125235825777e-06 등) → L=2 세 ds의 S run은 발사판과 같은 계산. (b) tiny L=3(`tests/a1_tiny3.out`): 항등식 |e_u − Σ R̃ m| ≤ 1.2e-7, 합산항 대 벡터 ≤ 6e-8, 잔차/|S| .446, corr .82 (L=2 결과 불변). 첫 tiny L=3 실행은 테스트 스크립트의 벡터 검산 함수가 0.5를 하드코딩한 테스트 버그로 항등식 불일치가 나와 그 함수만 고침(코드 아님; 테스트 sha `6ed87753…`). (c) **실제 Sports seed-1 체크포인트**(`tests/a1_real_sports.out`, L=3, 40 간선): |s| 4.713, S 1.692 (S_C 1.084, S_V .608), 정확한 LOO 하락 1.718, 잔차 .209 = **|S|의 12.3%**, corr(S, LOO 하락) **.929**, 항등식 1.2e-7.
4. **재실행 규칙**: `S_sports_s1`만(결과 없음) 같은 seed로 1회 재실행. 이전 마커는 이름 변경 보존(`S_sports_s1.FAIL_launch1_nlayers_assert`, `.RUNNING.closed_launch1`), 실패 run 디렉터리는 `runs/_launch1_failed/S_sports_s1`로 mv. xres0s manifest의 src sha 갱신 → S run_id 지문이 바뀐다(발사판으로 이미 끝난 `S_pantry_s1` 지문 `8da1a7c5ad`, 이후 S run은 새 지문; L=2 계산은 3(a)로 동일함을 확인). wt 파일 목록 `wt_manifest_a1.sha256`.
5. **GPU0 예약**: 18:31:48 `study/GPU0_RESERVED` 생김 → GPU0 slot 2개가 규칙대로 종료(실행 중이던 `F_sports_s1`은 GPU0에서 끝까지). 이후 GPU1 3 slot만. 예상 벽시계 ≈ 7–9 h로 늘어남(cap 회계 불변).
