# PREREG_MES9 — MSCA+MES를 Arts·Office·Clothing·Instruments·MicroLens에서 학습해 융합 사실 F를 9 ds로 확장 (측정 전 고정)

작성 2026-10-15 (서버 시계 20시대), PREREG-study. 상태: **초안 = 동결 대기** (스크립트 sha256 기입 + 저자 승인 전 실행 금지).
근거: DESIGN_G2M.md §1.2 F, 연구 규칙 §5.3(한 표의 셀은 같은 seed 집합 [U1.20]; OOM이 나도 설정을 몰래 줄이지 않음 [U2.8]; 측정 안 한 칸은 추정값으로 채우지 않음 [U2.39]), 내부 규칙(seed당 1 run), 내부 규칙(서버 스크립트+마커, 자동 폴링 없음), 내부 규칙(시작 직후 1–3분 확인).

## 0. AMENDMENT A1 (2026-10-15 20:2x KST, 저자 결정, **결과 전**, 이후 본문과 충돌하면 A1이 우선)
1. **데이터셋 = instruments, clothing, arts, office** (추정 비용 오름차순, 블록 단위). **MicroLens 제외** → 표기 "— (융합 불가: W1/classic 없음)" (본문 §2·§4·§8의 microlens 행은 무효). 총 job 20개 = 4 ds × seed 1–5, seed당 1 run, 복제 없음.
2. **cap = 16.0 GPU-h** (10에서 상향; 사용자가 거부할 수 있음). `study/MES9/budget.txt`=16.0. **STOP_OFFICE**: 로컬 `<PREREG_DIR>\study\STOP_OFFICE`(시작 시점 없음 확인) 또는 서버 `<RUN_ROOT>/study/STOP_OFFICE`·`study/MES9/STOP_OFFICE`가 office 블록 판정 시점에 존재하면 office를 시작하지 않는다(`BLOCK_office.STOP`). 블록 승인 규칙은 유지: 블록 첫 job 디스패치 시 1회 `used_committed + 5·est·ρ ≤ 16`. ρ는 DONE job만으로 계산(clip .5–2.0, 없으면 1), used_committed = 끝난 job(DONE+FAIL) 실측 합 + 실행 중 job의 est·ρ. 블록 거절 시 그 블록 job은 `SKIP_BLOCK`, 표기 "— (cap)". 하드 가드 used_actual ≥ 16.0 → 새 job 중단(`SKIP_CAP`).
3. **office 디스크 검사**: office 블록 판정 직전에 `<DISK>` 여유 ≥ 60 GB가 아니면 `BLOCK_office.DISK`를 쓰고 office를 시작하지 않는다 → 표기 "— (디스크)". (시작 시점 여유 61 GB, 앞 블록의 체크포인트·덤프가 쌓이므로 디스크 사유 탈락 가능성이 크다. 이 경우도 규칙대로 보고하며 정리·삭제는 하지 않는다.) 그 밖의 디스크 규칙: 레인 대기(GPU 프로세스 <2, 여유 ≥ 40 GB)와 §6의 블록 REJECT_DISK(5·최대 run 용량 > 여유−40 GB)는 그대로.
4. **새 4 ds MES = 미튜닝(고정 λ 0.1, K 40)**. 표 각주·Settings에 "미튜닝, 기존 4 ds는 validation 선택" 문구를 쓴다. 결과 후 (λ,K)를 바꿔 재실행하지 않는다.
5. **plan 사전 점검 결과(10/15 20:21, CPU, 학습 없음)**: ① 스냅샷 `wt`에서 4 ds config를 생성(`mes9_gen_jobs.py`: xre config + `nc_parts: entry` 블록 + `nc_anchor: drop` 블록; 같은 함수가 커밋된 baby·pantry·sports xeoa0 config를 바이트 단위로 재현함을 확인). ② 첫 `--stage plan`은 4개 모두 `UNSUPPORTED: dataset not supported by package` — 원인 = 스냅샷 `models/xres0/run.py`의 `SUPPORTED_DATASETS` 등록 목록이 4 ds(baby, pantry, scientific, sports)만 가짐(경로 문제 아님, 등록 문제). 훈련 코드·하이퍼파라미터는 건드리지 않고 **이 등록 리스트 한 줄에 4 ds를 추가**(스냅샷 사본만, 원본 저장소 불변, `patch_run_registration.py`)하고 manifest의 run.py sha를 갱신했다. ③ 재실행한 `--stage plan`(ds당 1회, seed 1)은 4개 모두 `cells=1 unsupported=0`, rc 0 (`study/MES9/logs/plan_<ds>.txt`). 4 ds 모두 "— (설정)" 아님. ④ 데이터: `canonical_v2` 농장(<DISK>, 읽기)에 4 ds 디렉터리·mm_adj 캐시 존재, data_manifest 행 존재. ⑤ run_id 지문은 git을 쓰지 않는다(`core`에 git 호출 없음), 스냅샷은 `.git` 제외 복사.
6. **큐 구현**: `run_mes9.sh` 1개(nohup), 레인 2개(GPU0/1, nice 15), 두 레인이 `mes9_lane.py`로 job을 락 아래 동적 선점(순서 = jobs 파일 순서). 마커(`study/MES9/markers/`): job별 `<job>.DONE|FAIL|SKIP_BLOCK|SKIP_CAP`, 블록별 `BLOCK_<ds>.ADMIT|REJECT|REJECT_DISK|STOP|DISK`, 5 job 종료 시 `BLOCK_<ds>_DONE` + `summary/BLOCK_<ds>.json`, 전체 `study/MES9/QUEUE_DONE|QUEUE_FAIL`. OOM(로그 `out of memory`) → FAIL 사유 `OOM`, 재시도·설정 변경 없음, 다른 seed는 계속, 그 ds 블록은 "— (OOM, seed k)". 덤프(`dumps/<ds>/mes__<ds>__s<s>[__valid].npz|json`)는 TABLE_FUSION 9-ds 재실행용으로 test+valid 모두 생성.
7. **재집계 스크립트**(`tf_queue_g2m.py`, `merge_9ds.py`)는 이 발사에 포함하지 않는다. 덤프가 생긴 뒤 재집계 단계에서 작성하고 실행 전에 별도 sha를 기입한다(§8 표의 해당 행은 "후속"). 본문 §7 PASS 규칙은 그대로이며 분모에는 microlens의 기존 9ds 표 셀이 그대로 남는다.
8. 데이터 경로: 실행은 hosts.yaml의 `canonical_v2` 농장(`<RUN_ROOT_B>/P04_data/canonical_v2`, 기존 모든 run과 같은 경로)을 `--data-root`로 쓴다. 모델이 없는 캐시를 쓰면 농장에 쓸 수 있으나 4 ds의 캐시는 이미 존재한다. 본문 §3의 `MES9/MMRec` 링크는 사용하지 않는다.

## 1. 가설
- **H_F9**: 사후 융합(W2 recipe, `tf_fuse.py`: f_combo 가중치 = validation R@20 argmax)이 R@20을 올리는 (모델, 데이터셋) 셀의 비율이, MSCA+MES 행을 5개 새 데이터셋으로 확장한 9-ds 표에서도 ≥ 0.9이다. (현재 6 ds 46/46, `TABLE_FUSION_9DS`.)
- 새로 생기는 셀은 최대 5개(MSCA+MES × {instruments, clothing, arts, microlens, office}). 다른 모델의 9-ds 셀은 이미 `TABLE_FUSION/fusion_cells_9ds.csv`에 있으며 이 사전등록으로 다시 계산하지 않는다(재집계만).

## 2. 정확한 실행 목록 (25 run = 5 ds × seed 1–5, seed당 1 run, 복제 없음)
| 우선순위 | ds | run | job id | 설정 |
|---|---|---|---|---|
| 1 | instruments | s1–s5 | `M9_instruments_s{1..5}` | `instruments_xres_xeoa0_l010_k40` |
| 2 | clothing | s1–s5 | `M9_clothing_s{1..5}` | `clothing_xres_xeoa0_l010_k40` |
| 3 | arts | s1–s5 | `M9_arts_s{1..5}` | `arts_xres_xeoa0_l010_k40` |
| 4 | microlens | s1–s5 | `M9_microlens_s{1..5}` | `microlens_xres_xeoa0_l010_k40` |
| 5 | office | s1–s5 | `M9_office_s{1..5}` | `office_xres_xeoa0_l010_k40` |

- 순서는 dataset 블록 단위(블록 안은 seed 1→5). 블록은 원자적: 5 seed가 모두 끝난 ds만 표에 들어간다(같은 seed 집합 규칙). 일부 seed만 끝난 블록은 "— (cap)"로 표기하고 부분 수치는 부록 로그에만 둔다.
- 각 run = 학습(`main.py --models xres0 --configs <cfg> --datasets <ds>@canonical_v2 --seeds <s> --stage train --execute --data-copy canonical_v2 --test-eval best_primed --save-ckpt`, GRID `grid_lane.py`의 검증된 호출과 같은 인자) 후 `s4r_dump2.py`로 test+valid 덤프 → 태그 `mes__<ds>__s<s>`.

## 3. 설정 (새 튜닝 없음)
- 패키지 `xres0` (MSCA + masked-view entry supervision). 코드 스냅샷 = `P04_novctrl_wt` @ dba2ba3 (7aa3e25 이후 `models/xres`, `models/xres0/src`, `core`, `main.py`의 변화는 추가된 config 파일뿐임을 `git diff --stat`로 확인함). 스냅샷 sha256 목록은 `study/MES9/wt_manifest.sha256`(§8).
- 5개 데이터셋 config는 기존 생성 규칙(`tools/xen0_package.py`)을 그대로 적용해 **생성**한다: `models/xres/configs/<ds>_xres_xre_l010_k40.yaml`(MSCA author config A5 + R12 키; 5개 모두 존재) + 두 줄 `nc_parts: entry`, `nc_anchor: drop` 부가 → `<ds>_xres_xeoa0_l010_k40.yaml`. backbone MSCA 값(lr 1e-3, emb 64, n_layers 2, knn_k 10, fusion_coeff .4, tau .2, cl_weight .005, reg 3e-7, batch 256, 100 epoch, patience 10, AdamW, cosine, AMP, wd 0)은 base yaml이 이미 가진 값이며 새로 정하지 않는다.
- **MES 초모수 = λ 0.1, K 40 (`xeoa0_l010_k40`, 기존 프로토콜의 사전 기본값)로 5 ds 모두 고정.** 기존 4 ds 주표는 seed 1 validation으로 ds별 (λ,K)를 골랐다(Baby .3/40, Pantry .1/20, Scientific .2/40, Sports .1/80). 새 5 ds에서는 그 선택 절차를 반복하지 않는다(DESIGN_G2M이 튜닝을 허용하지 않고, ds당 5 config × seed 1 = 25 run이 더 든다). 따라서 새 5 ds의 MES는 **미튜닝**이며 4 ds보다 불리한 쪽이다. 보고서와 본문에 그대로 쓴다. 결과를 본 뒤 (λ,K)를 바꿔 다시 돌리지 않는다.
- split/데이터: `canonical_v2`, leave-one-out(test 양성 1개 사용자 92–100%), 주표와 같은 seed 1–5, 같은 평가기(`best_primed`). 데이터 경로는 `configs/overall.yaml`의 `data_path: ../MMRec/data/`이므로 `study/MES9/MMRec -> <HOME>/MMRec` 심볼릭 링크로 해석(§8 사전 점검).

## 4. 비용과 cap
근거: 같은 5 ds의 기존 5-seed XRE(`xres`) elapsed_sec 중앙값 × 0.90 (xres0/xres 비: baby .89, pantry .94, scientific .87의 평균) + 덤프 0.03 h(GRID est 0.02/job + valid). 서버 result.json에서 읽은 값이다.

| ds | XRE 중앙값 h/run (max) | 추정 h/run | 블록(5 run) | 누적 |
|---|---|---|---|---|
| instruments | .250 (.406) | .255 | 1.27 | 1.27 |
| clothing | .320 (.338) | .318 | 1.59 | 2.87 |
| arts | .925 (1.018) | .863 | 4.31 | 7.18 |
| microlens | 1.057 (1.087) | .981 | 4.91 | 12.08 |
| office | 1.815 (3.119) | 1.663 | 8.32 | 20.40 |

- **총 추정 20.4 GPU-h > cap 10 GPU-h.** 전체 9-ds 확장은 cap 안에서 불가능하다(약 2배). cap은 서버 `study/MES9/budget.txt`에 10.0으로 고정(GPU-h = 모든 레인의 학습+덤프 elapsed 합).
- **동결 우선순위 = 추정 비용 오름차순(위 표 순서).** 블록 승인 규칙(서버 스크립트가 블록의 첫 job 디스패치 시 1회 계산, 번복 없음): `승인 ⇔ used_committed + 5·est(ds)·ρ ≤ 10.0`. used_committed = 끝난 job 실측 합 + 실행 중 job의 est·ρ. ρ = clip(끝난 job 실측 합 / 끝난 job 추정 합, 0.5, 2.0), 끝난 job이 없으면 1. 거절된 ds보다 뒤(더 비싼) ds는 같은 규칙으로 판정되며 사실상 함께 탈락한다.
- 합성 검증(§9): ρ_true = 1.0이면 instruments·clothing·arts(≈7.2 h)만 승인되고 **microlens·office 탈락**. microlens가 승인되려면 ρ_true ≲ 0.55, office는 ρ_true ≈ 0.4가 필요하다. **기본 예측: 9 ds 중 MES 셀은 7 ds(기존 4 + instruments·clothing·arts)에 생기고, MicroLens·Office의 MSCA+MES는 "— (cap)"**. 이 탈락은 OOM이 아닌 예산 사유이며 표에서 "— cap"과 "— OOM"을 구분 표기한다.
- 하드 가드: used_actual ≥ 10.0이면 새 job 발사 중단(실행 중 job은 종료까지 둠). 초과폭 ≤ 레인당 job 1개(arts/microlens ≈ 1.1 h, office 최악 3.1 h).
- **저자 결정이 필요한 점**: Office·MicroLens까지 채우려면 cap ≈ 21 GPU-h(2레인 벽시계 ≈ 10.5 h)가 필요하다. 이 사전등록은 cap 10을 바꾸지 않는다. 늘리려면 이 문서를 새 번호로 개정하고 시작 전에 승인을 받는다.
- 레인 2개(GPU0/GPU1), 레인당 1 job. 예상 벽시계(ρ=1) ≈ 4–5 h.

## 5. 큐 설계 (자동 폴링 없음)
- 서버 폴더 `<RUN_ROOT>/study/MES9/` 안에서만 쓴다: 코드 `code/`, 설정·스냅샷 `wt/`, 런 `runs/`, 덤프 `dumps/<ds>/`, 마커 `markers/`, 로그 `logs/`.
- `code/mes9_lane.py <lane>` = `GRID/grid_lane.py` 복제에서 다음만 변경: jobs 파일 `jobs_mes9.tsv`(§2의 25줄 + est), 학습 cwd = `study/MES9/wt`, 모델 `xres0`, 덤프 출력 루트 `study/MES9/dumps`, 블록 승인 로직(§4). 시작 전 대기 조건은 GRID와 동일(GPU 프로세스 <2, `<DISK>` 여유 ≥40 GB).
- 마커: `markers/<job>.DONE|FAIL|SKIP_CAP`, 블록 판정 `markers/BLOCK_<ds>.ADMIT|REJECT`(ρ, used, 판정 한 줄), 전체 종료 `MES9_DONE`.
- 시작은 `nohup bash code/run_mes9.sh &` 한 번. 상태 확인은 마커 `ls`만 한다(작업자가 sleep+ssh 하지 않음). 새 GPU 작업이므로 시작 직후 1–3분 간격으로 첫 job이 epoch 로그에 진입했는지 확인하고 끝낸다.

## 6. 실패 처리 (사전 고정)
- **OOM**(로그에 `CUDA out of memory`): 그 job은 마커 `FAIL`, 재시도 없음, 설정 변경(batch, K, λ, 해상도) 없음. 그 ds 블록은 표에서 "— (OOM, seed k)"; 같은 ds의 나머지 seed는 예산 안에서 계속 돌리되 블록은 무효(같은 seed 집합 규칙). 원인과 GPU 메모리 피크를 로그에서 인용해 보고한다.
- 그 외 rc≠0 또는 발산(NaN 등): 재시도하지 않는다(복제 금지). FAIL 사유만 기록하고 칸은 "—".
- 덤프 게이트(|recalc_r20 − json_r20| ≤ 5e-4) 실패 → 그 셀 제외, 사유 기록.
- `<DISK>`는 현재 99% 사용(여유 61 GB). 첫 블록 종료 시 런당 체크포인트·덤프 용량을 `du`로 기록하고, 다음 블록 합계 추정이 (여유 − 40 GB)를 넘으면 그 블록 REJECT(예산 사유와 구분 표기).

## 7. 지표·판정·보고 (동결)
- 지표: test R@20(주), N@20(보조), 게이트 후 `fusion_cells_9ds.csv`와 같은 열. 융합은 `TABLE_FUSION/code/tf_fuse.py`/`tf_queue.py`(CPU, nice 15)를 쓰되 덤프 루트에 `study/MES9/dumps`를 추가한 복제본 `study/MES9/code/tf_queue_g2m.py`로 `--datasets <승인된 ds>`만 돌려 `study/MES9/fusion_cells_mes9.csv`에 쓰고, 기존 9ds 표와는 `merge_9ds.py`로 합친다(기존 파일은 수정하지 않음).
- 셀 = (모델, ds). seed 1–5 평균 융합 R@20 > 융합 전 R@20이면 "이득 셀". 두 값이 같으면 이득이 아니다.
- **PASS(H_F9)**: (이득 셀 수) / (게이트 통과 평가 셀 수) ≥ 0.9. 평가 셀 = 기존 9ds 표 셀 + 새 MES 셀(생긴 것만). 보고에 반드시 쓴다: 분모 셀 수, 새 MES 셀만의 n/m, "— (cap)"·"— (OOM)" 목록. 새 MES 셀이 3개 미만이면 "F의 9-ds 확장은 부분적"이라고 명시한다.
- 비율 <0.9이면 FAIL로 그대로 보고하고 사실 F를 "확장 시 성립하지 않음"으로 기록한다. ds 부분집합으로 좁혀 되살리지 않는다(부분집합 보고는 별도 번호의 사후 분석으로만).
- "최고"·유의 표현은 쓰지 않는다. 새 5 ds MES는 미튜닝 표시. 새 seed 확인 단계 없음.

## 8. 스크립트 경로와 sha256 (동결 시 기입)
| 파일 | 서버 경로 (`<RUN_ROOT>/study/MES9/`) | sha256 |
|---|---|---|
| `mes9_gen_jobs.py` (jobs_mes9.tsv, config 생성) | `code/` | `84dfee6038cb9ec4885595adc67d69f988a4d101ae91c41acfa5bbc9f4c0f83d` |
| `patch_run_registration.py` (A1.5 등록 리스트 1줄) | `code/` | `df41142da292b84b7cb21316b054cc375c2a8eb3e5aa735220ec289d0f3efc81` |
| `plan_once.sh` (plan 사전 점검) | `code/` | `2b4f250ab4ad84681c6f8a0040308b21624ea0eb5ba548525fef5ec4e359f953` |
| `mes9_lane.py` | `code/` | `738ae38b30150ec13ed284124ad4c85224e4fbbfcd81bb7a28cfdbc3537bb6f5` |
| `run_mes9.sh` | `code/` | `f9cf66b19edbc897347973edb29088c4affef50d25f685074209ba137b90bb9c` |
| `tf_queue_g2m.py` / `run_tf_g2m.sh` / `merge_9ds.py` (재집계, 10/16 00:43 실행) | `code/` | `7842fbb3f16322160ba3b65f81cc5c4bf17feb393a73b29171b6101094ebc43c` / `169a476568e727d0fbd4be205087d4c2db45aeef84c391466a70a2a761ad1ca7` / `335e51d83945d1e7fc83c6d68f8e91cd875adf5bdb630f13374048656207c3be` (실행 직전 sha256sum) |
| `s4r_dump2.py` (GRID 것을 바이트 그대로 복사) | `code/` | `23d47181d47f6887d3a543754e6df337931a1067112bacc4a901cc9fa499fb04` (GRID/code 원본과 일치 확인) |
| 코드 스냅샷 (P04_novctrl_wt@dba2ba3, .git·__pycache__ 제외 tar 복사; 4 ds config 추가 + run.py 등록 1줄 수정 후 파일 목록) | `wt/` | `wt_manifest.sha256` 파일의 sha = `861fc5cb3ced2d748c912b3bd6c632e1ef962cf748c49cd543e4cdcdb53e559c` (1951 파일; 수정 전 목록 `wt_manifest_orig.sha256` sha `2f25df2a8a33367321b199208e2f9259ad361a0ed21b23d7e9d9cf9a408d29ad`) |
| 합성 검증 `synth_mes9_admit.py` (로컬 `study/`) | — | `c6b7a062c25f5e1b77b185153d988190baf4328aafb88e2bdd368f802ff64b30` |

**시작 전 사전 점검**(학습 아님, CPU 수 초):
1. `wt/` 안에서 `main.py --models xres0 --configs <ds>_xres_xeoa0_l010_k40 --datasets <ds>@canonical_v2 --stage plan`(--execute 없음) 5개가 오류 없이 계획을 출력하는가.
2. `MES9/MMRec` 링크로 5 ds 데이터 파일이 존재하는가.
3. `<RUN_ROOT>/feat/<ds>_feat.npz`와 `W1/classic/<ds>/classic__<ds>.json`이 5 ds에 있는가. **관찰(10/15 ls): `W1/classic/`에 arts·baby·clothing·instruments·office·pantry·scientific·sports만 보이고 microlens가 없다.** MicroLens 융합은 이 파일(또는 W2 쪽 대체 경로)이 확인되기 전에는 계산할 수 없다. cap 규칙상 MicroLens는 탈락 가능성이 높지만 승인될 경우를 위해 시작 전에 확인한다.
4. wt는 원래 git worktree라 `.git` 파일이 원본 저장소를 가리킨다. run_id 지문이 `git rev-parse`를 쓰는지 plan 출력으로 확인한다(쓰면 원본 경로를 읽기 전용으로 둔다).
- 점검 중 하나라도 실패하면 실행하지 않고 보고한다(설정을 임의로 바꾸지 않는다).

## 9. 합성 역검증 (로컬 실행 `synth_mes9_admit.py`, 출력 전문 = `synth_mes9_admit.out`)
```
frozen estimates (GPU-h/run): {'instruments': 0.255, 'clothing': 0.318, 'arts': 0.863, 'microlens': 0.981, 'office': 1.663} | block(5 seeds): {'instruments': 1.27, 'clothing': 1.59, 'arts': 4.31, 'microlens': 4.91, 'office': 8.32}
static cumulative: {'instruments': 1.27, 'clothing': 2.87, 'arts': 7.18, 'microlens': 12.08, 'office': 20.4} | cap 10.0
rho_true=0.4  admitted=['instruments', 'clothing', 'arts', 'microlens', 'office'] dropped=[] total_GPUh= 7.41 wall_h(2 lanes)= 3.71
rho_true=0.55 admitted=['instruments', 'clothing', 'arts', 'microlens'] dropped=['office'] total_GPUh= 6.53 wall_h(2 lanes)= 3.39
rho_true=0.7  admitted=['instruments', 'clothing', 'arts', 'microlens'] dropped=['office'] total_GPUh= 8.65 wall_h(2 lanes)= 4.62
rho_true=1.0  admitted=['instruments', 'clothing', 'arts'] dropped=['microlens', 'office'] total_GPUh= 7.57 wall_h(2 lanes)= 4.34
rho_true=1.4  admitted=['instruments', 'clothing', 'arts'] dropped=['microlens', 'office'] total_GPUh=10.54 wall_h(2 lanes)= 5.50
rho_true=2.0  admitted=['instruments', 'clothing'] dropped=['arts', 'microlens', 'office'] total_GPUh= 5.55 wall_h(2 lanes)= 2.85
admission flags in priority order 0.4 [True, True, True, True, True]
admission flags in priority order 1.0 [True, True, True, False, False]
admission flags in priority order 2.0 [True, True, False, False, False]
F all positive (51 cells): (1.0, True)
F 5 negative of 51 (ratio .902): (0.9019607843137255, True)
F 6 negative of 51 (ratio .882): (0.8823529411764706, False)
F 2 exactly-zero gains count as NOT helped (ratio .961): (0.9607843137254902, True)
SYNTH_MES9 OK
```
해석: (a) 총 추정 20.4 h는 cap의 2배이다. (b) 승인 집합은 항상 우선순위의 접두이고 ρ가 커질수록 줄어든다. (c) 총 소모는 cap + job 1개 이내(10.54 h가 최대)이다. (d) 판정 규칙은 .902 → PASS, .882 → FAIL을 올바르게 구분한다.

## 10. 한계 (사전 명시)
- 새 5 ds의 MES는 미튜닝(λ.1/K40)이고 기존 4 ds는 ds별 validation 선택값이다. F는 "융합이 MES 위에서도 이득을 낸다"는 사실이지 "MES가 이긴다"는 주장이 아니다.
- 비용 추정은 XRE 경과 시간에서 외삽했다(MES는 이 5 ds에서 학습한 적이 없다). ρ 보정은 첫 블록(instruments)이 끝난 뒤에야 적용된다.
- Office·MicroLens의 MES 셀은 기본 예측에서 비어 있다. 9-ds 완결은 cap 개정(≈21 GPU-h) 없이는 불가능하다.
