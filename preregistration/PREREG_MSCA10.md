# PREREG_MSCA10 — 융합 후 비교 MSCA+MES ⊕ vs MSCA ⊕ 를 seed 1–10으로 확장 (실험 ③, 가설 없음)

작성 2026-10-17 (서버 시계 12시대), MSCA10 작업자. 근거: 저자 결정(TMM 리뷰 2라운드 후; 저자 결정으로 취급). 상태: **동결** — 이 파일의 sha256을 `LAUNCH_LOG.md`와 서버 `study/MSCA10/prereg/PREREG_MSCA10.sha256`에 기록한 뒤 발사. 이후 변경은 번호 붙은 AMENDMENT로만(결과 전/후 표시).
규칙: 내부 규칙(seed당 1 run, 반복·best-of 금지), 내부 규칙(서버 스크립트+마커, 자동 폴링 없음), 내부 규칙(발사 직후 1회 확인), 원고 .tex·기존 생성기 수정 없음(새 파일만), <DISK>·/tmp 쓰기 없음, 삭제 없음, pkill/pgrep -f 금지, 기존 run·체크포인트 읽기 전용.

## 1. 목적 (가설 없음)
융합 후 마진 d = (MSCA+MES ⊕) − (MSCA ⊕)를 test Recall@20에서 seed 1–5(기존)에서 seed 1–10으로 늘려, 쌍체 검정·동등성 검정(TOST)·최소검출효과(MDE)를 n = 10에서 보고한다. 융합은 둘 다 본다.
- **F 융합**: RP3β+CBF (TABLE_FUSION 레시피 `tf_fuse.py`).
- **mix 융합**: 4-ranker 고전 혼합 (CLASSIC 레시피 `tf_fuse_ce.py`, `f_mix`).
판정 규칙·귀결은 없다(측정과 표만). 보조로 융합 없음(base) 대비도 보고한다.

## 2. 실행 목록
데이터셋: 우선 **튜닝된 4개(Baby, Pantry, Scientific, Sports)**, cap이 허용하면 **Instruments**(블록 X), 그다음 **Clothing**(블록 Y). 아래 run은 모두 seed당 1회, 복제 없음.

| 팔 | 데이터셋 | seed | 방식 | run 수 |
|---|---|---|---|---|
| MSCA+MES (M) | Baby, Pantry, Scientific, Sports | 6–10 | **재사용**: `study/MES10/runs/M_<ds>_s<seed>` (튜닝 λ/K: Baby .3/40, Pantry .1/20, Scientific .2/40, Sports .1/80; 체크포인트 20개 전부 존재 확인, 재학습 없음) | 0 (재사용 20) |
| MSCA native (N) | Baby, Pantry, Scientific, Sports | 6–10 | **신규 학습** `objx_s <ds>_native` | 20 (블록 A) |
| N + M | Instruments | 6–10 | 신규 학습: N `objx_s instruments_native`, M `xres0 instruments_xres_xeoa0_l010_k40`(MES9 설정) | 10 (블록 X, 조건부) |
| N + M | Clothing | 6–10 | 신규 학습: N `objx_s clothing_native`, M `xres0 clothing_xres_xeoa0_l010_k40` | 10 (블록 Y, 조건부) |

seed 1–5는 **기존 융합 셀 그대로**(재실행 없음): F = `study/MES9/fusion_cells_9ds_mes.csv`(`source` 체크포인트 = TABLE_FUSION의 MSCA `P04_listana`/`P04_sports`/`P04_clothing`/`P04_instr` objx_s native, MSCA+MES `P04_mestune`·`MANU`·`MES9`), mix = `study/CLASSIC/E1B/fusion_cells_ce.csv`. 신규 학습은 최대 40 run(4개 튜닝 데이터셋 20 run이 본체).

## 3. 코드·데이터 (보고된 MSCA fusion 체크포인트와 동일함을 확인)
- **MSCA native**: 워크트리 `study/MSCA10/wt` = `<HOME>/P04_xres_wt`(보고된 MSCA 융합 체크포인트의 `config_files`·`resolved_sources` 경로, branch xre_s2 406a7a7)를 .git·__pycache__·out·results(단 `results/data`의 data_manifest.csv는 포함) 제외하고 복사. 매니페스트 `wt_manifest.sha256`(1482 파일) sha `0b17d5ca…`. 보고 result.json의 `resolved_sources`(msca.py, objloss.py, objx_s.py, refset.py) sha가 사본 파일과 전부 일치. **지문 검증(`code/fp3.py`, CPU)**: `main.cell_fingerprint`가 보고 run_id 지문 6/6 재현 — baby(canonical) `a44b11973d`, pantry `0db37e4b42`, scientific `379bfe01a7`, sports `c20c07528b`, clothing `e59aeca5af`, instruments `9b022fad53` (모두 canonical_v2, baby만 canonical).
- **MSCA+MES (Instruments·Clothing 신규분)**: 워크트리 `study/MSCA10/wt_mes9` = `study/MES9/wt` 복사(1951 파일, 바이트 동일 diff 확인), 매니페스트 sha `861fc5cb…`. 지문 재현: instruments `ef7a3e3811`, clothing `4045f83524`(보고 MES9 run과 동일, canonical_v2). 네 튜닝 데이터셋의 M은 MES10 run(지문 baby `aa540a46dc`, pantry `90054d0b7a`, scientific `f8a16fda4c`, sports `9ffac46833`, 보고 MES 설정과 동일 코드·config).
- **데이터 사본**: 보고 MSCA 계열 run은 Baby에서 `canonical` 사본을 썼다(MES10 prereg §3에 기록된 kNN 캐시 차이 때문에 `canonical_v2`와 다름). **본 실험의 Baby는 N·M 모두 `canonical`**(M seed 6–10은 MES10에서 이미 canonical), 나머지 데이터셋은 `canonical_v2`. 지시에서 요구한 대로 명시한다.
- 학습 명령: `main.py --models <pkg> --configs <cfg> --datasets <ds>@<copy> --seeds <s> --stage train --execute --gpu <g> --allow-busy-gpu --data-copy <copy> --test-eval best_primed --save-ckpt --runs-dir study/MSCA10/runs/<id>` (cwd = 위 워크트리, nice 15, OMP 8). 보고 MSCA native 학습과 같은 stage 인자(지문 일치로 확인). 학습 환경(날짜, 동시 실행)은 seed 1–5와 다르다(§9 한계).

## 4. 융합 레시피 (기존 스크립트를 수정 없이 호출)
체크포인트 하나당 두 번의 CPU 재추론(학습 없음, GPU 없음):
- **F**: `TABLE_FUSION/code/tf_fuse.py` (sha256 `2399e9d0ea237aa80fe8b4322dc19db973bb07bb270967f6156520d99481ad8d`). 경로 f_combo_<w> = z(L) + w·z(RP3β+CBF), w ∈ {.05,.1,.2,.3,.5,.7,1,1.5}.
- **mix**: `study/CLASSIC/code/tf_fuse_ce.py` (sha `b0166f57ab1bc41bb404f145472bb87ed82bb3370ae73a152294cf21f2ec8ba5`, `classic_common.py` sha `c7019d9add0396d6c608ad086a3ce972c1cb8c318c570e1555f6cfdfe1791925`, 옵션 `--combo` 없음). 경로 f_mix_<w> (E1a 검증 선택 정수 가중 혼합).
- 가중 선택: **셀(= seed)마다 검증 Recall@20 최대 w**, 동점이면 작은 w (`tf_queue.py`·`tf_queue_ce.py`와 동일한 규칙; 구현 `code/msca10_fuse.py::pick`).
- 게이트(기존 큐와 동일): base test R@20·N@20이 result.json `best_test`와 5e-4 이내(두 레시피 모두) + F의 rp3b classic 게이트. 실패 셀은 `gate_ok=0`으로 기록하고 분석에서 제외(해당 데이터셋은 incomplete).
- **덤프에 대하여(지시와 다른 점, 사유 기록)**: 지시는 s4r_dump2.py 절차로 덤프 후 융합이었다. 두 융합 스크립트는 체크포인트를 직접 읽어 base를 다시 계산하고 result.json과 같은 5e-4 게이트를 걸므로 덤프(`tf_queue.py`가 쓰던 포인터·게이트 용도)는 정보가 없고 disk2(96%)에 큰 파일만 늘린다. 따라서 **S4R 덤프는 쓰지 않는다.** 대신 두 레시피가 같은 게이트를 자체 수행한다.
- **재현 점검(발사 전, CPU)**: Scientific MSCA seed 1(`P04_listana`)에서 F = 0.143720/0.072547 (w .1), mix = 0.143009/0.074639 (w .3)가 기존 셀(`fusion_cells_9ds_mes.csv`, `fusion_cells_ce.csv`)과 6자리까지 일치, 게이트 통과(`tests/fsmoke/`).

## 5. 검정 (동결; seed 1–10, 대비 = MSCA+MES ⊕ − MSCA ⊕, test Recall@20)
- **쌍체 검정**: `tables/src/sigtest.py` 의미(바이트 동일 사본 `code/sigtest.py`, sha `a3621bb34739294cd9045588b74b2556604feab37f4b7f455921d34af73adb47`) — 같은 seed 짝 양측 paired t(`ttest_rel`, df = n−1), 95% CI = t_{.975,n−1}·SD/√n, **Holm: 데이터셋 가족**. 기본 가족 = 튜닝 4개(`p_holm_tuned`), 보조 가족 = seed 10개가 모두 완성된 모든 데이터셋(`p_holm_all`). 유의 = p_holm ≤ .05.
- **TOST**: 마진 ±0.002 Recall@20, `tables/src/power_fusion_s4.py::analyse`를 수정 없이 import(사본 sha `208d67b56347b93b3b30c00ba67f1e08c775a4d0922c2d10549de23954d15b80`) — p_TOST = max(두 단측 p), α .05, 90% CI 한계 기준 최소 통과 마진도 보고.
- **MDE**: 같은 함수, 양측 paired t, α .05, 검정력 80%, 같은 n(= 10), 관측된 차이 SD 사용(비중심 t).
- 보고 단위: 융합(F, mix, base) × 데이터셋 × 창(seed 1–10, 비교용 seed 1–5를 같은 표에서 다시 계산)의 평균 MES·MSCA, 마진, CI, t, p, p_holm, wins, TOST p, MDE. 5-seed 재계산 F는 `power_fusion_s4.json`의 `old` view와 소수점까지 일치함을 발사 전에 확인(baby −0.55, pantry +1.08, scientific +0.21, sports +0.85 ×1e-3; p .335/.266/.852/.194; MDE 1.89/3.15/4.04/2.05).
- 데이터셋은 두 팔 모두 seed 10개와 모든 게이트가 완비된 경우에만 분석한다(아니면 incomplete로 표기, Holm m에서 제외).
- 산출: 서버 `MSCA10/summary/{msca10_perseed.csv, msca10_stats.csv, msca10_verdict.json, msca10_verdict.md}`; 원고 사본에는 새 파일 `tables/src/data/msca10_*.csv|json`만 추가.

## 6. GPU cap — 10 GPU-h, 하드 가드
- **정의**: GPU-h = 두 GPU 각각에 대해 이 큐의 job이 하나라도 돌고 있던 벽시계 시간(job 구간의 합집합)의 합. 동시 실행 job은 한 번만 센다. 보수적 비교용으로 job 벽시계 합(MES10식)도 마커에서 보고한다.
- **가드(claim 시 락 안)**: projected = 지금까지의 점유(합집합, 실행 중 job은 현재까지) + 실행 중 job의 남은 추정(max(est − 경과, 0.1 est)) + 새 job의 est + [블록의 첫 claim이면 그 블록 나머지 job의 est 합]. `projected ≥ 10`이면 시작하지 않는다: 블록 A는 해당 job `SKIP_CAP`, 블록 X·Y는 실행 중 job이 있으면 5분 대기(추정이 줄어듦), 없으면 블록 전체 `SKIP_CAP`(원자적). est = 보고된 seed 1–5 run의 평균 elapsed(단일 job 시간).
- **추정(단일 job 합)**: 블록 A(4개 N) **4.27 GPU-h**, 블록 X(Instruments N+M) 1.51, 블록 Y(Clothing N+M) 4.45; A+X = 5.78, A+X+Y = 10.23. 동시 실행(GPU당 최대 3 slot)은 점유를 추정 이하로 낮추므로 Y는 A·X의 실제 점유가 낮을 때만 가드를 통과한다(= "cap이 허용하면"). 가드를 통과하지 못하면 Clothing은 전부 SKIP_CAP이고 보고서에 그렇게 쓴다.
- 추가 안전: job당 timeout 5 h, OOM → FAIL(재시도·설정 변경 없음), 결과가 있는 seed는 재실행 거부(lane이 runs/<id>의 result.json 존재 시 거부), claim 전 `<DISK>` 여유 ≥ 40 GB와 GPU 여유 메모리 ≥ 5 GB 확인. FAIL/SKIP_CAP seed가 생긴 데이터셋은 incomplete.

## 7. 큐·마커 (자동 폴링 없음)
- `nohup bash code/run_msca10.sh > logs/run_msca10.out 2>&1 &` 한 번. slot 6개(`msca10_lane.py <gpu> <slot>`: GPU1 ×3, GPU0 ×3, nice 15), 공용 락 `state.lock`, 파일 순서로 동적 선점. `study/GPU0_RESERVED`는 현재 없음(`.released_20261017`만 존재); 생기면 GPU0 slot은 새 job 없이 종료(관례 유지). 두 GPU 모두 비어 있음 확인(2026-10-17 12:44, 메모리 18/104 MiB, 실행 중 앱 없음; MES10·CACHEFIX 큐 모두 종료).
- 순서 `jobs_msca10.tsv`: 블록 A(sports, pantry, baby, scientific; seed 6..10; 큰 데이터셋 먼저) → X(seed별 N, M) → Y.
- CPU 융합 드라이버 `msca10_fuse.py`가 옆에서 돈다(4 워커 × 2 스레드, nice 15): M 재사용 체크포인트는 즉시, N·M 신규 체크포인트는 DONE 마커가 생기는 대로. 결과 `cells/{F,CE}__<arm>__<ds>__s<seed>.json`, 요약 `summary/fuse_cells_new.csv`, 건너뜀 `summary/fuse_skipped.tsv`.
- 마커 `study/MSCA10/markers/`: `<job>.RUNNING(→.RUNNING.closed)|DONE|FAIL|SKIP_CAP`, `TRAIN_DONE|TRAIN_FAIL`(모든 lane 종료; DONE = 모든 job이 DONE 또는 SKIP_CAP이고 FAIL 없음), `FUSE_DONE`, **`QUEUE_DONE|QUEUE_FAIL`**(학습·융합·분석 모두 rc 0일 때만 DONE; `study/MSCA10/`에도 복사). 로그 `logs/msca10.log`, `logs/<id>.log`, `logs/fuse.log`. 큐 종료 시 `msca10_verdict.py`가 자동 실행.
- 디스크: 신규 체크포인트 약 3 GB(Baby 134 MB, Pantry 24, Scientific 21, Sports 343 MB × 5 + Instruments/Clothing). 덤프 없음. 셀 json은 수 KB. `<DISK>` 여유 213 GB(발사 전).

## 8. 스크립트 sha256 (서버 `study/MSCA10/`)
| 파일 | sha256 |
|---|---|
| `code/msca10_gen_jobs.py` | `35ac74a62d8eb399e2324b9cd58a9f07e37f1348b300b1311e88238a42921b53` |
| `code/msca10_lane.py` | `cae56a0900ce1cee1b18a88eee81f259b7ee915dd0bfb9d8e6ef7924b5a26c20` |
| `code/msca10_fuse.py` | `6941f371c63c7507023097d0d6c9c7ae8285de4c102a751da5c7577142ad54d0` |
| `code/msca10_verdict.py` | `10443ce1193a57226f9ee40d69058dfd09ccc02020fa31bff3391efed53321e1` |
| `code/run_msca10.sh` | `78bc332c65ad18b50ecc907cb773f3acbc77f1a921b88024ca44e7d89269ba83` |
| `code/fp3.py` (지문 점검) | `c9d72235800db60d0a84cad19a0081bf30d3c1e38e8ddb8017e61b9dd2e7b6e0` |
| `code/sigtest.py` (= 원고 `tables/src/sigtest.py`) | `a3621bb34739294cd9045588b74b2556604feab37f4b7f455921d34af73adb47` |
| `code/power_fusion_s4.py` (= 원고 `tables/src/power_fusion_s4.py`) | `208d67b56347b93b3b30c00ba67f1e08c775a4d0922c2d10549de23954d15b80` |
| `jobs_msca10.tsv` (40 job) | `f0e07940d167f680021eb63c22c7a4abbbd65f144d525be520e5e66e71e2419d` |
| `budget.txt` (10.0) | `dc1a600c48a24a74d5eac593b7695267b5253d65d95180f5e151329134b7b1b7` |
| `wt_manifest.sha256` / `wt_mes9_manifest.sha256` | `0b17d5ca…` / `861fc5cb…` |
| 호출하는 기존 스크립트 `tf_fuse.py` / `tf_fuse_ce.py` / `classic_common.py` | `2399e9d0…` / `b0166f57…` / `c7019d9a…` |

## 9. 한계 (사전 명시)
- seed 1–5(10/02–10/16 run)와 6–10(이번 run)은 같은 코드·지문이지만 GPU 비결정성(deterministic false)과 동시 실행 환경 차이가 남는다. 융합 체크포인트는 MSCA의 경우 seed 1–5가 별도 학습 run(`P04_listana` 등)이며 Table II 체크포인트가 아니다(원고 현재 융합 블록과 동일한 선택).
- 가중 w는 seed마다 검증으로 고르므로 마진에는 가중 선택 잡음이 포함된다(기존 5-seed 분석과 동일).
- n = 10에서도 데이터셋별 MDE가 ±0.002 마진보다 클 수 있다. TOST가 통과하지 못한 데이터셋은 동등성을 보인 것이 아니다.
- 결과 후 seed 추가·λ/K 재튜닝·설정 변경 없음.
