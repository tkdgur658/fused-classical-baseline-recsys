# PREREG_CACHEFIX — Baby kNN 캐시 빌드 정정 재실행 (tab:main MSCA 블록 DA-MRS·UltraGCN-LI, tab:sota COHESION)

작성 2026-10-16 (서버 시계 19시대), CACHEFIX 작업자. 근거: 저자 결정(저자 결정으로 취급). 상태: **동결** — 이 파일의 sha256을 `LAUNCH_LOG.md`와 서버 `study/CACHEFIX/prereg/PREREG_CACHEFIX.sha256`에 기록한 뒤 발사. 이후 변경은 번호 붙은 AMENDMENT로만, 결과 전/후 표시.
규칙: 내부 규칙(seed당 1 run, 반복·best-of 금지), 내부 규칙(서버 스크립트+마커, 자동 폴링 없음), 원고 수정 없음, <DISK>·/tmp 쓰기 없음, 삭제 없음(mv만), pkill/pgrep -f 금지, MESC·MES10 파일 접근은 읽기(마커 존재 확인)만.

## 1. 목적 (가설 없음)
Baby의 item kNN 캐시는 두 빌드가 있다(`P04_data/canonical/baby` vs `canonical_v2/baby`; 이미지 그래프 70,500 간선 중 1,365, 텍스트 14, mm_adj_10 1,377 간선이 다름; 감사 `study/AUDIT_BABYCACHE/`, `tables/src/data/baby_cache_audit.csv`). MSCA 계열 보고 run은 `canonical`로 돌았으나 tab:main MSCA 블록의 DA-MRS·UltraGCN-LI(RunPod, `canonical_v2` 사본, 캐시 수준 빌드 미검증)와 tab:sota COHESION(pod, `canonical_v2` 이름)은 v2 이름으로 돌았다. **빌드 정정 재실행**: 같은 설정·같은 코드·같은 평가기로 데이터 루트만 `canonical`로 바꿔 한 번씩 다시 돈다. 가설은 세우지 않는다.

**교체 규칙(동결)**: build-correction re-run; the previous values are replaced in the table, not pooled. 각 (행, seed)는 **새 run 1개**만 표에 들어간다(이전 값과 평균·풀링·best-of 금지). 이전 값(RunPod/pod run)은 보충자료의 빌드 표(`baby_cache_audit.csv` + 새 per-seed 파일)에만 남긴다. seed당 1 run이며 복제 없음, FAIL·SKIP_CAP seed는 재실행하지 않고(코드 버그는 AMENDMENT) 해당 셀은 "incomplete"로 두며 이전 값으로 되돌려 채우지도 않는다(셀은 표에서 5 seed 완료 시에만 교체).

## 2. 실행 목록 (15 run, seed당 1 run, 복제 없음)
데이터 복사본 이름은 모두 `canonical`(`--datasets baby@canonical --data-copy canonical`), 코드·평가기 `--test-eval best_primed`, `--stage train`.

| 블록 | 패키지 / config | 코드 사본(원본 worktree) | 보고 지문(canonical_v2) → 새 지문(canonical) | seed | run |
|---|---|---|---|---|---|
| ① tab:main MSCA DA-MRS | `plg2_s` `baby_damrs_a1b2_k10_x2` | `CACHEFIX/wt1` ← `<HOME>/P04_table1fix_wt` (branch table1fix, HEAD d57a9af; `models/plg2_s/run.py`만 보고 시점 커밋 198b975의 바이트로 복원) | `b70f660207` → `7cd4edbc34` | 1–5 | 5 |
| ① tab:main MSCA UltraGCN-LI | `plg2_s` `baby_ugcn_li_k10_l0007` | 위와 같음 | `c67867bd43` → `af0d46fd00` | 1–5 | 5 |
| ② tab:sota COHESION (Native) | `cohesion` `COHESION_A5_baby` | `CACHEFIX/wt2` ← `<HOME>/P04_cohsmo_wt` (branch cohsmo, 77bb243) | `819beee84a` → `333b31191e` | 1–5 | 5 |

- 보고 run 근거: ① `tables/table1_perseed.csv`(MSCA, DA-MRS·UltraGCN-LI, baby, root qfixA/qfixE, fingerprint b70f660207 / c67867bd43), 수집 규칙 `tables/src/collect_table1.py`(config = `plg2_s <ds>_damrs_a1b2_k10_x2`, `<ds>_ugcn_li_k10_l0007`, 규칙 1: Baby는 canonical·canonical_v2 둘 다 eligible). ② `tables/table2_runs.tsv`(COHESION, baby, chosen 1; config COHESION_A5_baby, fingerprint 819beee84a), `build_table2_main.py`가 읽는 per-seed 원천.
- 스냅샷 `wt1`·`wt2`는 원본 worktree의 복사본(.git·`__pycache__`·`results/table1fix` 제외, 복사 직후 모든 파일 sha256 대조: wt1 1,520개, wt2 2,184개 동일). `configs/hosts.yaml`의 HOST_A `data_roots.canonical`만 아래 개인 overlay로 바꾼다(hosts.yaml은 지문에 들어가지 않음).
- ② 순서 제약: COHESION은 첫 run에서 자기 kNN 캐시(`mm_adj_10.pt`)를 overlay 안에 만든다(보고 run의 cohsmo overlay와 같은 방식). 두 run이 같은 파일을 동시에 만들지 않도록 ②는 항상 한 슬롯(role B)에서 직렬로만 돈다.
- 순서(`jobs_cachefix.tsv`): ① seed 1–5 × (DA-MRS, UltraGCN-LI) 순으로 10개, 그다음 ② seed 1–5. role A = ①만, role B = ② 먼저 그다음 남은 ①. 한 GPU에 동시 2 job 이하.

## 3. 데이터 루트와 게이트 (보고 run과 같은 지문; 데이터 루트만 다름)
- **데이터 overlay(서버 `study/CACHEFIX/`)**: `data1/canonical/baby` = `P04_data/canonical/baby`의 비-.pt 파일은 실제 대상으로 심볼릭 링크, .pt 캐시 14개는 복사 후 `chmod 444`(어떤 쓰기도 공유 farm에 닿지 않음). `data2/canonical/baby` = 비-.pt 원자료 19개만(kNN 캐시 없음; COHESION이 자기 빌드를 만듦). `result.json data_root` 기본 이름은 `canonical`.
- **게이트 G1 (코드·설정·코어·평가기 동일, 데이터 이름만 다름)**: 스냅샷에서 `main.cell_fingerprint`를 보고 run과 같은 stage 인자(train, best_primed, valid trainer, eval_state_sha, data_copy)로 계산해 `canonical_v2`일 때 보고 지문과 같음을 확인 — **3/3 일치**(b70f660207, c67867bd43, 819beee84a). 지문은 config 3개 파일 sha, 패키지 manifest, `core` 전체 .py sha, 데이터 manifest 행, stage 인자를 포함하므로, 같은 스냅샷에서 복사본 이름만 `canonical`로 바꾼 새 지문(7cd4edbc34, af0d46fd00, 333b31191e)은 데이터 이름·manifest 행만 다르다.
- **G2**: 보고 `result.json`의 `resolved_sources`(모듈 경로·sha256)가 스냅샷 파일과 같음 — DA-MRS 9/9, UltraGCN-LI 9/9, COHESION 2/2 모듈 동일. `registry.verify_manifest`(plg2_s, cohesion) 위반 0.
- **G3 (데이터 빌드)**: data1의 `image_adj_10_True.pt` sha256 = `3c9645eb…`, `text_adj_10_True.pt` = `5c477a4e…`, `mm_adj_10.pt` = `849d56c5…`(감사의 canonical 기준, v2는 `e70931…`/`33960e…`); data1의 baby manifest(copy local) 33행 모두 sha 일치, 누락 0. data2의 원자료 19개는 보고 COHESION overlay(`P04_cohsmo/data/canonical_v2/baby`)의 같은 이름 파일과 바이트 동일.
- **G4 (run 단위, lane이 자동 기록, DONE 마커의 `gate=`)**: result.json `fingerprint` = 위 새 지문, `data_root` = 해당 overlay 경로, `test_eval` best_primed, status SUCCESS. 하나라도 어긋나면 `gate=FAIL:…`이고 QUEUE_FAIL.
- **G5 (사후, `code/cachefix_postcheck.py`, CPU)**: data1 파일 목록·모든 .pt sha256이 설정 시점과 같음(run이 canonical 빌드를 바꾸지 않음), data2에 새로 생긴 파일은 `mm_adj_10.pt`뿐, COHESION이 만든 `mm_adj_10.pt`를 canonical `mm_adj_10.pt`·보고 cohsmo overlay와 간선 집합으로 비교(수치만 보고, 판정 아님; 감사 기준 canonical과 4/7 간선 차이 예상).
- 설정 게이트 결과 전체: 서버 `study/CACHEFIX/summary/setup_gate.json`(PASS).

## 4. 비용 추정과 cap
- 단일 job 추정(보고 run 평균 elapsed × 1.1): DA-MRS 995.8 s → **0.30 GPU-h**, UltraGCN-LI 1,166.2 s → **0.36**, COHESION(보고와 같은 지문의 HOST_A 재현 run PB seed 1·2 평균 606.3 s) → **0.19**.
- 합계: ① 10 run = 3.30 GPU-h, ② 5 run = 0.95 GPU-h, **총 4.25 GPU-h**(cap 5.0 안). ②는 ≤ 3 GPU-h 조건을 충족하므로 **포함**.
- 회계: 각 job 벽시계 elapsed의 합(동시 2 job이면 점유를 과대 계산하는 보수적 정의, MES10과 같음). **cap 5.0 GPU-h, 하드 가드**: claim 시 committed(끝난 job elapsed 합 + 실행 중 job est 합) + est ≥ 5.0 → 해당 job `SKIP_CAP`. 가드가 발동하면 그 셀은 incomplete(§1)이며 ②가 먼저 잘리지 않도록 ①이 앞 순서다.

## 5. GPU 스케줄·마커(자동 폴링 없음)
- 서버 `nohup setsid bash code/run_cachefix.sh` 1회. 서버 쪽 루프가 300 s마다 확인하고 다음 조건에서만 시작한다: **GPU0** — `study/MESC/markers/`(또는 하위 `gamma/`)에 `KILLED_K1`·`QUEUE_DONE`·`QUEUE_FAIL` 중 하나; **GPU1** — `study/MES10/QUEUE_DONE` 또는 `QUEUE_FAIL`. 먼저 생긴 쪽(파일 mtime)이 이긴다. 그 GPU에서 job 2개 이하 동시, `nice -n 15`, 시작 전 GPU 여유 메모리 ≥ 4,500 MiB·`<DISK>` 여유 ≥ 40 GB 확인(아니면 5분 대기). MESC·MES10 파일은 읽기만 하고 건드리지 않는다. 대기 중 취소는 `study/CACHEFIX/STOP`.
- 마커(`study/CACHEFIX/markers/`): `LAUNCHED`(선택된 GPU), job별 `{A,B}/<id>.RUNNING(→.RUNNING.closed)|DONE|FAIL|SKIP_CAP`, 종료 `QUEUE_DONE`(15 job 모두 DONE, lane 게이트·postcheck PASS) 또는 `QUEUE_FAIL`(사유 기록). 로그 `logs/cachefix.log`, job별 `logs/<id>.log`, run `runs/<id>/`, 사후 점검 `summary/postcheck.json`, `summary/cachefix_gate.csv`.
- 시작 직후 확인(실행자): 마커 `LAUNCHED`, 로그 `START` 줄, GPU 사용률 1회.

## 6. 실패 처리
OOM → FAIL(사유 OOM), 재시도·설정 변경 없음. rc≠0·발산 → FAIL, 재시도 없음. 결과가 있는 seed는 절대 재실행하지 않음(lane이 `runs/<id>`에 result.json이 있으면 거부). 코드 버그로 결과 없이 FAIL한 job을 고쳐 다시 돌려야 하면 이 문서에 AMENDMENT로 원인·수정 sha를 적고, 결과 없는 job만 같은 seed로 1회 실행.

## 7. 수집·시뮬레이션 (원고 .tex 변경 없음)
- 수집: `tables/src/collect_cachefix.py`(아래 §9 sha). 서버 `runs/*/*/*/result.json`을 읽어 `tables/src/data/cachefix_perseed.csv`(① 행, `table1_perseed.csv`와 같은 열: backbone,row,dataset,seed,r20,n20,r10,n10,gpu,root,fingerprint)와 `tables/src/data/cachefix_sota_perseed.csv`(② 행, `tables/table2_runs.tsv`와 같은 열)를 쓴다. 게이트 PASS인 완전한 5-seed 셀만 쓴다.
- 시뮬레이션: 원고 `tables/` 복사본(임시 디렉터리)에서 새 값으로 해당 per-seed·cells 행을 교체한 뒤 **원고 생성기 그대로**(`build_table1_main.py`, `build_table2_main.py`, `sigtest.py`: 쌍체 t, Holm over datasets, α .05)를 돌려 `table1_marks.csv`·`sigtests_table1.csv`·`table2_marks.csv`·`sigtests_table2.csv`의 ▼/▲와 best(bold) 변화를 보고한다. 원본 `tables/`는 수정하지 않는다. 임시 복사본에서 교체 전 기준선이 현재 산출물과 바이트 동일함을 먼저 확인한다(확인함).

## 8. 한계(발사 전 기록)
- ② COHESION은 보고 run에서도 COHESION 자신이 만든 kNN 캐시(mm_adj_10, 감사: canonical 대비 4/7 간선 차이, v2 대비 1,379 간선)로 돌았으므로 `canonical` 원자료 overlay에서 자기 캐시를 다시 만드는 재실행은 그래프가 사실상 같다. 따라서 ② 결과의 이전 값과의 차이는 대부분 seed·비결정성 변동이다(보고 값 대 HOST_A PB 재현 run의 seed 1·2는 R@20 0.1014 vs 0.1013, 0.1033 vs 0.1024). 그래도 지시대로 교체 규칙(§1)을 적용한다.
- ① 보고 run의 캐시는 RunPod에서 첫 사용 때 다시 만들어져 캐시 수준 빌드가 검증되지 않았다(감사). 새 run은 sha256으로 확인된 canonical 캐시를 읽는다.
- 서버 시계·동시 실행 상황에 따라 벽시계가 늘 수 있으며, cap 회계는 보수적(§4)이다.

## 9. sha256 동결 (서버 `study/CACHEFIX/` 기준, 2026-10-16 19:2x)
| 파일 | sha256 |
|---|---|
| code/cachefix_setup.py | `97f7602124a8cd16828263682a1023ee8e1d99ce4b4aaddb1d1cd0901adc9bec` |
| code/fp_calc.py | `40b57682a6a18ce3b452b20e42acae58bf166add809dacf402c61722c5fe1d65` |
| code/cachefix_lane.py | `57f6c0561fe20bac1ff31e7c4ac2f2bf1cedae4172662c5860eb45d7364cc2bb` |
| code/cachefix_postcheck.py | `0b4507a3dcea193f38ca3b322cea3cfb259965d5da75daf71a8d894c5660a15a` |
| code/run_cachefix.sh | `c2625dc02eb06d71f921fd298156f831f7c8a327c2d826b571b9bde8d439e0a6` |
| jobs_cachefix.tsv | `edcca777b070a29b70128623637dc56c82509d4763df1c700e8b65c92d9540ad` |
| budget.txt | `3bcb6f928021a3a46936a3f47316f26c4c80dd9bbfddb4df0110abd25baf98f3` |
| summary/setup_gate.json | `704e63005c4d6f12357c17344d0dba6bcc6aba390fd8481be1a7e01187b23474` |
| summary/wt1_manifest.sha256 | `c8ac2a5e56f6cd4c201d9a39683768078885a93f3611701fa5cd795dd1a9e433` |
| summary/wt2_manifest.sha256 | `36bd4b9742b47fa68b7cf943fd972092ec069767545730630cdb6cee9821ab97` |
| tables/src/collect_cachefix.py | (§10에서 기입) |

## 10. 수집기 sha (발사 후 기입 가능; 결과와 무관한 도구)
(collect_cachefix.py sha256은 발사 직후 `LAUNCH_LOG.md`에 기입.)
