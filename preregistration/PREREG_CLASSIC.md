# PREREG_CLASSIC — 논지 B 범위 (i): 고전 단독·융합·동일 체크포인트·프로토콜 민감도 (CPU 전용, 가설 없음)

작성 2026-10-16 22시대. 근거: 저자 결정(저자 결정으로 취급). 원고 .tex 미수정(새 파일만 추가). 서버 HOST_A, CPU 전용(`CUDA_VISIBLE_DEVICES=`, nice 15, 총 ≤8 스레드), 출력 `<RUN_ROOT>/study/CLASSIC/`. CACHEFIX·MES10 디렉터리 접촉 없음, <DISK>·/tmp 쓰기 없음, 삭제·pkill 없음. 가설·판정 기준 없음: 모든 항목은 측정과 표 생성이다. seed당 1 run, 복제 없음(고전 랭커는 결정적이므로 seed 1 = 한 값).

## 1. 항목
- **E1a 고전 단독 (9 ds)**: RP3β, ItemKNN-CBF(text), ItemKNN-CF, EASE, F(= z(RP3β)+z(CBF)), mix(검증 선택 z-혼합)의 test Recall@20·NDCG@20. rp3b·icf 설정은 W1 `classic__<ds>.json`의 검증 선택값 재사용(격자 topK{50,200}×shrink{0,10}; RP3β α{.5,1}×β{.3,.6}, topK 200). MicroLens는 W1 json이 없으므로 같은 격자를 같은 코드(W1 `evaluate`, tie-adjusted 검증 R@20)로 새로 돌린다. CBF는 `feat/<ds>_feat.npz`(MicroLens·Arts·Office는 `W2/feat`)의 txt_knn(kNN-10) 연산자(tf_fuse.py와 동일). **EASE(신규)**: B = I − P diag(1/diag P), P = (XᵀX + λI)⁻¹ (float64 Cholesky 역행렬, float32 저장), **λ ∈ {10, 30, 100, 300, 1000, 3000, 10000}에서 검증 Recall@20 최대**(동점 시 작은 λ). **mix**: z(rp3b), z(iknn), z(icf), z(ease)의 정수 가중합, 가중 ∈ {0,1,2}⁴ (영벡터·전부 짝수인 중복 제외 65벡터), 검증 R@20 최대(동점 시 첫 벡터). 평가 규약 = tf_fuse.py: 학습 아이템 마스크, 검증·테스트 아이템은 후보, 테스트 타깃이 있고 학습 이력이 있는 사용자, 사용자 평균, 상위 20 동점은 낮은 아이템 인덱스, NDCG@20은 min(#타깃, 20) 이상적 DCG. 게이트: rp3b·icf의 tie-adjusted test R@20이 W1 `test_r20`과 5e-4 이내, rp3b·iknn·F가 STEP1 `f_standalone_<ds>.json`과 1e-6 이내. 스크립트 `code/classic_ds.py`(+`classic_common.py`), 출력 `<ds>/classic_<ds>.json`, 연산자 `<ds>/ops_*.npz`, `<ds>/ease_B.npy`.
- **E1b 융합 (CPU 재추론, 학습 없음)**: **이미 덤프(`<RUN_ROOT>/dumps`, `study/MES9`)가 있는 셀만**. 셀 = 비교 6모델(COHESION, DiffMM, FREEDOM, LGMRec, PGL, SMORE) 9 ds × seed 1–5, MSCA 6 ds, MSCA+MES 8 ds(TABLE_FUSION `fusion_cells_9ds_mes.csv`의 `source` 체크포인트). 레시피 = tf_fuse.py(sha 2399e9d0, z-점수 per user, 가중 w ∈ {.05,.1,.2,.3,.5,.7,1,1.5}, **seed마다 검증 R@20 최대 w**, 동점 시 작은 w): 경로 f_combo(기존 F, 재현 게이트), **f_ease**, **f_icf**, **f_mix**(= E1a의 mix 가중). 셀 게이트 = tf_fuse와 같음(base test R@20·N@20이 result.json과 5e-4 이내); 통과하지 못한 셀은 보고서·CSV에 `gate_ok=0`으로 남기고 제외 집계(한계 문장용 목록). 스크립트 `code/tf_fuse_ce.py`, 큐 `code/tf_queue_ce.py`, 출력 `E1B/cells/*.json`, `E1B/fusion_cells_ce.csv`.
- **P1 동일 체크포인트**: 로컬 CSV 변형. Table I·II 융합 블록의 MSCA 행을 **융합 run의 MSCA 체크포인트(TABLE_FUSION `base_r20`·`base_n20`)** 로 인쇄하는 데이터(`tables/src/data/classic_p1_msca.csv`)를 만들고, 비융합 MES−MSCA(R@20, 쌍체 t seed 1–5, ds Holm; `tables/src/sigtest.py`)가 Table I MSCA 행 대비 어떻게 변하는지 보고. 표 파일은 바꾸지 않는다.
- **E6 프로토콜 민감도**: 기존 run만(`protocol_runs.tsv`, `protocol_sota_runs.tsv`, `appendix_protocol*_cells.csv`, `table1_perseed.csv`, `table2_runs*.tsv`). 셀마다 seed 수를 인쇄하고 각주 "mixed seed counts". 생성기 `tables/src/build_table_protocol_restored.py` → `tables/appendix_protocol_restored.tex`(새 파일, `\input` 안 함). 새 run 없음.
- **플러그인 개수(사실 2)**: `table1_perseed.csv`(MSCA)·`cohsmo_perseed.csv`(COHESION, SMORE)에서 플러그인 5종(INA-style, MDVT, MG, DA-MRS, UltraGCN-LI) − Native, 데이터셋 4개(Table I 열), 쌍체 t seed 1–5, **백본×플러그인 대비마다 데이터셋 Holm**(`sigtest.py`), 유의 상승/하락 셀 수를 `tables/macros_plugincount.tex`로(0–10은 단어).

## 2. 출력과 규칙
- 새 파일(원고 사본, 추가만): `tables/src/data/classic_*.csv|json`, `tables/src/build_*_classic.py`, `tables/src/build_table_protocol_restored.py`, `tables/appendix_classic.tex`, `tables/macros_classic.tex`, `tables/appendix_protocol_restored.tex`, `tables/macros_plugincount.tex`. 표는 `\resizebox` 없이 데이터셋 패널을 쌓는다. 숫자는 손으로 쓰지 않는다.
- 긴 CPU 작업은 `nohup` + 마커(`CLASSIC/E1A_DONE`, `CLASSIC/QUEUE_DONE`), 원격 단일 루프로 ≥300 s 대기. 도구 호출 폴링 없음.

## 3. 스크립트 sha256 (동결 시점 = E1a 본 발사 22:14 + E1b 큐 발사 22:23, 서버 `CLASSIC/code/`)
```
c7019d9add0396d6c608ad086a3ce972c1cb8c318c570e1555f6cfdfe1791925  code/classic_common.py   (AMENDMENT 1 이후)
172539e633d186ee447e041dd87c819d4b861e49224a3228297ef6d829f62c75  code/classic_ds.py
0dbe1e6a29aeb3f606151208f934ebbc96364303c376d5ba6c05702c04396965  code/run_e1a.sh
b0166f57ab1bc41bb404f145472bb87ed82bb3370ae73a152294cf21f2ec8ba5  code/tf_fuse_ce.py
cf09f7694b2c6dbc1db41b8dfee7b485613162ebb7425abd7a54d7cca6ebf50f  code/tf_queue_ce.py
```
로컬 생성기(원고 사본 `tables/src/`): `build_p1_classic.py`, `build_plugincount_classic.py`, `build_table_protocol_restored.py`, `build_table_classic.py` (sha는 결과 보고서에 기재).
스모크: Scientific 28 s E1a 전 게이트 통과(W1·STEP1 일치); E1b Scientific MSCA seed 1에서 `f_combo`가 TABLE_FUSION의 기존 F(0.143720 / 0.072547, w .1)를 소수 6자리까지 재현.

## 4. AMENDMENT 1 (결과 전, 22:13)
EASE 역행렬을 scipy `dpotrf/dpotri` → `torch.linalg.cholesky_ex + cholesky_inverse`(float64)로 교체. 이유: scipy/OpenBLAS `dpotrf`가 Office(26k 아이템) 행렬에서 "8964번째 선행 소행렬이 양정부호가 아님"으로 실패(`G+10I`는 수학적으로 양정부호이고 20000×20000 부분행렬은 통과: 수치 버그). 교체 후 무작위 32행 잔차 max|AP−I| < 1e-8 단언을 추가. 첫 시도(scipy)로 만든 Baby·Pantry·Scientific 산출물은 `CLASSIC/v0_scipy_ease/`로 옮기고(삭제 없음) 9개 데이터셋 모두 torch 판으로 다시 계산한다.

## 5. AMENDMENT 2 (결과 전, 00:05–01:05)
E1a를 두 레인 4+4 스레드로 돌리는 동안 E1b 큐(`tf_queue_ce.py`, E1A_DONE 대기)를 쓰지 않고, **레인이 비는 대로 같은 셀·같은 러너를 데이터셋별로 일찍 시작**하도록 큐를 둘로 쪼갰다(총 스레드 ≤8 유지): `tf_queue_ce_early.py`(scientific, pantry, baby, instruments, sports, clothing, arts; 2 워커×2 스레드, 00:05 발사)와 `tf_queue_ce_early2.py`(office, microlens; E1A_DONE 후 2 워커×2 스레드). 두 파일은 `tf_queue_ce.py`에서 (a) E1A_DONE 대기·요약 파일 생성을 빼고 (b) 데이터셋 필터를 넣은 것뿐이며 셀 목록·러너(`tf_fuse_ce.py`)·가중 선택 규칙은 동일하다. 대기 중이던 원래 큐는 같은 셀을 이중 실행하지 않도록 PID로 종료했고, 모든 셀이 끝난 뒤 `tf_queue_ce.py`를 한 번 더 돌려 요약 CSV·`QUEUE_DONE`을 만든다(기존 셀은 건너뜀).
```
ddee60a49372a138a311a09c5571212ce66d800e2762fb117bbee917fe47749a  code/tf_queue_ce_early.py
fa43e9ed4fd251ae397b740a978fed0c770becffb10ea525796992e50908bcbd  code/tf_queue_ce_early2.py
```
E1a 결과(전 데이터셋 게이트 통과): `CLASSIC/<ds>/classic_<ds>.json`, EASE λ는 전부 격자 내부(경계 없음).

## 6. AMENDMENT 3 (결과 전, 01:57–02:05)
(a) `classic_common.py`의 `top20`을 k=21 방식으로 바꿔 보았으나(30개 합성 시행·실제 셀 1개에서 결과 비트 동일, 속도 이득 없음) **원래 파일(위 sha c7019d9a...)로 되돌렸다.** 모든 E1a·E1b 산출물은 c7019d9a... 판으로 만든다(`classic_common_v1.py`는 같은 파일의 사본, 로컬 `classic_common_v2_unused.py`는 폐기본).
(b) office 셀을 `tf_queue_ce_early2.py`(정순)와 `tf_queue_ce_early3.py`(역순, 셀 `.lock` 파일, `tf_queue_ce_early.py`가 끝나면 자동 시작, 2 워커×2 스레드)가 나눠 처리한다. 총 스레드 ≤8 유지.
```
2c739efb205670da4207dd4944ec1f87da3eda3435ca69dd57e5afea6f95aea6  code/tf_queue_ce_early3.py
```

## 7. AMENDMENT 4 (결과 전, 02:09)
셀 비용이 사용자×아이템에 비례(Scientific 12 s, Sports ≈4 min, MicroLens ≈15–20 min, Office ≈15 min, 2 스레드 기준)하고 numpy 원소 연산은 단일 스레드이므로 2스레드 워커 4개는 코어 약 5–6개만 쓴다. 예정 소요를 줄이려고 early·early2·early3 러너를 PID로 종료하고(진행 중이던 셀 4개 손실, 이미 끝난 191개 셀은 보존, 출력 파일이 있는 셀은 건너뜀) **원래 큐 `tf_queue_ce.py`를 단일 큐로 8 워커×1 스레드(총 8 스레드)로 재발사**했다. 변경은 `ORDER`(큰 데이터셋 먼저, 부하 균형)뿐이다. 마지막 단계에서 요약 CSV·`all_routes.json`·`QUEUE_DONE`을 이 큐가 쓴다.
```
5d3fb352be35505e548825ecdf40b88407ef92015beb761b942f33f9f0174c68  code/tf_queue_ce.py   (AMENDMENT 4판; 위 cf09f769...판은 ORDER만 다름)
```

## 8. 완료 기록 (2026-10-17 07:15, 결과 해석 아님)
- E1a: 9개 데이터셋 전부 게이트 통과(rp3b·icf 대 W1 5e-4, rp3b·iknn·F 대 STEP1 1e-6; MicroLens는 W1 격자를 같은 코드로 새로 돌려 선택, STEP1 없음). 서버 시간 합 4.6 h(4 스레드 = 18 core-h).
- E1b: `QUEUE_DONE` 324 셀, 오류 0, 전부 base 게이트 통과. 미생성 36 셀 = 덤프 없음 28(MSCA: Arts·Office·MicroLens 각 5, MSCA+MES: MicroLens 5, FREEDOM·LGMRec: Arts 각 4) + TABLE_FUSION 게이트 8(COHESION Sci s3·s4, LGMRec Pantry s3·s5, Sci s2·s4·s5, Sports s3). F 재현: 기존 TABLE_FUSION 셀 85개(Baby·Scientific)에서 `f_combo`가 6자리까지 일치(최대 차 0.000000). 셀 서버 시간 합 46 h(2스레드 191셀 6.0 h + 1스레드 133셀 40.2 h → 약 52 core-h).
- 로컬 생성기 sha256: `build_p1_classic.py` 76ebc9c4…, `build_plugincount_classic.py` ac6ce489…, `build_table_protocol_restored.py` 01052a98…, `build_table_classic.py` a8d177d9….
- 서버 최종 코드 sha: `tf_queue_ce.py` 5d3fb352…(AMENDMENT 4판), `tf_fuse_ce.py` b0166f57…, `classic_ds.py` 172539e6…, `classic_common.py` c7019d9a….
