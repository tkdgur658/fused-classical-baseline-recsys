# PREREG_MESCKPT — MES 체크포인트로 Settings 통계·Fig. nbrpop(및 체크포인트 분석 일체) 재생성 (측정 전 고정 초안)

**초안 — 실행 대기(저자 승인 필요).** 작성 2026-10-16 (PREREG-TODO, assistant). 서버는 읽기 전용으로만 조회했고 아무것도 발사·삭제하지 않았다.
근거: LAUNCH_LOG 야간 규칙(서버 todo는 초안만, 새 GPU 작업은 자체 cap + 승인), REVIEW_AUDIT_1 §6 "MES 체크포인트 재생성(Settings, Fig. nbrpop)", 연구 규칙 §5.3(한 표의 셀 = 같은 seed 집합, 측정 안 한 칸을 추정값으로 채우지 않음), NAME_CHECK(MES = xres0 xeoa0 entry-only, XRE = 이전 풀 설계 E+R+A).

## 0. 핵심 발견 (이 todo는 새 학습이 필요 없다)
1. **MES 체크포인트 셀은 이미 서버에 있다.** `<RUN_ROOT_B>/P04_hyp_strat/cells/<ds>_mes_s<k>.npz|json` 4 ds × seed 1–5 = 20 셀(10/10 생성, `hyp_strat_mes.py` + `run_mes.sh`, CPU 전용, 셀당 약 20 s). 이번 조회에서 mes 20 + native 20 셀의 json이 모두 `repro_ok ∧ recomputed_ok = True`(40/40, test R@20·N@20를 result.json과 소수 4자리까지 재현).
   MES 설정 = 본표 MES(검증 선택 λ/K): Baby .3/40, Pantry .1/20, Scientific .2/40, Sports .1/80 (NAME_CHECK와 일치). 체크포인트 파일(.pt)은 `P04_mestune/runs_mestune/xres0/…`·`MANU/runs/xres0/…`에 있고 셀은 이미 이것에서 추론된 결과다.
2. **로컬에도 이미 MES로 만든 데이터가 있다.** `tables/hyp_nbrpop.csv`(sha256 `92aef35c…a1736`, 400행 = 4 ds × 5 seed × graph 2 × link 2 × quintile 5, `model_b = MES` 전부), `tables/analysis_selfmsg.csv`에 `arm = mes` 100행, `tables/mmchan_mes_perseed.csv`(MSCA·MES 채널 분해). `figures/src/fig_nbrpop.py`는 CSV 라벨에서 "MES"를 자동 표기한다.
3. **그러나 원고 본문은 아직 옛 서술이다.** 4c L45–47(Fig. nbrpop 캡션) = "full design E + R + A … \todo{regenerate for MES}", 4a L28 = "same-seed runs of MSCA and of this full design\todo{regenerate on MES checkpoints}", 4d L35–38의 nbrpop 서술은 E+R+A 수치 기준. 즉 **남은 일 = (i) 그림 PDF가 현재 CSV(MES)와 일치하는지 재생성·확인, (ii) 캡션·본문·Settings 문장 교체, (iii) MES 기준으로 바뀌어야 하는 부록 분석(hyp_strat 계열) 재집계. GPU 불필요.**
4. **경고(주장 영향).** 현재 MES CSV의 co-occurrence 행에서 4d의 두 문장("popular neighbors gain hit rate on every seed of every dataset, more than twice the gain of other popular targets" / "Both rows lose hit rate in the third and fourth quintiles on every dataset")은 **E+R+A에서만 참이고 MES에서는 성립하지 않는 것으로 보인다**(예: Pantry Q5 neighbor +0.47 pp, 5/5 seed 부호 일치 아님; Baby Q3 neighbor +2.11 pp). 이 판독은 작성자가 CSV 평균을 이미 열람한 뒤의 것이므로 §4 규칙은 **사후(결과 열람 후) 규칙**으로 표시한다. 문장은 규칙이 정한 대로 기계적으로 바뀐다(주장 방향을 사람이 고르지 않는다).

## 1. 가설·목적
- 목적: 원고의 모든 "체크포인트 분석"이 제안 방법(MES) 한 가지 arm으로 통일되게 한다(이전 풀 설계 E+R+A와 혼용 금지). 새 가설 검정은 없다. 아래 §4의 문장 조건은 **서술의 정당성 점검**이다.
- 비교 쌍은 항상 같은 seed의 MSCA(objx_s `<ds>_native`) vs MES(`xres0`)이며 사용자·타깃 순서가 같다(셀 수집기가 assert).

## 2. 실행 목록 (학습 run 0개, CPU 수집 7개 단위)
| 단위 | 입력(이미 존재) | CPU 동작 | 산출(서버 `<RUN_ROOT>/study/MESCKPT/out/` → 로컬 `study/results/mesckpt/`) |
|---|---|---|---|
| U1 nbrpop | `hyp_strat/cells/<ds>_{native,mes}_s<k>` 40 셀 | `collect_nbrpop.py --arm mes` 재실행, 결과가 현 `hyp_nbrpop.csv`와 바이트 동일해야 함 | `hyp_nbrpop.csv`(재생성본), `nbrpop_check.txt`(sha 비교) |
| U2 nbrpop 그림 | U1 CSV | `fig_nbrpop.py` 재실행(CSV sha를 `fig_nbrpop.sources.json`에 기록) | `fig_nbrpop.pdf`(+ 인쇄 크기 PNG, 연구 규칙 U1.50 가독성 점검) |
| U3 strat | 같은 40 셀 | `collect_hyp_strat.py`의 `ARMS`를 `native,mes`로 바꾼 **사본**(원본 불변)으로 hist/pop 재집계 | `hyp_strat_hist_mes.csv`, `hyp_strat_pop_mes.csv` + 매크로 (부록 Fig. h_strat_hist, h_strat_pop 교체용) |
| U4 rankshift | 같은 40 셀 | `hyp_rankshift`의 pair `xre`→`mes`(사본) | `hyp_rankshift_mes.csv` (부록 Fig. h_rankshift) |
| U5 curves | MES run의 `*_train_val_loss_log.csv` | `collect_hyp_curves.py`의 xre 행을 MES run(본표 설정 20개)으로 교체한 사본 | `hyp_curves_*_mes.csv` (부록 Fig. h_curves) |
| U6 selfmsg | 서버 `P04_newprin/prop2/out/<ds>_mes_s<k>` | 로컬 `analysis_selfmsg.csv`(arm mes 100행)와 일치 확인만 | 점검 로그 (Fig. selfmsg는 MSCA 전용이라 그림 불변; 4d의 smShare·gradLost 매크로는 MSCA 값 그대로) |
| U7 events | MES(xres0) vs MES full view(xres0f) run 로그 | **조건부**: 두 arm이 `collect_hyp_events.py`가 읽는 이벤트 필드를 로그에 남기는지 확인한 뒤에만. 필드가 없으면 "— (로그 없음)"으로 두고 재현을 위한 학습은 하지 않는다 | `hyp_events_*_mes.csv` 또는 사유 한 줄 |
- 제외(사유 명시): `fig_h_gradsplit`(P04_prune의 stop-gradient arm은 E+R+A 기반, MES 기반 arm을 새로 학습해야 함 → GPU 필요, 이 prereg 범위 밖), `fig_h_forest`·`fig_h_sens`(각각 Table III 대조·PREREG_MESSENS가 담당), `fig_modality_todo`(채널 분해 데이터 `mmchan_mes_perseed.csv`는 이미 있음, 그림 제작은 별도 CPU 작업으로 저자 결정).
- 입력이 없는 ds·seed는 "—"로 두고 다른 arm 값으로 대체하지 않는다.

## 3. 비용과 cap
- **GPU 0 GPU-h**(모든 단위 `CUDA_VISIBLE_DEVICES=`, nice 15, 스레드 ≤ 6, 동시 프로세스 ≤ 2). 새 cap 요청 없음. 예상 CPU 시간: U1–U2 ≤ 5분, U3–U4 ≤ 15분(셀 npz 235 KB × 40), U5 ≤ 5분, U7 확인 ≤ 5분. 디스크 쓰기 < 50 MB.

## 4. 동결 규칙 — 무엇이 원고의 무엇을 대체하는가
**R1 (Settings 문장, 4a L28).** 교체문: "Checkpoint analyses use same-seed runs of MSCA and of MES, with the λ and K selected for each of the four original datasets." (`\todo` 삭제.) 아래 모든 U1–U5 게이트가 통과한 경우에만 `\todo`를 지운다.
**R2 (게이트).** 셀별 `repro_ok ∧ recomputed_ok`(test R@20·N@20 4자리 재현) 실패 → 그 (ds, seed)를 모든 단위에서 제외하고 "—", 같은 ds의 나머지 seed로 평균하지 않는다(같은 seed 집합 규칙: 5 seed가 모두 통과한 ds만 표·그림에 들어간다).
**R3 (Fig. nbrpop 캡션, 4c L47).** "from MSCA to the full design E + R + A"를 "from MSCA to MES"로, `\todo{regenerate for MES}`를 삭제. 나머지 서술(5 seed 평균, 색 포화 8 pp, 행·열 정의, 부호 일치 점)은 그림이 그대로 만족하는지 U2에서 확인(포화 값은 CSV 최대 |값|이 8 pp를 넘으면 캡션 숫자를 실제 값으로 바꾼다).
**R4 (4d nbrpop 문장은 아래 조건표로 기계적으로 선택).** 지표는 co-occurrence 그래프(`graph = cooc`)의 `gain_pp`(MES − MSCA, 같은 seed), 셀 = (ds, link, quintile), "부호 일치" = 5 seed 모두 같은 부호.
| 조건 | 참이면 문장 | 거짓이면 대체 문장 |
|---|---|---|
| C1: 4 ds 모두 (Q5, nbr)의 seed 평균 gain이 (Q5, non)의 평균 gain의 2배 초과이고 (Q5, nbr)이 5/5 부호 일치(양) | 현 문장 유지, 수치만 MES로 교체 | "Among the most popular fifth, neighbors of the history gain more than other targets on {k} of four datasets, with the sign shared by all five seeds on {m}." (k = 평균 차이 > 0 인 ds 수, m = (Q5, nbr) 5/5 양인 ds 수) |
| C2: 4 ds 모두 Q3·Q4의 두 행(nbr, non)이 평균 음수 | "Both rows lose hit rate …" 유지 | "Mid-popular targets (third and fourth quintiles) lose hit rate in {j} of 16 dataset–row–quintile cells, with the sign shared by all five seeds in {i}." (16 = 4 ds × 2 행 × 2 quintile 중 평균 음수 셀 수 j) |
| C3: 케이스 스터디 실패 사례 문장("This failure follows the third and fourth popularity quintiles …", 4d L47) | 그 타깃의 popularity quintile에서 Baby의 해당 행(nbr)이 평균 음수이고 5/5 부호 일치이면 유지 | **삭제**(케이스 문장과 그림의 직접 연결을 주장하지 않는다) |
- 사례 연구 그림은 이미 MES(`fig_case_study_mes_b`)이므로 불변. C 조건은 두 개의 "every dataset" 단정이 MES 데이터와 맞는지의 점검이며, 통과 못 하면 단정을 지운다(주장 약화이지 강화가 아니다).
**R5 (부록 U3–U5 그림 교체).** 부록 그림의 arm 표기는 "E + R + A"/"XRE" → "MES". 숫자가 바뀌는 문장은 매크로(`tables/macros_hyp_*.tex`)에서 재생성되며 방향 표현("not later", "every dataset")은 해당 CSV 요약이 조건을 만족할 때만 남기고 아니면 개수 표기("in k of four datasets")로 바꾼다. 매크로 생성기가 조건 불일치를 assert로 막지 않으면 이 규칙이 우선한다.
**R6 (arm 혼용 금지).** 한 그림·표 안에서 MES와 E+R+A를 섞지 않는다. 교체 후 E+R+A 기반 그림(U7 불가, gradsplit 등)은 캡션에 "earlier full design E + R + A"를 유지한다. 원고 .tex 편집은 저자 승인 뒤 revision log ⑭로 기록한다(이 초안은 .tex를 건드리지 않는다).
**R7 (표기).** 새 수치 앞 "최고", 유의 표현 금지. 이 규칙은 사후(결과 열람 후) 작성임을 revision log와 prereg 머리에 그대로 남긴다.

## 5. 실패 처리
- 입력 셀 누락·게이트 실패·OOM(CPU 메모리) → 해당 단위 마커 `FAIL_<unit>`, 재시도 시 설정 변경(스레드 수 제외) 없음, 그 칸은 "—". `.pt`를 다시 학습해 메우지 않는다.
- U1 결과가 현 `hyp_nbrpop.csv`와 바이트 다르면 **멈추고 보고**(수집기·셀 중 어느 쪽이 바뀌었는지 sha로 확인; 새 값으로 몰래 덮어쓰지 않는다).

## 6. 큐·마커 설계 (자동 폴링 없음)
- 서버 `…/<RUN_ROOT>/study/MESCKPT/`에 `code/run_mesckpt.sh` 하나(nohup, 레인 1개 = CPU 직렬, nice 15). 단위별 마커 `markers/<unit>.DONE|FAIL|SKIP`, 전체 `QUEUE_DONE|QUEUE_FAIL`, 로그 `logs/`. 시작 전 sha256 기입(수집기 사본, 쉘 스크립트), 상태는 마커 `ls`로만 확인(작업자가 sleep+ssh 하지 않음).
- 읽기 전용: `P04_hyp_strat/cells`, run 디렉터리. 쓰기는 `study/MESCKPT/` 아래만. 삭제 없음.

## 7. 산출물 경로와 보고 계획
- 서버 `…/study/MESCKPT/out/`, 로컬 `<PREREG_DIR>\results\mesckpt\`(csv·pdf·png·`nbrpop_check.txt`·`verdict_mesckpt.md`).
- 보고: (1) U1 바이트 동일 여부, (2) R4 조건 C1–C3 참/거짓과 채택된 문장, (3) 교체된 4a·4c·4d·부록 문장 목록(before/after), (4) "—" 셀 목록. 원고 반영은 승인 뒤 revision log ⑭(편집 전 사본 보존).

## 8. 합성·스모크 점검 계획 (실행 전)
1. **합성**: 손으로 만든 두 arm의 npz(알려진 rank, degree, link)로 5분위 교차표의 n·hit20·gain_pp를 정답과 대조(분위 규칙 = `fig_h_strat_pop`, `floor(5·pos/n)`), 부호 일치·C1–C3 판정 함수를 경계 사례(정확히 2배, 한 seed만 음수, 0 이득)로 시험. 로컬 CPU, 서버 접근 없음.
2. **스모크**: Baby seed 1–2 두 셀만으로 U1을 돌려 현 CSV의 같은 행과 비교(동일해야 함) → 통과 후 전체 U1–U5.
3. 셀 게이트 40/40이 실행 시점에 다시 통과하는지 확인(R2).

## 9. 한계
- 체크포인트 분석은 원래 4개 원 데이터셋만 다룬다(새 4 ds의 MES run은 `--save-ckpt`로 .pt가 있으나 셀 추론은 이 prereg 범위 밖; 포함하려면 별도 CPU 셀 비용(셀당 약 20 s×40)과 prereg가 필요).
- R4 조건은 결과 열람 후 작성이므로 확증 검정이 아니다. 숫자는 보고용이며 "최고"·유의 표현을 쓰지 않는다.

## 10. 타이밍 단계 (추가 2026-10-16 15:30, 실행자; 저자 18:3x "MES 제목 아래 모든 수치는 MES run에서" = 저자 결정으로 취급) — 측정 전 고정
- **현황 확인(읽기 전용)**: 서버 `MANU/timing/out_main_a1/timing_summary.tsv`(10/10 17:17, `tools/xabl_timing.py`, P04_novctrl_wt `dba2ba3`, seed 1, 3 epoch, 다른 프로세스 없는 GPU 한 장)에 **MES(xres0) 타이밍이 이미 Baby(l030_k40)·Pantry(l010_k20)·Scientific(l020_k40) 3개 데이터셋에 있다**(본표 MES 설정). `tables/timing_summary.tsv`와 `macros_timing.tex`(timeRatio 1.42/1.59/1.55, 1.4–1.6×)가 이미 이 MES 값이다. **빠진 칸은 Sports(l010_k80)뿐**이다.
- **프로토콜(Sports만 측정, 나머지 3개는 위 10/10 측정값을 그대로 쓴다. 3개를 다시 재지 않는다 = 반복 없음)**: 같은 도구·같은 커밋 `dba2ba3`(스크립트가 HEAD를 assert), 잡 2줄 `objx_s sports_native sports 1` (MSCA) 과 `xres0 sports_xres_xeoa0_l010_k80 sports 1` (MES = 본표 Sports MES 설정), seed 1, `--epochs 3`, 한 GPU에 다른 compute 프로세스가 없을 때만 시작(드라이버 REFUSE 가드), 에포크 0은 워밍업이라 평균에서 제외(도구 규칙: `mean_train_sec_excl_e0` = e1·e2 평균), 훈련 단계 peak allocated 메모리와 validation 1 pass 시간을 함께 기록. 체크포인트·test 평가 없음.
- **cap(하드 가드)**: 이 단계 GPU 5 GPU-h 상한. 실제 예상 ≈ 0.05 GPU-h(잡당 모델 구성 + 3 epoch ≈ 1–2분). 잡당 `timeout 1500 s`, 두 잡 합계 ≤ 3000 s = 0.83 GPU-h < cap. 사용량은 `timing/gpu_hours.txt`(벽시계 기준)에 기록.
- **마커**: `MESCKPT/markers/timing.RUNNING`, 종료 `timing.QUEUE_DONE | timing.QUEUE_FAIL`; 로그 `logs/timing_sports.log`; 결과 `timing/out_sports/<package>/…json|timing_summary.tsv`. 스크립트 sha256 `c4ee97f7…d926`(run_timing_sports.sh), 잡 파일 `fe7cda47…4204`.
- **실패 처리**: 어느 잡이든 rc≠0·REFUSE(GPU 공유)이면 `timing.QUEUE_FAIL`, 재시도 시 설정 변경 없음(GPU 번호만 바꿀 수 있음). Sports 타이밍이 없으면 timeRatioSports는 "—"로 두고 3개 데이터셋 범위만 보고한다.
- **판독 규칙(측정 전 고정)**: timeRatio = MES / Native의 `mean_train_sec_excl_e0`, memRatio = MES / Native의 `peak_train_alloc_mib`. 4개 데이터셋 범위(최소–최대)를 매크로로 낸다. "최고"·유의 표현 금지.

## 11. 실행 기록 규칙 (U1–U7)
- 실행자는 §6 대로 sha256 기입 후 발사한다. 아래는 구현상 결정(결과 열람 전):
  - U3·U4(strat/rankshift) = 원본 `collect_hyp_strat.py`의 사본 `collect_hyp_strat_mes.py`(`ARMS=['native','mes']`, 비교 arm 키 `xre`→`mes`, 출력 파일명 `*_mes.csv`·`macros_*_mes.tex`)를 쓴다. 매크로 이름(hz…Xre…)은 원고의 \input 교체만으로 대체되도록 **그대로 둔다**(값은 MES, 파일 머리 주석에 명시).
  - U1·U2: `collect_nbrpop.py --arm mes` 로직을 그대로 쓰되 출력 경로만 MESCKPT/out으로 바꾼 사본 `collect_nbrpop_mes.py`. 기준: 현 `tables/hyp_nbrpop.csv` 와 바이트 동일(sha `92aef35c…a1736`).
  - U5: MES run 경로는 `mes_perseed.csv` 행 `MES`(backbone MSCA) run_dir, MSCA 쪽은 기존 `table1_runs.tsv`와 동일 run. 로그 `*_train_val_loss_log.csv`의 `is_best_epoch`와 `result.json`의 best_epoch가 어긋나면 그 run은 FAIL, "—".
  - U7: MES(xres0) vs MES 전체 뷰(xres0f) run의 `stdout.log`에 `[R12]`/`[XRE]` 모니터 줄이 있는지 먼저 확인. 없으면 SKIP.

## 12. 실행 전 고정 기록 (2026-10-16 16:0x, 실행자)
- 합성 시험(§8.1) 통과: `study/mesckpt_verdict.py selftest` — 손으로 만든 10-타깃 두 arm으로 분위×이웃 표의 n·hit20이 brute-force 정답과 일치(7셀), C1–C3 판정 함수 경계 사례(정확히 2배는 불통과, 한 seed 음수, 0 이득, 평균 음수이나 한 seed 양수) 통과.
- 스모크(§8.2) 통과: 서버에서 Baby seed 1–2 두 셀로 `collect_nbrpop.py remote mes` 실행, 현 `hyp_nbrpop.csv` 같은 행 40개와 모두 동일(차이 0).
- 서버 코드 sha256 (`MESCKPT/code/sha256_frozen.txt`): collect_nbrpop.py 2b10e16a…, collect_hyp_strat_mes.py 5fa42dca…, collect_hyp_curves_mes.py db8bb926…, collect_hyp_events_mes.py 4daaba00…, check_selfmsg_mes.py f861427b…, gate_cells.py 191d731e…, run_mesckpt.sh e76a0745…. 기준 CSV `ref_hyp_nbrpop.csv` 92aef35c…a1736(= tables/hyp_nbrpop.csv).
- 추가 확정(결과 열람 전): U7 비교 쌍은 **MES_k40(xeoa0_l010_k40, 마스크 뷰) vs MES_fv(xeoa0f_l010_k40, 전체 뷰)**: 두 arm 모두 λ 0.1·K 40이라 뷰만 다르다(본표 MES λ/K와 다른 Baby·Pantry·Scientific·Sports 설정을 비교에 쓰지 않는다). MES는 retention 항이 없어 mass_retention = 0(assert), 원 분석의 "entry share"는 구조적으로 1이라 보고하지 않고 p_in_list(listed)와 mass_entry를 보고한다. U6은 재집계가 아니라 셀 json·run_dir·test R@20의 일치 점검. 새 그림은 `fig_*_mes.py`(기존 파일 불변), 출력 `figures/fig_*_mes.pdf`.
- 발사: 단일 nohup `code/run_mesckpt.sh`, CPU 직렬, nice 15, GPU 0.
