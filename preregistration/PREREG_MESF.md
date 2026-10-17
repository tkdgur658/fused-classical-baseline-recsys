# PREREG_MESF — MES가 요건 F(융합 기준선)에 답하게 만든다 (저자 결정 10/16 07:5x "P04 MES-F ㄱ")

작성 2026-10-16 07:5x, 저자. 측정 전 고정(세부 수치·sha는 실행 작업이 발사 전 기입). 서버 HOST_A(GPU 유휴), GPU cap 10 h.

## 0. 문제 (확정 사실)
- F: 사후 융합(학습 점수 ⊕ 고전 RP3β·kNN·content)이 50/50 셀에서 R@20을 올린다. 요건 = "학습 추천기의 이득은 융합된 고전 위에 남는 만큼만 인정".
- MES의 이득은 융합 전 +.0007…+.0027(튜닝 4 ds; 융합 run의 MSCA 체크포인트 기준. Table 1 기준으로는 +.0008…+.0030, 10/16 09:5x 확인), 융합 후 MSCA⊕ 대비 −.0005…+.0011로 **유의하지 않음**. 즉 현재 MES는 F에 답하지 못한다(MED QFIX 전 상황과 동형).

## 1. 설계: MES-F (방법이 F에서 유도됨)
- 원리: 학습 목표를 "융합 기준선 위에 남는 이득"으로 둔다. 고전 융합 점수 s_C(u,i)(TABLE_FUSION 레시피의 고전 결합 점수, 학습 중 **고정**, train split 상호작용만으로 계산)를 모델이 받는다.
- 구현 1순위 **MES-F(res)**: 최종 점수 s(u,i) = s_C(u,i) + g_θ(u,i), g는 MES 구조 그대로(자유 ID 임베딩·멀티모달·self-message). 손실은 MES 그대로(BPR/softmax 계열), 단 양·음 모두 s에 대해. 학습 시 s_C는 정규화(z-score, 데이터셋별 고정)·스케일 α는 validation 선택(α∈{0.5,1,2}, seed 1만, 이후 고정).
- 구현 2순위(1순위가 발산·OOM 시) **MES-F(in)**: s_C를 추가 특징으로 입력(아이템 측 1차원 + 사용자 측 1차원), 점수는 기존대로.
- 비교: 같은 융합 레시피로 사후 융합한 MSCA⊕·MES⊕(기존) 대 MES-F(추가 융합 없음; "이미 F를 포함"). 공정성: MES-F에 다시 고전 점수를 융합하지 않는다(이중 계산 방지). 보조 행으로 MES-F⊕도 보고(해석 없음).
- 데이터·프로토콜: 튜닝 4 ds(Baby·Pantry·Scientific·Sports) × seed 1–5, canonical_v2, 평가기 best_primed, 주표와 동일. 미튜닝 4 ds는 예산 남으면 seed 1만(보조).

## 2. 판정 (사전 고정)
- 주 기준 **H_F**: test R@20에서 MES-F > MSCA⊕ 가 4 ds 중 **≥3**에서 Holm 보정 후 유의(paired t over seeds, α .05) ∧ MES-F ≥ MES⊕ 4/4 평균. 통과 → "MES-F는 F에 답한다"(기여 2 = F에서 유도된 방법), 원고 Table 2에 MES-F 행 추가·정식화-방법 유도 문장·기여 재배열. 
- 보조: N@20 동일 방향 ≥3/4; 융합 전 비교(MES-F 단독 vs MES)는 보고만.
- 실패: (a) 유의 0–2 ds → "F 위의 추가 이득은 작다"를 사실로 보고, 기여 2는 현 MES(방법 중하)로 유지, 사용자에게 투고 여부 재결정 요청. (b) MES-F < MES⊕ → 잔차 학습이 해롭다는 사실로 보고.
- 사후 튜닝(사용자 허용): α 외 하이퍼파라미터 조정은 validation만, "사후" 표시, seed 1회, cap 안.
- 예측: H_F 통과 .35. 위험: 고전 점수 포함은 "학습 랭커 + 고전 결합"으로 읽힐 수 있음 → 원고는 "결합이 아니라 F를 목표로 둔 학습"임을 손실 식으로 보임(G-K4·G-F5), 사후 융합(⊕)과의 차이를 Table로.

## 3. G0′ (발사 전, 서버 0)
- 검색 2회: "residual learning over heuristic/classical recommender score", "boosting-style recommender on top of item-kNN/RP3β", "hybrid with fixed classical prior in loss". 정확 선행(같은 목적: 융합 기준선 위 이득을 학습 목표로)이 있으면 그 이름으로 비교하고 방법 기여는 "적용"으로 낮춘다. 발견 내용은 RW에.

- G0′: partial prior — BRIDGE (2605.22073): fixed train-only neighbor offset + learned graph residual with BPR, top-200 candidates; differs in fused RP3β+CBF z-score offset, full catalog, MES loss, and the F-derived objective; see study/G0P_MESF.md

## 3b. A1 구현 확정 (MESF-IMPL, 2026-10-16 09:1x 서버 시계, **측정 전**; 본문과 충돌 시 A1 우선)
1. **s_C 정의**: TABLE_FUSION `Zs['combo']` (RP3β: W1 validation 선택 config α.5 β.3 topK200 / ItemKNN-CBF(text) 각각 사용자별 z, 합, 재 z; train(x_label 0) 상호작용만). z 통계는 해당 버전의 *가시 이력 밖* 아이템으로 계산하고 값은 전 아이템(이력 포함)에 부여해 학습 쌍을 점수화한다. 사용자·아이템 수 = 데이터 파일 최대 id+1.
2. **캐시 3종 (dense float32)**: `full`(train 전체; 추론 및 `mesf_sc=full`용), `f0`/`f1`(= MES 마스크 뷰의 가시 이력, 접힘 f의 간선 제거; `mesf_sc=oof`용). **채택 = `oof`**: 학습 쌍 (u,p)(접힘 f)과 그 음성은 f-제외 캐시, 마스크 뷰 f의 목록·엔트리 쌍도 같은 캐시 → 쌍 자신의 간선이 s_C에 들어가지 않는다. `full`(문자 그대로의 고정 s_C)은 in-sample 부풀림으로 엔트리 항이 퇴화함을 CPU 진단으로 확인(baby: 학습 양성 top-20 적중 .669 vs 접힘 제외 양성 .128 vs 검증 양성 .062; α=1 학습 4스텝에서 엔트리 행 비율 oof .93 vs full .11). 추론은 항상 `full` 캐시. 두 캐시 모두 train 상호작용만 사용·학습 중 고정이므로 §1 문구를 위반하지 않는다. `full`은 보고용 절제로만 남긴다(발사 안 함).
3. **점수 주입 지점(MES 손실 불변, 합산 점수를 봄)**: (1) 주 BPR의 pos·neg 점수, (2) 마스크 뷰 top-2K 목록 선택 점수, (3) 엔트리 쌍 softplus(s_uD − s_up)의 점수, (4) 추론 full_sort_predict. α=0이면 모든 재정의가 XRES0 계산을 그대로 반환(같은 코드 경로).
4. **α 선택**: seed 1, 4 ds × α∈{.5,1,2} = 12 run, 지표 = seed-1 validation Recall@20 (`best_valid_primed` → 없으면 `best_valid_score`; test 필드는 읽지 않음), 최대 α, 동률이면 작은 α, 이후 동결. 선택된 α의 seed-1 run이 곧 주 큐의 seed-1 run이다(복제 금지; 주 큐는 덤프만 추가). 주 큐 = 4 ds × seed 2–5 신규 16 run. (λ, K)는 주표 MES와 동일(Baby .3/40, Pantry .1/20, Scientific .2/40, Sports .1/80).
5. **cap 10 GPU-h**: α 선택 + 주 큐 레인 경과시간 합(두 단계 마커), 추정 합 ≈ 7.9(α 선택 3.1 + 주 큐 4.8; 근거 = 기존 MES 5-seed 평균 경과 baby .113 / pantry .144 / scientific .075 / sports .462 h/run × 1.3 + 덤프 .03). 하드 가드: used + est ≥ 10 → `SKIP_CAP`. OOM → `FAIL`(사유 OOM), 재시도·설정 변경 없음, 표기 "—".
6. **MES-F(in)**: 구현하지 않음(§1 2순위 = 1순위 발산·OOM 시의 대비책).
7. **마커**: `study/MESF/markers/{alpha,main}/<job>.RUNNING|DONE|FAIL|SKIP_CAP`, `markers/ALPHA_DONE`, `QUEUE_DONE|QUEUE_FAIL`, `summary/alpha_selected.json`. 덤프 `study/MESF/dumps/<ds>/mesfres__<ds>__s<seed>[__valid]`.
8. **G0′ 외 위험(사실)**: g(MES 내적) 사용자별 sd ≈ .59, s_C ≈ 1.1 → α·s_C가 점수를 지배한다(α .5–2). 결과가 "고전 점수가 대부분"으로 읽힐 수 있으므로 MES-F 단독 대 s_C 단독·MSCA⊕도 보고한다.

## 4. 비용·운영
- 4 ds × 5 seed = 20 run(+α 선택 seed 1 × 3 = 12 run 단기) ≈ 6–8 GPU-h, cap 10. HOST_A 2레인, 마커 study/MESF/markers, QUEUE_DONE. 사전등록 sha·합성 검증(판정 스크립트에 가짜 결과 3 시나리오) 발사 전. 결과 1회.
- 완료 후: 판정 → 원고 반영(통과 시 ⑮) 또는 보고.

## 5. 스크립트 sha256 (frozen at 2026-10-16 09:12:42 서버 시계, 발사 전; 서버 `<RUN_ROOT>/study/MESF/`)
| 파일 | sha256 |
|---|---|
| `wt/models/mesfres/src/mesfres.py` (MESFRES 클래스) | `0d0fc761f4bacd20fa3ad881864dbf2af9d4031d2cfb0e41a89fb43b64d84315` |
| `wt/models/mesfres/run.py` | `e43e8a4a3bb1486b6798aafe4e8be101b55ccee043e163e26f28d40aad15bc1c` |
| `wt/models/mesfres/manifest.json` | `258d7be843dc193cdd0f8ea530b474fb67bf0df4744f32e7c39e06222363a35b` |
| `wt/models/mesfres/configs/*.yaml` (32개, `sha256sum *.yaml` 의 sha) | `fc28c71955fe2667198b1d58b333dd06c2831bd2baea44a12a2b07b198a7a910` |
| `code/mesfres_sc.py` (s_C 캐시) | `972079e8c52f2cc15d65462ef9a8b14f4f6c1c84ede83634bf8943cbf853ba51` |
| `code/mesf_setup.py` | `363c92265346a4f8170b024c31d2e2b9417a69668dec0c1f830ca89f5c155c8c` |
| `code/mesf_lane.py` (09:13:24 1차 발사가 락 파일명 연산자 우선순위 오류로 job 0개 실행·학습 0 상태에서 즉시 종료 → 괄호 1곳 수정, 09:14:49 재발사. 결과는 본 것이 없음. 최초 sha 494f62ac…) | `bc413144595a2b68fb976287b783863617fdcf0d4ba213d4d90f1f259ddf2061` |
| `code/mesf_gen_alpha.py` | `fb65538028781fdeb54692d4f3b5d3efd1244dd44777f291e0dfee4ecac09879` |
| `code/mesf_alpha_select.py` | `1261dccd151023ff662a912f200502fe3c9999638e0452263545b9741ff1a463` |
| `code/mesf_gen_main.py` | `1f141e6f36d734c59d5d5419f2ad8b8cd15c784c97f6be1f94622868e9623fde` |
| `code/run_alpha.sh` | `fceaacabedb246a617c32f50987cce0308942b031a1360fb73993e5bfd201516` |
| `code/run_main.sh` | `478c4d39169e970bb01790aa6332c8036ac5466d09169126562ef4e8dfd274c4` |
| `code/mesf_tests.py` (검증) | `c3e966c31b22e938dc7b3eb9f7693e30208dd8349048e2a2f3f45ca185f62f53` |
| `code/s4r_dump2.py` (MES9 것과 바이트 동일) | `23d47181d47f6887d3a543754e6df337931a1067112bacc4a901cc9fa499fb04` |
| s_C 캐시 sha (앞 12자; full/f0/f1) | baby 9daa24dea6c0/f399115f5f10/30f4273cd543; pantry fe90a714d323/5b95f4cc7252/1c59326f28e7; scientific 7cfd817fa7b0/c1b65dfc9743/267be17266f0; sports 0cd4faa086fc/47d87ec6d4dc/8e560ca3b7ab (전체는 `sc_cache/<ds>_sc_meta.json`) |

## 6. 발사 전 점검 결과 (CPU, 상세 `study/MESF_IMPL.md`)
(a) 누설: train-only 파일(x_label 0만)로 재계산한 캐시가 기존 캐시와 바이트 동일(baby full·f0), val/test 쌍 중 train과 겹침 0. (b) α=0: 첫 배치 손실·모든 기울기 XRES0와 비트 동일(오프셋 코드 경로 강제), 기존 MES seed-1 baby 체크포인트 재평가 test R@20 .1040(저장 .1041), valid .1045(저장 .1045) — 허용 5e-4 이내. (c) 합성 역검증: MES-F(res) R@20 .413 vs MES .377(잔차 상관 .72 vs .70).

## 7. α 선택 결과 (frozen at 2026-10-16 10:05, seed 1 validation Recall@20만; α 선택 12 run 전부 DONE, 사용 1.64 GPU-h)
| ds | α=0.5 | α=1 | α=2 | 선택 α (동결) | 참고: 주표 MES seed-1 valid |
|---|---|---|---|---|---|
| Baby | .0891 | .0843 | .0821 | **0.5** | .1045 |
| Pantry | .1347 | .1321 | .1295 | **0.5** | .1309 |
| Scientific | .1853 | .1791 | .1758 | **0.5** | .1919 |
| Sports | .1034 | .0975 | .0939 | **0.5** | .1208 |
(사실: 4 ds 모두 α가 클수록 validation이 단조 하락, 선택은 격자 하한. 하한 α=.5에서도 Baby·Scientific·Sports는 MES 단독의 validation보다 낮고 Pantry만 높다. 격자 밖 α는 사후 튜닝(validation만, 사용자 허용 범위)으로만 가능하며 여기서는 하지 않음. test 필드는 선택에 읽지 않았다.)

## 8. 사후 튜닝(사전등록 밖) 관문 (frozen at 2026-10-16 10:09 서버 시계(발사 10:10:19 직전), 결과 보기 전; 저자 결정, 저자 위임)
- 승인 사항: oof 캐시 = 사전등록 안. 주 큐는 α=0.5 그대로 끝까지(건드리지 않음). cap은 MES-F 전체 **12 GPU-h**(10에서 증액; 두 단계 마커 합 + 이 관문).
- 관문 run: α∈{0.1, 0.2}, **seed 1만**, 4 ds = 8 run (복제 없음, 주 큐와 별도 레인: GPU0/GPU1에 각 1 프로세스 추가, nice 15, 마커 `study/MESF/markers/posthoc/`, 종료 `POSTHOC_DONE|POSTHOC_FAIL`). 학습·평가는 주 큐와 동일(oof, best_primed). 덤프 없음.
- **관문(통과 조건)** = [4 ds 중 ≥3 ds에서 max(val R@20 α=.1, α=.2) > MES 단독 seed-1 val] ∧ [MES⊕ seed-1 val이 있는 모든 ds에서 max(val α=.1, .2) ≥ MES⊕ val]. 통과 시에만 seeds 2–5 실행(**별도 승인 단계, 이번에 발사하지 않음**). 미통과면 종료. test는 열람하지 않는다(게이트 스크립트는 result.json의 validation 필드와 융합 json의 valid_r20만 읽음).
- 비교값 출처(seed 1, validation, 읽기 전용): MES 단독 val = `<RUN_ROOT>/TABLE_FUSION/cells/mes__<ds>__s1.json` routes.base.valid_r20 (주표 MES run의 tf_fuse 산출물; baby .10451 / pantry .13091 / scientific .19193 / sports .12075). MES⊕ val = 같은 파일의 `f_combo_<w>` 경로 중 valid_r20 최대(= 융합 파이프라인이 val로 고른 가중치; tie → 작은 w): baby .10906 (w .2) / pantry .13954 (w .2) / scientific .19573 (w .2) / sports .12758 (w .1). 새로 계산하지 않았다(기존 산출물 4개 모두 있음). MES-F val = result.json `best_valid_primed.recall@20`(없으면 `best_valid_score`).
- 보고(관문 통과 여부와 무관): α 격자 {0.1, 0.2, 0.5, 1, 2} × 4 ds seed-1 val 표, MES 단독 val, MES⊕ val, 판정, 누적 GPU-h (`summary/posthoc_gate.json`, `logs/posthoc_gate.out`).
- sha256 (서버 `<RUN_ROOT>/study/MESF/code/`): `mesf_posthoc_setup.py` `730fbbcdd052f4a47c057e687fb1feeec05724bfc476e05ea440681afb6e9710`; `mesf_posthoc_gate.py` `3449d100a84afd7bf0f3d3adf33b52f90a69a2634fd45a70ef52ccaac275c4a9`; `run_posthoc.sh` `215394d5a34ef27560b0cb2d2279498e21e3c03ed679f448571dd46920eccb37`; `mesf_lane_ph.py` (mesf_lane.py에서 cap 합산 단계 목록에 posthoc 추가 1줄) `c163360b78304fb014a9fa75524ac39f2ae7ee2d444d301734c7d1c80e9ac66f`.


## 9. 관문 결과 (10:46 완료, validation만; 8 run 전부 DONE, `POSTHOC_DONE`)
seed-1 validation Recall@20 (MES-F, oof):
| ds | α=.1 | α=.2 | α=.5 | α=1 | α=2 | MES 단독 | MES⊕ (w) |
|---|---|---|---|---|---|---|---|
| Baby | .1077 | .0988 | .0891 | .0843 | .0821 | .1045 | .1091 (.2) |
| Pantry | .1382 | .1391 | .1347 | .1321 | .1295 | .1309 | .1395 (.2) |
| Scientific | .1996 | .1954 | .1853 | .1791 | .1758 | .1919 | .1957 (.2) |
| Sports | .1268 | .1178 | .1034 | .0975 | .0939 | .1208 | .1276 (.1) |
판정: 조건 1 (소α 최대 val > MES val) 4/4 충족. 조건 2 (≥ MES⊕ val 전 ds) 불충족: Baby −.0014, Pantry −.0004, Sports −.0008, Scientific만 +.0039. **관문 미통과 → 종료, seeds 2–5 소α 실행 안 함.** 사실: val은 α가 작을수록 단조 개선(Pantry만 .2 최대), 4 ds 중 3개에서 최선이 격자 하한 .1. 누적 GPU-h(관문 시점, 주 큐 진행 중 포함) 4.105.

## 10. H_F 판정 (2026-10-16 14:47 서버 시계, QUEUE_DONE 뒤 1회)
- 스크립트 `code/mesf_verdict.py` sha256 `db840988a72ba92167b8d5f51384a38926d4f07d745be0ae7ff8c93f50fef2b7`(결과 전 자체 검사 4 시나리오 통과; 첫 실행 sha ae446380…은 판정 출력 뒤 JSON 저장에서 int64 직렬화 오류만 나서 default= 한 줄 수정, 판정 로직 불변). 출처: MES-F result.json best_test(seed 1 = A_<ds>_a050_s1, seeds 2–5 = F_<ds>_s<k>), MSCA⊕·MES⊕ = TABLE_FUSION/fusion_cells.csv fused_r20(중복 행 동일 확인 후 제거).
- 결과(test R@20, 5-seed 평균; MES-F / MSCA⊕ / MES⊕, MES-F−MSCA⊕, Holm p, 이긴 seed): Baby .0890/.1095/.1090, −.0205, p_holm<.0001, 0/5 · Pantry .1154/.1175/.1186, −.0021, .0558, 1/5 · Scientific .1337/.1424/.1427, −.0087, .0015, 0/5 · Sports .1057/.1276/.1284, −.0219, <.0001, 0/5.
- **H_F = FAIL**: 양의 Holm 유의 0/4(≥3 필요), MES-F ≥ MES⊕ 0/4(4 필요). §2 실패 분기 (a)와 (b) 모두 해당: MES-F < MES⊕ 4/4이고 Baby·Scientific·Sports는 MSCA⊕보다 Holm 유의하게 **낮다**.
- 보조: N@20 MES-F > MSCA⊕ 1/4(Pantry .0535 vs .0532). MES-F − MES(미융합 base) R@20: Baby −.0158, Pantry +.0031, Scientific −.0067, Sports −.0155.
- 누적 GPU-h(MES-F 전체 run elapsed 합: α 선택·주 큐·사후 관문) 4.786 / cap 12.
- 원자료: 서버 `study/MESF/summary/hf_verdict.json`, `logs/hf_verdict.out`; 로컬 `study/results/mesf_hf_verdict.json|.out`, `mesf_posthoc_gate.json`.
