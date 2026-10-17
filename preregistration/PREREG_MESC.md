# PREREG_MESC — 고전 미스 가중(MES-C): F 위 이득을 목표로 한 학습 (초안, 저자 승인 전 발사 금지)

작성 2026-10-16 18:5x, 설계 라운드 A(저자, 저자 논지 A "방법 논문" 결정의 위임). 측정 전 고정 초안. 서버 HOST_A **GPU0만**, cap 12 GPU-h.
관련: G0′ `study/G0P_MESC.md`, 관문 `study/GATE_MESC.md`, 선행 실패 `PREREG_MESF.md` §§9–10 (교훈: 고전 점수를 모델 점수에 넣지 않는다).

## 0. 문제 (확정 사실)
- F(Remark rem:protocol): 학습 점수 z(L) + w·z(s_C), s_C = z(z(RP3β)+z(ItemKNN-CBF)), w는 validation 선택. 학습 목적함수는 같은 backbone의 F 위 이득으로만 인정한다.
- MES⊕ − MSCA⊕ = −.0005…+.0011 (유의 0). MES-F(점수에 고전 오프셋) 실패: MSCA⊕ 대비 −.0021…−.0219.
- 관문 측정(GATE_MESC (b)): MES의 미융합 우위는 고전 top-20이 이미 맞히는 표적(A 층)에 몰려 있다(A +.0008…+.0035, B −.0010…+.0005). 그래서 F가 흡수한다.

## 1. 방법: MES-C (패키지 `mescw`; 기존 패키지 `mes_c`·덤프 태그 `mesc`와 이름 충돌 회피)
- 점수·모델·추론은 MES(XRES0) 그대로다. 고전 점수는 모델 점수에 들어가지 않는다. 바뀌는 것은 양성별 손실 가중 하나다.
  w(u,p) = 1 − γ·1[rank_C(u,p) ≤ 20],  w̃ = w / mean_batch(w)  (배치 자기정규화: 평균 보폭 불변, 배분만 이동).
- rank_C: 학습 쌍 (u,p)의 접힘 f(= r12.fold_of, MES 마스크 뷰와 동일)에 대해 MES-F oof 캐시 s_C^f(f 접힘 간선을 뺀 학습 간선만으로 만든 것; 쌍 자신의 간선 없음)를 쓴다. f 가시 이력 밖 아이템 중 순위다. 사전 계산은 `code/mesc_rank.py`가 한다.
- 적용 항: **BPR 주항과 마스크 뷰 엔트리 항 둘 다**, 같은 행에 같은 w̃. 근거(정식화 F): F가 인정하는 단위는 항이 아니라 **표적**이다. 엔트리 항은 접힘에서 가린 양성이 top-K 목록에 들어가는 사건으로, 테스트 표적의 학습 중 대리물이다. BPR 주항은 같은 양성의 전체 뷰 감독이다. 한쪽만 가중하면 다른 쪽이 F가 이미 얻는 표적에 용량을 계속 쓴다. 절제 `_bpr`·`_entry` config는 있으나 발사 대상이 아니다.
- **G-F5 유도 문장(원고용 후보)**: "Under F, a further hit on a target that the classical scorer already places in its top-20 is credited only for the fraction that F does not obtain from its classical summand, whereas a hit on a classical miss survives fusion at rate s_B; MES-C therefore weights each training positive by w = 1 − γ·1[rank_C ≤ 20], with rank_C computed out of fold so that the weight sees each positive as F would see a held-out target." 수치 근거: 인정 몫 비 w_A/w_B = (1−포착률)/s_B = .75/.71/.76/.66 (GATE_MESC (a)), 즉 γ* ≈ .24–.34. 구성요소–정식화 대응: 가중 = F(기준선 위 이득), oof 순위 = F의 표적 정의(학습 간선 밖), MES 마스크 뷰·엔트리 = 기존 기여(자기 메시지) 유지. 판정: G-F5는 이 문장으로 **충족 가능**하다. 단 G-K4(결합 자기 서술 0)는 "고전 순위를 가중에 쓴다"는 서술이 결합으로 읽힐 위험이 있다. 그래서 고전 점수가 점수·추론에 들어가지 않음을 손실 식으로 보인다.
- G0′(G0P_MESC.md): **부분 선행**. exact 선행(고정·훈련 없는 고전 랭커 순위로 학습 추천기의 양성을 재가중해 이후 융합에서 보완)은 발견되지 않았다. 최근접은 AdaBPR(IJCAI 2015)·BoostFM(IUI 2017)로, 동종 학습 구성원 부스팅 재가중이며 가중 갱신식은 미확인이다. 개념 최근접은 Learning to Complement Humans(2005.00582)다. 구조 근접은 BRIDGE·D3ER이고 둘 다 재가중이 없음을 확인했다. 기여 서술은 "부스팅식 보완 가중을 고정 고전 구성원에 적용하고 융합 후 이득으로 평가"로 제한한다.

## 2. 판정 (사전 고정)
- 주 기준 **H_C**: test R@20에서 두 조건을 모두 만족해야 한다. ① MES-C⊕ > MSCA⊕가 4 ds 중 **≥3**에서 Holm 유의(paired t over seeds 1–5, 양측 α .05, Holm은 튜닝 4 ds 위, 양의 방향). ② 평균 MES-C⊕ ≥ MES⊕가 **4/4**.
- MES-C⊕: MES-C 체크포인트에 `TABLE_FUSION/code/tf_fuse.py`(sha 2399e9d0…, **수정 없음**)를 적용한다. w 격자 {.05,…,1.5}에서 valid_r20 최대(동률은 작은 w)인 경로의 test 값을 쓴다. base 게이트 tol_ok가 필수다.
- 비교자(기존 셀, 새 학습 없음): MSCA⊕ = `TABLE_FUSION/fusion_cells.csv`의 MSCA 행이다. 이는 **별도 학습된 MSCA 융합 체크포인트** objx_s native run으로, Table I MSCA와 다른 체크포인트다. 편집자 ④ 동일 체크포인트 문제는 이 라운드에서 다루지 않고 H_F와 같은 비교자를 유지한다. MES⊕ = 같은 파일의 MSCA+MES 행(xres0 mestune run)이다. 중복 행은 동일해야 한다(assert).
- 보조(해석 없음): N@20 같은 방향 ds 수. MES-C 단독(route base) − MES 미융합(base_r20)은 |차| > MES seed 간 1 SD면 표시한다. 층별(A: rank_C ≤ 20, B: > 20) MES-C⊕ − MES⊕ 질량 차는 strata 덤프에서 낸다.
- 실패 분기: H_C FAIL → "상보성(고전 미스) 학습도 F 위 이득을 주지 못한다"를 사실로 보고하고 **사용자 재결정**(MES-D, 논지 B, 중단)을 받는다. 사후 튜닝은 하지 않는다(필요 시 별도 관문·승인).
- 예측: **P(H_C) = .3** (저자 사전 지정). 관문 기반 작성자 추정은 .15다(현실 상한 A2가 필요 이득에 1/4만 도달).

## 3. 절차 (한 번의 nohup, 규칙 기반 자동)
1. γ 격자: seed 1 × 4 ds × γ ∈ {.25, .5, .75} = 12 run (`code/run_all.sh` 1단계, `jobs_gamma.tsv`). (λ, K)는 주표 MES와 같다(Baby .3/40, Pantry .1/20, Scientific .2/40, Sports .1/80). 학습·평가 명령은 MES-F와 같다(canonical_v2, best_primed, --save-ckpt). 각 run 뒤 tf_fuse(CPU).
2. γ 선택(`code/mesc_gamma_select.py`, **validation만**, test 미열람): ds별 MES-C⊕ seed-1 valid R@20 최대 γ(동률은 작은 γ)를 고르고, 이후 동결한다. F가 목표이므로 선택 지표는 융합 점수의 validation이다.
3. **K1 pred_kill**: 최선 γ의 MES-C⊕ seed-1 val > MES⊕ seed-1 val(Baby .10906, Pantry .13954, Scientific .19573, Sports .12758)인 ds가 3개 미만이면 `summary/KILL_K1`, `markers/KILLED_K1`을 남기고 종료한다. seed 2–5는 돌리지 않는다.
4. K1 PASS 시 주 큐: 선택 γ로 seed 2–5 신규 16 run을 돌린다. seed 1은 γ 단계 run을 재사용한다(**복제 금지**, 융합 json 복사). 모든 seed에 tf_fuse와 strata 덤프(`mesc_gate_dump.py`)를 적용한다. 종료 마커는 `markers/QUEUE_DONE|QUEUE_FAIL`이다.
5. 판정 1회: `code/mesc_verdict.py`(자체 검사 5 시나리오 후 실행) → `summary/hc_verdict.json`. 결과는 1회만 낸다.
- 비용: 추정 6.9 GPU-h(γ 2.76 + 주 4.16, 융합 CPU 포함). cap **12 GPU-h**는 두 단계 DONE/FAIL 경과 합 기준이며, used+est ≥ 12면 SKIP_CAP이다. **GPU0만** 쓰고 동시 ≤ 2 프로세스, nice 15다. OOM → FAIL(재시도 없음, "—").
- 마커: `study/MESC/markers/{gamma,main}/<id>.RUNNING|DONE|FAIL|SKIP_CAP`, `markers/GAMMA_DONE`, `KILLED_K1`, `QUEUE_DONE|QUEUE_FAIL`.
- 발사(부모가): `study/GPU0_RESERVED` 생성(부모) → `cd <RUN_ROOT>/study/MESC && nohup bash code/run_all.sh > logs/run_all.out 2>&1 &`. `budget.txt`·`jobs_*.tsv`는 발사 시 생성된다(현재 없음).

## 4. 구현·점검 (CPU, 학습 0회; 서버 `<RUN_ROOT>/study/MESC/`, 로그 `tests/*.out`, 로컬 사본 `study/results/mesc/`)
- 워크트리: `MESC/wt` = `study/MES9/wt`의 `cp -a`(MES-F와 같은 원본)에 `models/mescw/`만 추가. 기존 파일 수정 0. 매니페스트·config 28개는 `code/mesc_setup.py`가 만든다.
- (b) γ=0 재현(`tests/unit.out`): 첫 배치 손실 13.691152572631836이 XRES0와 비트 동일하다. 15개 파라미터 기울기는 비트 동일(max abs diff 0.0)이다. 이는 강제 가중 경로(`_force`)와 일반 경로 모두에서 성립한다. 기존 MES seed-1 Baby 체크포인트를 MESCW(γ .5)로 추론하면 test .10396(저장 .1041), valid .10445(저장 .1045)다. missing/unexpected는 0이다.
- 경로 도달(`tests/path.out`): 실제 calculate_loss 안에서 가중 BPR·가중 엔트리 호출이 각 1회 일어난다(apply 설정대로). γ .5에서 첫 배치 기울기가 γ 0과 다르다(최대 7.6e-6). 가중 BPR은 수식 직접 계산과 차 0.0이다(`tests/step.out`). γ .5/.75의 4스텝 손실·기울기는 유한하다. w̃ 평균은 1.0이다. 배치 rank_C ≤ 20 비율은 .08–.13이다.
- (a) 누설(`tests/leak.out`): 캐시 sha는 MES-F 메타와 같다(train-only 캐시, MES-F 누설 검사 통과분). 무작위 3000쌍 × 2 ds를 독립 경로(R12.build_folds 가시 이력 + 접힘 캐시)로 재계산하면 3000/3000 일치한다. 쌍 자신의 간선이 가시 이력에 있는 경우는 0이다. 참고로 in-sample(full 캐시)이면 rank_C ≤ 20 비율이 Baby .71 / Scientific .86으로 부풀고, oof는 .11 / .26이다. 스크립트 단언으로 접힘 쌍의 가시 이력 포함 0, 학습 간선 수 일치를 확인했다.
- (c) 합성 역검증(`tests/toy*.out`, 실제 `MESCW.pair_weight`·`cal_bpr_loss`, oof 2접힘 순위, 융합 w는 validation 선택, 장난감 seed 3개 평균):
  | 설정 | γ 0 융합 test | γ .25 | γ .5 | γ .75 | val 선택 γ |
  |---|---|---|---|---|---|
  | dim 4, content 2.0 (가중이 돕는 경우) | .4521 | .4600 | **.4633** | .4550 | .5 |
  | dim 4, content 1.0 | .3800 | .3821 | .3879 | .3906 | .25 |
  | dim 8, content 1.0 | .3883 | .3904 | .3881 | .3838 | .5 |
  | dim 2, content 2.0 | .3414 | .3386 | .3402 | .3350 | 0 대비 하락 |
  가중이 돕는 합성 경우가 존재한다(+.011, validation이 같은 γ를 고름). 효과는 용량 제한·강한 고전 복원 성분 체제에 한정된다.
- 남은 위험(미점검): mescw 패키지로 main.py 학습 1 epoch GPU 스모크는 하지 않았다(학습 금지 범위). 첫 γ job의 tf_fuse 게이트(tol_ok)가 사실상 스모크다. 실패 시 FAIL 마커로 멈춘다.

## 5. 스크립트 sha256 (초안 시점 2026-10-16 18:5x 서버; 발사 직전 재확인, 변경 시 여기 기록)
| 파일 (서버 `<RUN_ROOT>/study/MESC/`) | sha256 |
|---|---|
| `code/mescw.py` = `wt/models/mescw/src/mescw.py` (MESCW 클래스) | `dd52f67decbabc3f647c75a7e467f38f153b4c3029cbd51b182b5773530ca3f7` |
| `wt/models/mescw/run.py` | `055b6b2b715ccf02c783478a448c9569388d9d2f094354a20e2ffc504405e7d8` |
| `wt/models/mescw/manifest.json` | `57086a8dec4201edb3b9e638ff11ed0ba204b31d66f06cd8905df01c5d6d2192` |
| `wt/models/mescw/configs/*.yaml` (28개, `cat *.yaml \| sha256sum`) | `7b7b96617c97f11b20258e03d8b92281f84de5066b71f94aa2dbe5948a734e80` |
| `code/mesc_setup.py` | `84e46cd3c4013a91b4747f85781467c7c56cbf45fd9fa40b96abe898abbe8234` |
| `code/mesc_rank.py` (rank 캐시) | `7b191ff12841d65ee14399b2fcacbbf59c289d5394a7c6384d5ae9c7422ebba8` |
| `rank_cache/<ds>_rankC_oof.npz` baby/pantry/scientific/sports (앞 12자) | `453957eb820c` / `2c0ee22be0d2` / `dfeff4b4f6e0` / `a6a699698709` |
| `code/mesc_lane.py` | `736e85d87d9152e1620115b88a2367cbaadd60e7b1907020827cf0c94b5fe8d7` |
| `code/mesc_gen_gamma.py` | `63f788fa0a599bf87ed339aad44a82bcb79b816b45bf84449d018cdfed438b81` |
| `code/mesc_gamma_select.py` (선택 + K1, 자체 검사 4) | `c1a7587944aa71837b50089341d1ef6fd4a703432aee9434d1fe3062f49e1c38` |
| `code/mesc_gen_main.py` | `8ceb2642ced9718ae35232f1560ef2198041c950cb626d17b7b8d24cd6bd86d8` |
| `code/run_all.sh` | `a522aad0d9d867dd12c00a59c8d7c1e25b8944a1fa19183fd29b58e097756f37` |
| `code/mesc_verdict.py` (H_C, 자체 검사 5) | `fc036d40648c23f69ef53abf89ab7d36af2e169a7970a7d16201657b47271ed3` |
| `code/mesc_tests.py` (점검) | `d09fa637e76c27b47b41eac0d5b981729b1d48210f5e1ef064ded2619d82e2b3` |
| `code/mesc_gate_dump.py` (tf_fuse + 타깃별 배열) | `188e97e0fe4788fcbd564fdf3b40447a404f2d97e8dfc58c8e1084addd15b4cd` |
| `code/mesc_gate_stats.py` (관문) | `4f21b5bb5be387849eea8d5fa4fcae04b696afbc050daee97d5e95c4fc4b5365` |
| `TABLE_FUSION/code/tf_fuse.py` (불변, 참조) | `2399e9d0ea237aa80fe8b4322dc19db973bb07bb270967f6156520d99481ad8d` |

## 6. 운영 기록
- 18:0x–18:4x: CPU만 썼다. 관문 덤프 40개(`gate/dumps`), rank 캐시 4 ds(`rank_cache`), 점검(`tests`)을 만들었다. 학습·GPU 사용 0회. `GPU0_RESERVED`는 만들지 않았다. MES10(study/MES10)은 건드리지 않았다.
- 편차: 원격 셸이 zsh여서 toy 첫 일괄 실행의 인자 분리가 실패했다. 잘못된 출력 3개는 `tests/junk_zsh_split/`로 이동했다(삭제 아님). 이후 개별 실행으로 재실행했다.

## 발사 기록 (2026-10-16 18:23 서버 시계, 저자)
- 관문 (a) 재배치 상한이 필요 이득에 1/4(Sports)만 도달 → pred_kill .85로 기록. 규칙상 측정 없이 탈락이지만 K1 비용(≈2.8 GPU-h)이 작아 **K1만 "관문 감사"로 발사**: audit = {"type": "gate_a", "pred": "fail"}.
- K1 PASS(≥3 ds에서 MES-C⊕ seed-1 val > MES⊕ val) → 관문 오판으로 기록하고 seeds 1–5 주 단계 진행(run_all.sh가 자동 진행). K1 FAIL → MES-C 종료, 관문 적중 기록.
- 설계 선택 승인: γ 1-파라미터, BPR·entry 동일 행 가중, 배치 자기정규화, γ 선택 = MES-C⊕ seed-1 validation, MSCA⊕ 비교자 = H_F와 동일한 별도 학습 MSCA 융합 체크포인트.
- study/GPU0_RESERVED 생성 → MES10의 새 job은 GPU1에만. 발사: cd <RUN_ROOT>/study/MESC && nohup bash code/run_all.sh > logs/run_all.out 2>&1 &

## H_C 판정 (2026-10-16 21:4x 서버 시계, QUEUE_DONE 뒤 1회)
- 스크립트 code/mesc_verdict.py sha256 fc036d40648c23f69ef53abf89ab7d36af2e169a7970a7d16201657b47271ed3 (자체 검사 5 시나리오 통과).
- test R@20 5-seed 평균 MES-C⊕ / MSCA⊕ / MES⊕, MES-C⊕−MSCA⊕, MES-C⊕−MES⊕, p, p_holm, wins: Baby .1087/.1095/.1090, −.0008, −.0002, .363, .726, 2/5 · Pantry .1192/.1175/.1186, +.0017, +.0006, .161, .484, 4/5 · Scientific .1431/.1424/.1427, +.0007, +.0005, .526, .726, 4/5 · Sports .1292/.1276/.1284, +.0017, +.0008, .0032, .0128, 5/5.
- **H_C = FAIL**: MSCA⊕ 대비 양의 Holm 유의 1/4(Sports; ≥3 필요), MES-C⊕ ≥ MES⊕ 3/4(4 필요; Baby −.0002).
- 보조: N@20 MES-C⊕ > MSCA⊕ 3/4. MES-C 단독 − MES 단독(R@20) Baby −.0005, Pantry .0000, Sci −.0005, Sports −.0001(모두 1 SD 이내).
- 관문 감사: 관문 (a) 재배치 상한이 "1/4(Sports)만 도달"로 예측했고, 결과도 Holm 유의 1/4 = Sports다 → **관문 적중**. K1은 pass_thin(3 ties ≤2e-4).
- 원자료: 서버 MESC/summary/hc_verdict.json, logs/verdict.out; 로컬 study/results/mesc/hc_verdict.json|.out.
