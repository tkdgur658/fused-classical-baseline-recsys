# PREREG_SOTAFILL — tab:sota 결측 seed 채우기 (저자 결정), 2026-10-16

## 0. 가설
없음(표 완성 작업). 새 가설 검정·새 주장 없음.

## 1. 결측 셀과 원인 조사 결과 (실행 전)
tab:sota에서 5 seed 미만인 8개 셀은 모두 마지막 블록(training-free post-hoc fusion, backbone ⊕ RP3β + ItemKNN-CBF)이다. 기본 비교 행(MSCA+MES, MSCA, … FREEDOM)은 84셀이 모두 5 seed다. 융합 셀은 "저장된 best checkpoint를 다시 읽어 만든 test 점수 dump의 기준 재현 gate"를 통과한 seed만 모은다(TABLE_FUSION: tf_queue.py, `|recalc R@20 − result.json R@20| ≤ 5e-4` 그리고 N@20 ≤ 5e-4, 고전 랭커 gate). 결측 seed는 "학습하지 않은 것"이 아니라 "gate 불통과"다.

| 셀(표) | 모델 | 데이터셋 | 결측 seed | 원인(서버 기록) |
|---|---|---|---|---|
| ⊕ 행 R@20/N@20 | COHESION | Scientific | 3, 4 | GRID FAIL: recalc .1319 vs json .1325/.1326 |
| ⊕ 행 | LGMRec | Pantry | 3, 5 | s3 dump gate(.0971 vs .0978), s5 fuse gate(R tol False, .1008 vs .1014) |
| ⊕ 행 | LGMRec | Scientific | 2, 4, 5 | GRID FAIL s2(.1265 vs .1251), s5(.1288 vs .1293); s4 fuse gate(.1269 vs .1277) |
| ⊕ 행 | LGMRec | Sports | 3 | dump gate(.1046 vs .1041) |
합계 8 seed(셀 8개 = 5 모델·데이터셋 쌍). 학습 run과 checkpoint는 8개 모두 존재(result.json SUCCESS).

원인: 코드가 스스로 문서화한 평가 확률성이다. LGMRec은 forward가 `F.gumbel_softmax`를 eval 모드에서도 뽑으므로(core/eval_state.py AUDIT 'LGMRec' 주석, models/lgmrec/src/lgmrec.py:118-130) 같은 checkpoint를 다시 평가하면 보고값과 다른 추첨값이 나온다. COHESION은 scoring state 재구성(pre_epoch_processing)이 np.random으로 사용자 그래프 패딩을 다시 뽑는다(AUDIT 'COHESION'). 학습 시 평가의 RNG 상태는 저장되지 않았다(result.json, 8개 모두 `deterministic: false`). 사전 확인(이 prereg 작성 전, 읽기 전용 추론 3회): LGMRec Scientific s2를 같은 GPU에서 두 번 재평가하면 recalc 0.126540 두 번 동일(고정 seed 재현), CPU에서는 0.125592(json 0.1251) — 즉 같은 checkpoint라도 난수 추첨에 따라 결과가 달라진다.

## 2. 실행 계획: 새 학습 run 없음
- 같은 seed를 다시 학습하면 기존 base 결과(표의 비융합 행)가 있는 seed의 replicate가 되어 "한 seed 한 run·replicate 금지" 규칙에 어긋난다. 따라서 학습은 하지 않는다.
- gate가 평가 난수 때문에 실패하므로, 훈련 설정을 바꾸지 않고 gate를 통과시키는 방법이 없다. (a) eval 난수를 끄면 base 점수 정의가 바뀌고, (b) gate 허용치를 바꾸면 gate가 바뀌고, (c) 난수 seed를 바꿔가며 통과할 때까지 재평가하면 best-of-N 선택이다. 셋 다 "같은 설정·같은 gate" 조건 위반이므로 이 8개 seed는 **실행하지 않는다(SKIP, 보고)**.
- 대신 저자 판단에 쓸 **진단 1건**만 실행한다: 8개 seed 각각(+ gate 통과 대조 2개: COHESION Scientific s5, LGMRec Pantry s1)의 저장 checkpoint를 20회 재평가(난수 seed 9001–9020)해 R@20/N@20의 추첨 산포와 gate 통과 확률을 측정한다. 추론만 하며 표에 들어가는 값을 만들지 않는다.

## 3. 사용 설정·스크립트(고정)
- 기존 seed와 동일한 worktree `<HOME>/P04_t2b2048L_wt` (git 351f83e05191f12fc6ae8b846f07bb6c4e07a65a), core/eval_state.py sha256 82f02fb1…, core/runner.py e9d0f499…, lgmrec.py ad1cf48e…, cohesion.py c28ba6a5….
- 모델 구성·checkpoint 로드·rank 계산은 GRID/code/s4r_dump2.py(sha256 23d47181d47f6887d3a543754e6df337931a1067112bacc4a901cc9fa499fb04)와 동일. 진단 스크립트는 그 복사본에 "난수 재설정 후 평가 반복"만 더한 것.
- gate(참조): TABLE_FUSION tf_queue.py sha256 6a83382981e8b046f326565c20b3b802f937ca2087c03f2cffde2b267f8a66e0, tf_fuse.py 2399e9d0ea237aa80fe8b4322dc19db973bb07bb270967f6156520d99481ad8d (바꾸지 않음).
- 진단 스크립트 sha256(동결): `code/sf_noise.py` d0e874f522a355d451b2832e5d752a51d69da7bc0e9f4a0cb5cd94fecd0f156b, `code/sf_queue.py` 2f71e5914992923dd1de2223c2ed91dbbfe68d5364e00326e838b5e79e77f31d.
- 서버 HOST_A, GPU 0, python project-env, nice 15, `<RUN_ROOT>/study/SOTAFILL/`에만 쓴다. 삭제 없음, disk1·/tmp 쓰기 없음.

## 4. 통과 조건과 예산
- 새 run이 통과해야 할 gate = 기존 GRID/TABLE_FUSION gate(위). 진단은 표 값을 만들지 않으므로 합격 판정 대상이 아니고, 통과 비율만 보고한다.
- 예산 상한 6 GPU-h. 작업 전 `used + 추정(0.03 h) ≥ 6`이면 `markers/<job>.SKIP_CAP`. 예상 총 0.1–0.3 GPU-h.
- 마커: `markers/<job>.DONE|FAIL|SKIP_CAP`, 종료 `markers/QUEUE_DONE|QUEUE_FAIL`.

## 5. 표 병합 영향 분석
tab:sota의 융합 블록에는 굵게·밑줄·유의 표식이 없고(build_table2_main.py 캡션: "rows carry no bold or significance marks"), 순위·표식은 비융합 행에서만 계산된다. 새 융합 행이 없으므로 병합 전후 표식은 동일하다는 것을 코드로 확인해 보고한다.

## 6. 실행 결과 (2026-10-16 17:21–17:23, QUEUE_DONE, 0.027 GPU-h, FAIL 0)
- 새 학습 run 0개, 표에 들어갈 새 값 0개. 진단 10건(결측 8 + 대조 2) 모두 DONE. 결과는 tables/src/data/sotafill_noise_diag.csv, 서버 SOTAFILL/out/*.json.
- 20회 재평가의 R@20 표준편차는 0.0002–0.0005, result.json의 보고값은 8개 중 7개에서 추첨 범위 안(LGMRec Pantry s5는 범위 밖: 보고 .1014 vs 추첨 .1005–.1012). 5e-4 허용치(R·N 모두) 통과 비율은 셀마다 30–90%, 4자리 일치 비율은 0–10%.
- 따라서 gate 불통과는 코드·설정 오류가 아니라 평가 난수 때문이며, 8개 seed는 실행하지 않았다. 표의 병합 전후 표식은 동일하다.
