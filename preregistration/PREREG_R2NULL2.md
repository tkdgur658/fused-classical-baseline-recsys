# PREREG_R2NULL2 — 사용자별 오라클 여유(g_O)의 강화 귀무: 재표집 + 이력 길이 층화 + 승자 지속성 (측정 전 고정)

작성 2026-10-15, PREREG-study. 상태: **초안 = 동결 대기** (스크립트 sha256 기입 후 실행; CPU ≤1 h, GPU 없음). 근거: DESIGN_G2M.md §1.2 N, 기존 `R2/PREREG_R2NULL.md`·`r2null.py`(sha 7916689c…, 결과 `R2/results/r2null.md`: 9/9 below), `R2/r2s1.py`·`r2z.py`의 로더. 코드는 `r2s1.py`/`r2z.py`를 import해 쓰며 복사·수정하지 않는다.

## A1. AMENDMENT (2026-10-15 20:2x KST, 저자 결정, **결과 전**; 본문과 충돌하면 A1이 우선)
1. **초안대로 승인**: N2(B=1000), N3(B=200), W(순열 B=1000), 판정 규칙 `g_O ≤ hi(N2) ∧ g_O ≤ hi(N3) ∧ p_W > 0.0056`, CPU 상한 60분. 유효한 valid 덤프가 없는 데이터셋은 W "—"(N2·N3만으로 판정, "W 미평가" 표기).
2. **구현 확정(스크립트 작성 중 발견한 초안과의 차이, 판정 규칙은 불변)**: ① `r2z.py`에는 valid 덤프 로더가 없다(r2z는 test 목록으로 validation 양성을 평가). 그래서 `r2null2.py`가 `r2s1.py`(import, 읽기 전용)만 쓰고 `<L>__<ds>__s<s>__valid.npz`(dumps, W1/dumps, W2/dumps 순)의 top20·pu·pi로 vL을, 같은 validation 양성에 대한 고전 목록(C)으로 vC를 계산한다. r2z는 import하지 않는다. ② 다중 양성 사용자의 "한쪽만 맞힘"은 recall 차이의 부호(≠0)로 정의한다. ③ 민감도(십분위·5분위)는 N2에만 B=200으로 적용하고 N3·W는 기본 층화 결과를 쓴다(판정 제외, 판정이 바뀌면 별도 표시). ④ 난수: `SeedSequence([20261015 + 1000·ds번호(0–8, DSS 순) + seed, kind])`, kind N2=2, N3=3, W=4(seed 인덱스 0, 순열 1개를 모든 seed가 공유), 민감도=20+분위수. ⑤ N3 미끼 재추출은 최대 30회, 그래도 학습 이력과 겹치는 미끼는 결과 JSON에 `n3_unresolved_decoys_mean`으로 보고. ⑥ 스레드 ≤8(`OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`), `nice -n 15`, `CUDA_VISIBLE_DEVICES=`, 래퍼 `timeout 4200`, 스크립트 내부 60분 cap은 ds 사이에서 확인(초과 시 남은 ds "—"). ⑦ 마커 = `<RUN_ROOT>/study/R2NULL2/DONE` 또는 `FAIL` (본문 §5의 `R2NULL2_DONE|FAIL` 이름을 대체). ⑧ 현재 서버에서 valid 덤프가 확인되는 ds = baby, pantry, scientific, sports, clothing, instruments (arts, office, microlens는 `dumps`, `W1/dumps`, `W2/dumps`에 없음 → W "—" 예상).
3. **시작 전 검사(결과 수치 미출력)**: `--selftest` 통과(합성 12회씩, 규칙이 noise 모형 1.00 noise 판정, struct 모형 0.00), `--smoke`(baby seed 1) 게이트 Δ = 0.0, 평가 사용자 19445, 층 5개 [3],[4],[6–9],[10–19],[≥20] 크기 5998/6192/4497/2305/453 (200명 미만 병합 규칙 적용 후), valid 덤프 존재.
4. **첫 발사 중단 기록(20:26–20:28, 로더 버그, 판정 규칙·B·난수 불변)**: 첫 발사(sha 4367bd4e…)에서 `<L>__valid.npz`의 users가 test 덤프보다 6명 적은 ds(pantry 13093 대 13099 등)에서 정렬 assert가 실패했고 코드가 이를 W "—"로 삼켜 버렸다(자동 대체 금지 원칙 위반). 약 20초 만에 내가 해당 프로세스를 직접 종료했고(마커 `FAIL_killed_by_agent_2027_wcatch_bug`로 보존), 부분 결과는 `results_aborted_2027/`에 보존했다(baby·pantry·scientific의 N2/N3 분류와 baby의 W를 이미 봤다는 사실을 기록한다). 수정 = ① valid 덤프의 users(validation 양성이 있는 사용자 부분집합)를 test 사용자 색인으로 매핑해 정렬, ② W 로더 오류는 삼키지 않고 FAIL로 전파(valid 덤프 자체가 없는 ds만 W "—"). 수정 sha = 위 표. 이 외 변경 없음. 두 번째 발사는 같은 규칙·같은 난수로 처음부터 다시 돌린다.

## 0. 기존 귀무의 약점 (설계 이유, 측정값은 기존 결과 파일에서 인용)
- 기존 R2NULL은 이력 길이 십분위 안에서 C의 사용자 recall을 순열로 섞어 L과 독립 짝지음한 뒤 max를 취했다. 결과: 9/9 데이터셋이 **below**(g_O +14.9~+36.4% 대 귀무 평균 +58.9~+85.1%; L–C 사용자 recall Spearman .78–.90). 독립 짝지음 귀무가 실제 L–C 상관을 무시해 g_O의 2–5배를 만들어 내므로 "잡음으로 설명된다"가 **공허하다**(above가 사실상 불가능).
- 실제 dump에서 test 양성이 1개인 사용자는 baby 92%, sports 92%, clothing 96%, microlens 96%, pantry·scientific·instruments·arts·office 100%이다(서버 `dumps/<ds>/<L>__<ds>__s1.npz`의 `pu`로 계산). 사용자 recall은 사실상 이진이고, **"사용자별 test item 재표집"은 사용자 안에서는 퇴화**한다(양성이 1개뿐이라 복원 추출해도 같다). 그래서 아래 N2/N3는 재표집 단위를 사용자와 test item 두 층에서 정의한다.

## 1. 가설
- H_noise(데이터셋 단위): g_O는 (a) 독립 잡음 추정량 둘의 max로 나오는 크기를 넘지 않으며(N2, N3), (b) 사용자별 승자(L vs C)가 validation에서 test로 이어지는 지속성이 없다(W).
- H_struct: g_O가 N2 또는 N3의 상한을 넘거나, W가 유의하게 양(승자 지속)이다.

## 2. 데이터와 정의 (R2-S1과 동일)
- 9 ds(baby, pantry, scientific, sports, clothing, instruments, arts, office, microlens), seed 1–5, L/C 선택은 `results/r2s1.json`의 Ltag/Ctag(현재 L = xre, C = combo 또는 rp3b), 평가 사용자 = test 양성이 있는 사용자, 사용자 recall@20 r^L_u, r^C_u, 오라클 r^O_u = max.
- g_O = mean_s R_O,s / mean_s R_L,s − 1 (seed 5개 평균비). **게이트**: 재계산한 R_L, R_C, R_O가 r2s1.json과 |Δ| ≤ 1e-6, g_O 일치. 위반 → 그 ds는 `R2NULL2_GATE_FAIL`로 표시하고 판정에서 "—"(분모에서 제외하지 않고 실패 수로 계산: 9개 중 판정 불가는 N_noise에 넣지 않는다).
- **이력 길이 층(사전 고정 경계)**: 평가 사용자의 학습 degree를 [1], [2], [3], [4–5], [6–9], [10–19], [≥20]의 7층으로 나눈다. 평가 사용자가 200명 미만인 층은 다음 높은 층에 합친다(마지막 층이면 이전 층에). 민감도: 기존 십분위, 5분위.

## 3. 귀무와 통계량
각 귀무는 데이터셋 × seed마다 독립 난수(고정 기본 시드 20261015 + 1000·ds 번호 + seed)를 쓰고, 복제 B회의 데이터셋 값 g_null,b = mean_s R_null,s,b / mean_s R_L,s − 1 (g_O와 같은 정의)의 [2.5%, 97.5%] = [lo, hi]로 요약한다.
- **N2 (층화 독립 부트스트랩, B = 1000)**: 각 층 안에서 L의 사용자 recall 벡터와 C의 벡터를 **서로 독립으로** 사용자 복원추출해 같은 층 크기로 짝지은 뒤 g를 계산한다. 순열(기존)과 달리 주변분포의 표본 변동도 포함한다. 분모 R_L도 그 복제의 값으로 다시 계산한다(비율 g의 변동 포함).
- **N3 (test item 재표집 = 미끼 귀무, B = 200)**: 각 사용자의 L·C top-20 목록은 그대로 두고 test item을 **같은 층 사용자들의 test item 분포에서 복원추출한 미끼 item**으로 바꾼다(사용자 자신의 학습 이력에 있는 item이 뽑히면 다시 뽑음). 미끼 item이 L·C 목록에 있는지로 recall을 새로 계산한다. 목록의 구조(L–C 겹침, 인기 편향)는 보존하고 사용자–item 신호는 제거하는 귀무이다. 다중 양성 사용자(전체의 0–8%)는 양성마다 독립으로 미끼를 뽑는다.
- **W (승자 지속성, 순열 B = 1000)**: 같은 사용자의 validation과 test 둘 다에서 L, C 중 정확히 한쪽만 맞힌 "이중 불일치" 사용자에 대해, validation 승자와 test 승자가 일치하는 비율 A를 계산한다. 귀무 = validation의 (r^L, r^C) 쌍을 같은 층 안의 다른 사용자와 섞은 뒤 A를 다시 계산(전체 L 대 C 우세와 층 효과는 보존). seed 5개의 A는 평균하고, 귀무도 seed마다 같은 순열 인덱스를 써서 평균한다. 단측 p_W = (1 + #{A_null ≥ A}) / (B + 1). validation 쌍은 `r2z.py`의 valid 덤프 로더(라벨 val)를 import해 얻는다. 어느 (ds, seed)에 valid 덤프가 없으면 그 ds의 W는 "—"이고 판정은 N2·N3만으로 하되 "W 미평가"를 표기한다.

## 4. 동결 판정 규칙
- 데이터셋 분류(N2, N3 각각): above = g_O > hi, inside = lo ≤ g_O ≤ hi, below = g_O < lo.
- **데이터셋 단위 "headroom = noise" ⇔ [g_O ≤ hi(N2)] ∧ [g_O ≤ hi(N3)] ∧ [p_W > 0.05/9 = 0.0056]** (9개 ds에 대한 Bonferroni).
- **전체**: N_noise = 위를 만족하는 ds 수. "headroom = noise, 9/9 유지" ⇔ N_noise = 9. N_noise ≥ 7 이면 "대부분 유지(예외 ds 명시)", N_noise ≤ 6 이면 "유지되지 않음 — 구조적 여유 신호 있음"으로 기록하고 사실 N을 본문에서 삭제 또는 해당 ds 범위로 제한한다(결과를 본 뒤 규칙을 바꾸지 않는다).
- 층화(§2 7층)가 기본, 십분위·5분위는 민감도(판정 제외; 판정이 바뀌면 보고에 별도 표시).
- 반드시 같이 보고할 진단: (i) 공허도 진단 N2 평균 / g_O와 N3 평균 / g_O (기존과 같은 over-generation 여부), (ii) W의 A와 이중 불일치 사용자 수 k, (iii) seed별 분류, (iv) 층별 g_O와 층 크기.
- **해석 한계(사전 명시)**: N2·N3는 독립·무신호 기준이라 g_O보다 크게 나오는 것이 보통이므로 단독으로는 "잡음" 판정을 약하게만 지지한다. 강한 근거는 W(승자 지속성이 없음)이다. W는 사용자별 숨은 능력의 이질성 때문에 큰 n에서 A가 0.5를 약간 웃돌 수 있다(합성에서 n = 60000, A = .540, p = .07); 이를 막기 위해 층 안 순열과 Bonferroni 기준을 쓴다. test 양성이 1개인 데이터에서는 이진 결과의 동시분포 이상을 식별할 수 없으므로 N 사실은 "이 프로토콜에서 구분되지 않는다"이지 "구조가 없다"가 아니다. 본문 문구도 그렇게 쓴다.

## 5. 비용과 실행
- CPU 1 프로세스, nice 15, 스레드 ≤16, GPU 없음(`CUDA_VISIBLE_DEVICES=`). 서버 폴더 `<RUN_ROOT>/study/R2NULL2/`(코드 `code/`, 결과 `results/r2null2.json|md`, 로그 `logs/`, 마커 `R2NULL2_DONE|FAIL`)에만 쓴다.
- 비용 추정: 기존 `r2null.py`가 9 ds × 5 seed × (순열 1000 + 민감도 400)에 332.6 s였다. N2(복원추출 1000, 사용자 최대 98k)는 같은 자릿수, N3(200 × 목록 20 비교)와 W(순열 1000)는 합쳐 그보다 작다. 추정 합계 ≈ 15–25분, **cap 60분**. 60분이 지나면 실행을 중단하고 그 ds까지의 결과만 부분 보고(남은 ds는 "—").
- 시작 전 `--smoke`(baby seed 1, B = 20): 층 수·사용자 수·게이트 Δ만 출력하고 g_O·null·판정 수치는 출력하지 않는다. `--selftest`(합성) 통과가 선행 조건.
- 실패 처리: 게이트 위반·로더 오류는 `R2NULL2_FAIL` + traceback, 수정 후 재실행은 새 번호(R2NULL3)로 하며 판정 규칙은 바꾸지 않는다. 재실행으로 난수 시드를 바꿔 판정을 고르지 않는다.

## 6. 스크립트 경로와 sha256 (동결 시 기입)
| 파일 | 경로 | sha256 |
|---|---|---|
| `r2null2.py` | 서버 `<RUN_ROOT>/study/R2NULL2/code/` (r2s1.py를 `<RUN_ROOT>/R2/`에서 import) | `2b3e11de1b4f6eaeab5f2d1b39e0713a2bafda5855dd486903de88483d1d3e50` |
| `r2null2_run.sh` (nice·timeout 래퍼) | 〃 | `abc3b4eaa33a085d8cd8c4c31a9b2fbea96a0c6f78a28a0c3b2898ee1af4f0ac` |
| `r2s1.py` (import 대상, 읽기 전용) | `<RUN_ROOT>/R2/` | `1bd74ca9aa554b10aa0f17baaa7fed1e6a5fcadb5d9cf2e70bd97b6545d0891e` |
| `r2z.py` (A1.2: import하지 않음, 참고만) | `<RUN_ROOT>/R2/` | `e5aa219d16deb92ee941cfc6559e1c7f132ed3bea71e888e59990206ae4329f7` |
| `results/r2s1.json` (L/C 선택·게이트 기준) | `<RUN_ROOT>/R2/results/` | `a0a48d97c1d041611245f22f3a817643529de35a263797e437652e1b10d5f836` |
| 합성 검증 `synth_r2null2.py` (로컬 `study/`) | — | `cc6de5f033ce09a621da7f3cfd4888f1abbd996b48ff25fcd2a500a234e5bf4e` (아래 §7 출력 전 실행분의 규칙 상수 α = 0.05/9 포함) |

## 7. 합성 역검증 (로컬 실행, 출력 전문 = `synth_r2null2.out`)
설정: 사용자 n, 이력 층 7개, 공유 잠재 능력 q_u(Beta, 층 의존)로 L·C test/validation 결과를 생성, test 양성 1개(실제 dump와 같은 퇴화 조건), 목록은 정답을 확률 p로 포함하고 나머지는 무작위 item. `noise` = 같은 q_u 아래 L·C 결과가 독립(L이 C보다 평균 우세), `struct` = 사용자별 고정 승자(지속).
```
noise {'g': 0.4152, 'N2_mean': 0.6523, 'hi2': 0.6834, 'N3_mean': 0.9513, 'hi3': 1.0135, 'k': 1873, 'agree': 0.5104, 'p': 0.6833, 'noise': True}
struct {'g': 0.8377, 'N2_mean': 0.8544, 'hi2': 0.8926, 'N3_mean': 0.9476, 'hi3': 1.0142, 'k': 4047, 'agree': 0.9498, 'p': 0.001, 'noise': False}
P(rule says NOT noise | noise)  = 0.00  (40 reps, n=12000)
P(rule says NOT noise | struct) = 1.00  (40 reps, n=12000)
noise: N2_mean/g_O = 1.59, N3_mean/g_O = 2.26
W agree (noise, global L>C advantage kept) = 0.540, p = 0.071
SYNTH_R2NULL2 OK
```
해석: (a) 순수 잡음에서 N2/N3는 g_O의 1.6–2.3배를 만들어(기존 R2NULL과 같은 over-generation 양상) 단독으로는 변별력이 낮고, (b) **구조(승자 지속)가 있을 때 N2도 거의 상한에 붙지만(g .838 vs hi .893, 단독으로는 'inside') W가 A = .95, p = .001로 분리**해 규칙 전체가 power 1.00, 위양성 0.00이다(40회씩, n = 12000). (c) 큰 n에서 W의 잠재 능력 이질성 부풀림(A .540, p .071)은 α = 0.0056 안에 들어온다. 한계: 합성은 단일 모형 하나이며 40회 반복이라 위양성률의 상한(95% ≈ 7%)만 보장한다.

## 8. 보고 형식 (동결)
`results/r2null2.md`: ds별 한 줄(g_O, N2 [lo,hi]·분류, N3 [lo,hi]·분류, W의 k·A·p, 데이터셋 판정), 맨 위에 N_noise/9, 규칙 §4의 결론 문구 한 줄, 진단 4종. 보고서에는 §4 "해석 한계"를 생략하지 않는다.
