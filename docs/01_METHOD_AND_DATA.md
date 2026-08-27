# 방법과 데이터

## 구현 범위

논문의 AR·MLR 비교와 예측오차 및 경제적 손실을 함께 고려하는 학습 구조를 구현하고자 했습니다. 정확한 표본 날짜, solar zone, price node와 timestamp 결합 방식은 논문에서 확인하기 어려워 다음 조건을 사용했습니다.

- GEFCom2014 Zone 1
- source-naive timestamp를 공통 UTC index로 해석
- Sydney 현지시간 09:00–20:00
- 관측된 POWER·DA·RT만 사용
- fitted value, evaluation prediction과 oracle commitment를 `[0,1]`로 제한
- gap percentage의 분모는 bounded oracle profit

논문에는 train 300일·test 100일이라는 설명과 train 3개월·test 1개월이라는 설명이 함께 있습니다. 여기서는 전자의 크기에 맞추되 정확한 날짜는 데이터 범위 안에서 정했습니다.

## 전처리

`operational_preprocessing.py`의 순서는 다음과 같습니다.

1. `predictors15.csv`의 Zone별 `01,...,23,00` 반복을 확인합니다.
2. 전체 시간열에서 accumulated SSRD·TSR을 차분합니다.
3. Timestamp를 UTC로 간주하고 `Australia/Sydney`로 변환합니다.
4. Sydney 현지시간 09:00–20:00을 선택합니다.
5. UTC timestamp를 기준으로 POWER, DA LMP, RT LMP를 합칩니다.
6. Sydney 현지 날짜로 train과 test를 나눕니다.

현재 표본은 다음과 같습니다.

| 용도 | Sydney 현지 날짜 | 일수 | 행 수 |
|---|---|---:|---:|
| AR 직전 POWER 이력 | 2013-03-26 | 1 | 12 |
| Train | 2013-03-27–2014-01-20 | 300 | 3,600 |
| Test | 2014-01-21–2014-04-30 | 100 | 1,200 |

AR 이력 날짜는 첫 train day의 입력에만 쓰며 target에는 포함하지 않습니다. 각 train·test 날짜는 현지시간 09시부터 20시까지 정확히 12행입니다. DST 날짜를 직접 지정하지 않고 timezone database의 변환 결과를 사용합니다.

## AR

09시부터 20시까지 target hour별 모형 12개를 적합합니다. 현재 lag는 전날의 12개 daylight POWER이며 가장 최근 값부터 배열합니다.

$$
\hat S_t=\alpha^h+\sum_{l=1}^{12}\beta_l^h S_{t-h-l}, \qquad t\in U_h.
$$

시간대별 design matrix는 intercept를 포함해 13열입니다. Test는 한 날짜의 12개를 모두 예측한 뒤에만 그 날짜의 actual POWER를 다음 날 history에 추가합니다.

## MLR

MLR은 모든 시간대를 하나의 표로 합쳐 적합합니다. Design matrix는 14열입니다.

- intercept
- train 통계로 표준화한 `dSSRD`, `dTSR`
- 09시를 기준으로 한 hour indicator 11개

Hour indicator는 표준화하지 않습니다. Conventional AR·MLR은 train fitted value가 `[0,1]`에 들어오는 bounded LAD로 적합합니다.

## Proposed 학습과 이진변수 축소

Proposed 모형은 Conventional 모형과 같은 입력을 쓰고 학습 목적함수만 바꿉니다.

$$
W_1\frac{\Pi^{\mathrm{oracle}}-\Pi(x)}{\Pi^{\mathrm{oracle}}}
+W_2\frac{1}{N}\sum_i |S_i-x_i|,
\qquad W_1=1,\;W_2=20.
$$

각 관측에서 약정량을 $x$, 실제 발전량이 약정량보다 큰 부분을 $y'$, 약정량이 실제 발전량보다 큰 부분을 $y''$라고 두면 다음 균형식이 성립합니다.

$$
x+y'-y''=S.
$$

코드에서는 각각 `x_slice`, `y_plus_slice`, `y_minus_slice`로 표시합니다. $y'$와 $y''$를 함께 $\delta$만큼 증가시켰을 때의 목적함수 변화는 두 변수의 목적계수 합에 $\delta$를 곱한 값입니다.

- 두 목적계수의 합이 양수이면 공통 증가가 목적함수를 악화시키므로 최적해에서 상보조건이 성립합니다.
- 두 목적계수의 합이 0이면 공통 부분을 제거해도 약정량과 목적값이 변하지 않으므로 binary가 필요하지 않습니다.
- 두 목적계수의 합이 음수이면 상보조건을 강제하기 위해 Big-M binary를 유지합니다.

기존 89일 train·29일 test 표본에서는 모든 관측에 binary 변수를 두어도 계산이 가능했습니다. 그러나 300일 train·100일 test로 확대하자 Proposed MLR의 계산 부담이 크게 증가하여, 위의 수식상 조건에 따라 binary가 필요한 관측에만 유지하는 방식으로 재정식화했습니다.

현재 train 데이터에서 사용된 binary 변수 수는 다음과 같습니다.

| 모형 | 축소 전 binary | 축소 후 binary |
|---|---:|---:|
| Proposed AR 12개 합계 | 3,600 | 1 |
| Proposed MLR | 3,600 | 1 |


## 평가

관측치별 수익은 다음과 같습니다.

$$
\Pi(q)=C\Delta t\left[DPq+RP\max(S-q,0)-PC\max(q-S,0)\right].
$$

`C=30 MW`, `\Delta t=1 hour`, `PC=0.5\times DP`를 사용합니다. Raw prediction은 저장하고 지표에는 exact projection을 적용합니다.

$$
q^{\mathrm{eval}}=\min(1,\max(0,q^{\mathrm{raw}})).
$$

nRMSE는 test 1,200개를 한 번에 모아 계산합니다.

$$
\operatorname{nRMSE}=100\frac{\sqrt{N^{-1}\sum_i(q_i^{\mathrm{eval}}-S_i)^2}}{\bar S}.
$$

Oracle은 각 관측에서 `q\in\{0,S,1\}`의 수익 중 가장 큰 값을 선택합니다. Gap은 다음과 같습니다.

$$
100\frac{\Pi^{\mathrm{oracle}}-\Pi(q^{\mathrm{eval}})}{\Pi^{\mathrm{oracle}}}.
$$

논문 표의 gap percentage 분모는 확인되지 않아 현재 값과 직접 비교하지 않았습니다.

## 결과 검산

`operational_predictions.csv`에는 1,200개 관측치 × 4개 모형인 4,800행이 있습니다. 실행기는 저장 직전 CSV를 다시 읽어 nRMSE, raw nRMSE, profit, oracle, gap과 boundary 진단 52개를 평가함수와 별도로 재계산합니다. 현재 결과의 검산 상태는 `VERIFIED`입니다.

