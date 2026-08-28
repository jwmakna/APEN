# 방법과 데이터

## 1. 무엇을 구현했는가

이 프로젝트의 목적은 두 가지 학습 방법을 비교하는 것입니다.

- **기본 모형**: 논문의 `Conventional`에 해당하며, 실제 발전량과 예측값의 절대오차만 최소화합니다.
- **논문 제안 모형**: 논문의 `Proposed`에 해당하며, 예측오차와 하루전 약정에서 발생하는 경제적 손실을 함께 최소화합니다. 

두 학습 방법을 AR과 MLR 예측식에 각각 적용하여 다음 네 모형을 실행했습니다.

| 예측식 | 기본 모형 | 논문 제안 모형 |
|---|---|---|
| AR | 시간대별 bounded LAD 12개 | 시간대별 MILP 12개 |
| MLR | 전체 시간을 합친 bounded LAD 1개 | 전체 시간을 합친 MILP 1개 |

논문에서 정확히 확인하기 어려운 표본 날짜, solar zone, price node, timestamp 결합 방식은 다음과 같이 정했습니다.

- GEFCom2014 Zone 1
- 시간대 표시가 없는 timestamp를 UTC로 해석
- Sydney 현지시간 09:00–20:00
- 결측이 없는 POWER·DA·RT만 사용
- Train 300일, test 100일

논문에는 train 300일·test 100일과 train 3개월·test 1개월이라는 설명이 함께 있습니다. 현재 구현은 300일·100일 설명에 맞추었으며, 정확한 날짜는 사용 가능한 데이터 범위 안에서 정했습니다.

## 2. 기호

| 기호 | 의미 |
|---|---|
| $S_i$ | 관측 $i$의 실제 태양광 발전량 |
| $x_i$ | 모형이 예측한 하루전 약정량 |
| $y'_i$ | 실제 발전량이 약정량보다 많은 부분, 즉 surplus |
| $y''_i$ | 약정량이 실제 발전량보다 많은 부분, 즉 shortage |
| $DP_i$ | Day-ahead 가격 |
| $RP_i$ | Real-time 가격 |
| $PC_i$ | 부족 패널티 가격. 현재 $0.5DP_i$ |
| $C$ | 설비용량 30 MW |
| $\Delta t$ | 한 관측의 길이 1 hour |

`POWER`와 $x$는 설비용량 대비 비율이므로 0과 1 사이의 값으로 다룹니다.

## 3. 전체 실행 흐름

1. 태양광·기상 원자료에서 Zone 1과 필요한 변수를 선택합니다.
2. 누적 SSRD·TSR을 시간별 증분으로 변환합니다.
3. Timestamp를 Sydney 현지시간으로 변환하고 09:00–20:00을 선택합니다.
4. 같은 UTC timestamp의 POWER, DA LMP, RT LMP를 합칩니다.
5. Train에서 기본 모형과 논문 제안 모형 AR·MLR을 학습합니다.
6. Test에서 raw prediction을 만든 뒤 `[0,1]`로 projection합니다.
7. Projected prediction으로 nRMSE, realized profit, oracle profit, gap을 계산합니다.

## 4. 전처리

`operational_preprocessing.py`는 다음 순서로 데이터를 만듭니다.

1. `predictors15.csv`를 timestamp 순서로 정렬합니다.
2. `01, 02, ..., 23, 00`을 하나의 24시간 예보 묶음으로 보고 accumulated SSRD·TSR을 차분합니다.
3. Zone 1을 선택합니다.
4. Source-naive timestamp를 UTC로 간주하고 `Australia/Sydney`로 변환합니다.
5. Sydney 현지시간 09:00–20:00의 행만 남깁니다.
6. UTC timestamp를 기준으로 DA·RT 가격을 결합합니다.
7. Sydney 현지 날짜로 history, train, test를 나눕니다.

| 용도 | Sydney 현지 날짜 | 일수 | 행 수 |
|---|---|---:|---:|
| AR 직전 POWER 이력 | 2013-03-26 | 1 | 12 |
| Train | 2013-03-27–2014-01-20 | 300 | 3,600 |
| Test | 2014-01-21–2014-04-30 | 100 | 1,200 |

AR history는 첫 train day의 입력으로만 사용하고 train target에는 넣지 않습니다. 각 날짜는 현지시간 09시부터 20시까지 12행을 갖습니다. DST는 특정 날짜를 수동으로 나누지 않고 timezone database의 변환 결과를 사용합니다.

## 5. 기본 모형 학습

### 5.1 Bounded LAD

기본 모형 AR과 MLR은 평균절대오차를 최소화합니다.

$$
\min_\beta \frac{1}{N}\sum_{i=1}^{N}\left|S_i-X_i\beta\right|
$$

Train fitted value에는 다음 제약을 둡니다.

$$
0\le X_i\beta\le1.
$$

실제 발전량이 0과 1 사이로 정규화되어 있으므로, train에서 학습하는 약정량도 같은 범위에 들어오게 한 것입니다. 

### 5.2 AR

AR은 현지시간 09–20시에 대해 target hour별 모형 12개를 따로 학습합니다. 날짜 $d$의 입력은 전날 발전량 12개입니다.

$$
X_d=
\left[
1,
S_{d-1,20},
S_{d-1,19},
\ldots,
S_{d-1,09}
\right].
$$

가장 최근 시간인 20시부터 거꾸로 배열하고, 시간 $h$의 예측은 다음과 같습니다.

$$
\hat S_{d,h}=X_d\beta_h,
\qquad h\in\{09,10,\ldots,20\}.
$$

각 $\beta_h$는 intercept 1개와 lag 계수 12개를 가지므로 target hour별 design matrix는 13열입니다.

Test에서는 하루의 12개 시간을 모두 예측한 뒤, 그날의 actual POWER를 다음 날 입력에 추가합니다. 즉 당일 예측 도중에 당일 actual을 사용하지는 않지만, 다음 날을 예측할 때는 전날 actual이 알려져 있다고 가정합니다.

### 5.3 MLR

MLR은 12개 시간대를 하나의 train 표로 합쳐 계수 한 세트를 학습합니다.

$$
\hat S_i=
\beta_0
+\beta_1 z(\mathrm{dSSRD}_i)
+\beta_2 z(\mathrm{dTSR}_i)
+\sum_{h=1}^{11}\gamma_h I(\mathrm{hour\_idx}_i=h).
$$

Design matrix 14열의 구성은 다음과 같습니다.

- intercept 1열
- `dSSRD`, `dTSR` 2열
- 09시를 기준 시간으로 한 hour indicator 11열

`dSSRD`와 `dTSR`은 **train에서 계산한 평균과 표준편차**로 표준화합니다. Test 통계를 표준화에 사용하지 않습니다. Hour indicator는 0과 1로 구성되며 표준화하지 않습니다.

## 6. 논문 제안 모형 학습

### 6.1 약정량과 불균형

한 관측에서 모형이 $x$를 약정했고 실제 발전량이 $S$라면 두 경우가 생깁니다.

- $S>x$: 발전량이 약정량보다 많으므로 surplus $y'=S-x$
- $x>S$: 약정량이 발전량보다 많으므로 shortage $y''=x-S$

두 변수를 사용하면 관계식은 다음과 같습니다.

$$
x+y'-y''=S,
\qquad x,y',y''\in[0,1].
$$

정상적인 불균형 표현에서는 surplus와 shortage가 동시에 양수일 수 없으므로 $y'y''=0$이어야 합니다. 코드의 `x_slice`, `y_plus_slice`, `y_minus_slice`가 각각 $x$, $y'$, $y''$에 해당합니다.

### 6.2 수익

관측 $i$의 realized profit은 다음과 같습니다.

$$
\Pi_i(x_i)
=C\Delta t
\left[
DP_i x_i
+RP_i y'_i
-PC_i y''_i
\right].
$$

약정한 $x_i$는 day-ahead 가격을 받고, surplus $y'_i$는 real-time 가격을 받으며, shortage $y''_i$에는 패널티 가격이 부과됩니다. 현재 설정은 $C=30$, $\Delta t=1$, $PC_i=0.5DP_i$입니다. 여기서 `0.5`는 고정 금액이 아니라, 각 관측의 day-ahead 가격 중 50%를 shortage penalty 가격으로 사용하는 비율입니다.

### 6.3 Oracle

Oracle은 각 관측의 실제 $S_i$, $DP_i$, $RP_i$를 알고 있다고 가정했을 때 받을 수 있는 가장 큰 수익입니다. 수익식이 $x$에 대해 구간별 선형이므로, 코드는 다음 세 약정량을 비교합니다.

$$
x_i\in\{0,S_i,1\}.
$$

이 세 경우의 수익 중 가장 큰 값을 $\Pi_i^{\mathrm{oracle}}$로 사용합니다.

### 6.4 논문 제안 모형 목적함수

논문 제안 모형은 각 모형의 train 관측에서 다음 목적함수를 최소화합니다. MLR은 3,600개 관측을 한 번에 학습하고, AR은 시간대별 300개 관측으로 이 목적함수를 12번 따로 최소화합니다.

$$
W_1
\frac{
\sum_i\left[
\Pi_i^{\mathrm{oracle}}-\Pi_i(x_i)
\right]
}{
\sum_i\Pi_i^{\mathrm{oracle}}
}
+W_2
\frac{1}{N}
\sum_i|S_i-x_i|.
$$

- 첫 번째 항: oracle 대비 놓친 수익의 비율
- 두 번째 항: 평균절대예측오차
- 현재 가중치: $W_1=1$, $W_2=20$

즉 논문 제안 모형은 예측오차를 무시하고 수익만 최대화하는 모형이 아닙니다. 두 항을 같이 넣어 예측 정확도와 경제성 사이의 trade-off를 학습합니다.

현재 저장된 결과에서는 penalty 비율 `0.5`와 $W_1=1$, $W_2=20$ 조합만 실행했습니다. 다른 penalty 비율이나 가중치 조합은 아직 계산하지 않았으므로, 현재 결과는 이 한 조합에만 해당합니다.

### 6.5 Binary를 필요한 관측에만 남긴 이유

변수 $y'$ 와 $y''$를 동시에 $\delta$만큼 늘리면 $y'-y''$는 변하지 않습니다. 따라서 $x+y'-y''=S$는 그대로 만족하고 $x$도 변하지 않습니다.

$K=C\Delta t$, $D=\sum_j\Pi_j^{\mathrm{oracle}}$라고 하면, 코드에서 $y'_i$와 $y''_i$의 목적함수 계수는 다음과 같습니다.

$$
c_i^+=-W_1\frac{KRP_i}{D}+\frac{W_2}{N},
\qquad
c_i^-=W_1\frac{KPC_i}{D}+\frac{W_2}{N}.
$$

두 변수를 함께 $\delta$만큼 늘렸을 때 목적함수는 다음만큼 변합니다.

$$
\left(c_i^+ + c_i^-\right)\delta.
$$

이 합의 부호에 따라 binary 필요 여부가 결정됩니다.

- $c_i^+ + c_i^- > 0$: surplus와 shortage를 함께 늘리면 목적값이 나빠지므로 최적해에서 둘 중 하나는 자동으로 0이 됩니다.
- $c_i^+ + c_i^- = 0$: 둘의 공통 부분을 줄여도 $x$와 목적값이 변하지 않으므로 binary 없이 상보적인 최적해를 선택할 수 있습니다.
- $c_i^+ + c_i^- < 0$: 둘을 함께 늘리는 것이 목적값을 줄이므로, 인위적인 동시 surplus·shortage를 막기 위해 binary가 필요합니다.

즉 데이터 값을 먼저 보고 임의의 기준으로 binary를 줄인 것이 아니라, 각 관측의 **목적함수 계수에서 유도되는 부호 조건**을 적용한 것입니다. Binary가 필요한 행에서는 $M=1$인 Big-M 제약으로 $y'$ 와 $y''$가 동시에 양수가 되지 못하게 합니다.

기존 90일 train·10일 test에서는 모든 train 관측에 binary를 두어도 계산이 가능했습니다. Train을 300일로 늘리면 논문 제안 모형 MLR의 all-binary MILP에 binary 3,600개가 필요해 계산 부담이 크게 증가합니다. 위 조건을 적용한 결과는 다음과 같습니다.

| 모형 | 전체 train 관측 | 남은 binary |
|---|---:|---:|
| 논문 제안 모형 AR 12개 합계 | 3,600 | 1 |
| 논문 제안 모형 MLR | 3,600 | 1 |

이 방식은 train 행을 제거하거나 목적함수를 근사한 것이 아닙니다. 목적계수상 상보조건이 최적해에서 자동으로 성립하는 행의 binary만 제거한 것입니다.

## 7. Test 예측과 평가

### 7.1 Raw prediction과 projection

모형이 낸 raw prediction은 그대로 `operational_predictions.csv`에 저장합니다. 공식 평가에서는 태양광 발전량의 정규화 범위에 맞게 다음 projection을 적용합니다.

$$
x_i^{\mathrm{eval}}
=\min\left(1,\max\left(0,x_i^{\mathrm{raw}}\right)\right).
$$

Raw prediction이 0과 1 사이에 있으면 그대로 사용하고, 0보다 작으면 0, 1보다 크면 1로 바꾸는 단순한 범위 보정입니다.

### 7.2 nRMSE

nRMSE는 test 1,200개 관측을 한 번에 모아 계산합니다.

$$
\operatorname{nRMSE}
=100
\frac{
\sqrt{
N^{-1}\sum_i
\left(x_i^{\mathrm{eval}}-S_i\right)^2
}
}{
\bar S
}.
$$

이 값은 낮을수록 예측오차가 작습니다.

### 7.3 Realized profit과 gap

Realized profit은 projected prediction $x_i^{\mathrm{eval}}$을 약정량으로 넣은 수익입니다. Oracle profit은 각 관측에서 $x_i\in\{0,S_i,1\}$ 중 수익이 가장 큰 경우입니다.

전체 test gap은 다음과 같습니다.

$$
\operatorname{Gap}(\%)
=100
\frac{
\sum_i\left[
\Pi_i^{\mathrm{oracle}}
-\Pi_i\left(x_i^{\mathrm{eval}}\right)
\right]
}{
\sum_i\Pi_i^{\mathrm{oracle}}
}.
$$

Gap은 oracle이 받을 수 있었던 총수익 중 현재 예측으로 놓친 수익의 비율입니다. 따라서 낮을수록 좋습니다.

논문 표의 gap percentage는 분모가 충분히 명시되어 있지 않아, 현재 gap과 논문 gap을 직접 비교하지 않았습니다.

## 8. 현재 재실행 상태

코드를 공유용으로 간소화한 뒤 300일 train·100일 test 전체를 다시 실행했습니다. 네 모형이 모두 해를 반환했고, 새로 생성한 prediction 4,800행과 정리 전 저장 prediction을 비교했을 때 숫자형 열의 최대 절대차는 `0.0`이었습니다.

이 비교는 코드 정리 과정에서 별도로 수행한 확인입니다. 현재의 간소화된 실행 파일은 예측과 평가 결과를 저장하며, 이전의 독립 재계산·테스트 절차를 반복 실행하지는 않습니다.
