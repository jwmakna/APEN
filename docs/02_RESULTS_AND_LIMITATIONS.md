# 결과와 남아 있는 문제



| 모델 | Solver | 논문 nRMSE | 현재 nRMSE | Realized profit | Oracle profit | 현재 gap |
|---|---|---:|---:|---:|---:|---:|
| Conventional AR | OPTIMAL | 34.76% | 53.814411% | 499,359.739669 | 1,026,438.301786 | 51.350243% |
| Proposed AR | OPTIMAL | 34.89% | 54.027363% | 500,874.846336 | 1,026,438.301786 | 51.202635% |
| Conventional MLR | OPTIMAL | 21.76% | 43.455282% | 496,217.977756 | 1,026,438.301786 | 51.656327% |
| Proposed MLR | OPTIMAL | 21.92% | 43.855812% | 497,185.675851 | 1,026,438.301786 | 51.562050% |

같은 입력을 쓰는 모형끼리 비교하면 다음과 같습니다.

| 비교 | nRMSE 변화 | Realized profit 변화 | Gap 변화 |
|---|---:|---:|---:|
| Proposed AR − Conventional AR | +0.212952%p | +1,515.106667 | −0.147608%p |
| Proposed MLR − Conventional MLR | +0.400530%p | +967.698094 | −0.094277%p |

이번 표본과 현재 설정에서 Proposed 모형은 nRMSE가 조금 높았지만 realized profit은 증가하고 gap은 감소했습니다. 예측오차와 경제성 지표가 같은 방향으로 움직이지 않은 결과입니다.


## 계산 상태

네 모형은 모두 solver가 `OPTIMAL`을 반환했습니다. Proposed AR와 Proposed MLR은 정확한 binary 축소식을 사용했습니다.

- Proposed AR: 3,600개 후보 중 1개 binary
- Proposed MLR: 3,600개 후보 중 1개 binary
- 전체 실행 시간: 현재 환경에서 약 7초
- 저장 prediction 4,800행
- 독립 재계산 지표 52개 일치

90/10일 때는 모든 관측치의 surplus와 shortage에 binary 제약을 적용해도 계산이 원활했지만, 데이터를 300/100으로 확대하자 Proposed MLR의 계산 부담이 지나치게 커졌습니다. 그래서 수식상 상보조건을 자동으로 만족하는 관측에서는 binary를 제거하고, binary가 반드시 필요한 관측에만 유지하는 방식으로 재정식화했습니다.

축소식이 실제 구현에서도 원래 MILP와 동일하게 작동하는지 확인하기 위해, 데이터가 늘어난 뒤에도 두 방식을 모두 계산할 수 있었던 Proposed AR을 기준으로 비교했습니다. Binary 축소 전후 Proposed AR의 시간대별 train 목적값 최대 차이는 약 `5.97e-7`로, solver의 수치 허용오차 범위에 해당했습니다.

따라서 이 방식이 relaxation이나 표본 축소가 아니라, 모든 관측에 binary를 적용한 원래 MILP와 동일한 최적 목적값을 갖는 축소식이라고 판단했습니다. 다만 동일한 최적화 문제라도 solver tolerance나 복수 최적해로 인해 선택되는 계수와 test prediction은 미세하게 달라질 수 있습니다.

## 해석할 때 남는 문제

### 표본 날짜

논문에는 train 300일·test 100일과 train 3개월·test 1개월이라는 설명이 함께 있으며 정확한 날짜는 적혀 있지 않습니다. 현재 날짜는 전자의 표본 크기를 맞추도록 이 저장소에서 정했습니다.

### 데이터 선택과 시간대

GEFCom 원자료의 timestamp에는 시간대 정보가 없으므로, source-naive timestamp를 UTC로 간주한 뒤 Sydney 현지시간으로 변환하였습니다. 첨부된 2014년 1–4월 Zone 1 자료의 시간대별 평균 발전량을 확인한 결과, 02시가 가장 높았고 01시가 그다음으로 높았습니다. 이를 UTC로 해석하면 Sydney의 서머타임 기간에는 각각 현지 13시와 12시에 해당하므로, 태양광 발전량이 정오 무렵 가장 높다는 일반적인 패턴과 일치합니다. 다만 이러한 발전량 패턴은 UTC 및 Sydney 시간대 가정의 개연성을 뒷받침할 뿐, 해당 시간대를 확정하는 근거는 아닙니다.

### 코드에서 정한 조건

- fitted value와 oracle commitment의 `[0,1]` 범위
- 평가 전 exact projection
- Big-M = 1인 상보조건
- 목적함수 normalization
- bounded oracle을 분모로 한 gap
- SSRD·TSR 증분과 hour indicator MLR


## 현재 확인한 범위

- Train 300일·3,600행, test 100일·1,200행
- 날짜별 Sydney 현지시간 09:00–20:00의 12행
- 관측된 POWER·DA·RT만 사용
- 네 solver `OPTIMAL`
- 코드 정리 후 300일·100일 전체 실행 완료
- 기존 저장 prediction과 재실행 prediction의 수치 일치(최대 절대차 0.0)

위 검산은 현재 코드와 저장 결과의 내부 일관성을 확인한 것입니다.

