# 300/100 데이터 확장 및 실행 기록

## 변경 내용

기존 train·90일 tes·30일을 300일·100일로 늘렸습니다.

- AR history: 2013-03-26, 12행
- Train: 2013-03-27–2014-01-20, 3,600행
- Test: 2014-01-21–2014-04-30, 1,200행
- DA·RT: 각각 9,624개 연속 시간 가격
- 합성·forward-filled RT: 0개



## Proposed MILP 계산 변경

300일 Proposed pooled MLR을 기존 all-binary 식으로 실행하면 3,600개 binary의 최적성 증명에 오래 걸렸습니다. 목적함수를 바꾸지 않고 관측별 공통 surplus·shortage 증가비용을 이용한 축소식을 적용했습니다.

- 공통 증가비용이 음수가 아니면 최적성 자체가 상보조건을 보장하므로 binary 제거
- 음수인 관측에는 기존 Big-M binary 유지
- Solver 목적값과 원래 목적함수 재계산값을 실행 후 대조
- 양쪽 부호를 포함한 경우를 별도 계산으로 확인

현재 자료에서는 Proposed AR와 Proposed MLR에 각각 1개 binary가 남았습니다.

## 실행 확인

```text
300일·100일 전체 재실행 완료
기존 prediction과 최대 절대차 0.0
```

전처리, MISO 시간열 완전성, DST, AR rolling, LAD·MILP, 평가식과 저장 검산을 확인했습니다.

## 실행 결과

| 모델 | Solver | nRMSE | Realized profit | Oracle profit | Gap |
|---|---|---:|---:|---:|---:|
| Conventional AR | OPTIMAL | 53.814411% | 499,359.739669 | 1,026,438.301786 | 51.350243% |
| Conventional MLR | OPTIMAL | 43.455282% | 496,217.977756 | 1,026,438.301786 | 51.656327% |
| Proposed AR | OPTIMAL | 54.027363% | 500,874.846336 | 1,026,438.301786 | 51.202635% |
| Proposed MLR | OPTIMAL | 43.855812% | 497,185.675851 | 1,026,438.301786 | 51.562050% |

전체 실행은 현재 환경에서 약 7초 걸렸습니다. Prediction CSV 4,800행에서 52개 지표를 다시 계산했으며 결과 JSON과 일치했습니다.

## 남아 있는 가정

표본 날짜, source-naive timestamp의 UTC 해석, GEFCom과 MISO 자료 결합, accumulation grouping과 corrected gap 정의는 이 프로젝트에서 정한 조건입니다. 데이터 크기를 논문의 300/100 설명에 맞췄지만 논문의 정확한 표본을 확인한 것은 아닙니다.
