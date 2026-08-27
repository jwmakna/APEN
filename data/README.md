# 데이터

현재 실행에 필요한 파일은 다음 세 개입니다.

```text
data/
├── predictors15.csv
├── da_lmp_prices.csv
├── rt_lmp_prices.csv
└── README.md
```
발전량과 기상변수는 `predictors15.csv`에서 실행 중에 직접 선택합니다.

| 파일 | 현재 실행에서 사용하는 내용 | 행 수 | timestamp 범위 |
|---|---|---:|---|
| `predictors15.csv` | GEFCom2014의 Zone 1 `POWER`, `VAR169`, `VAR178` | 59,112 | 2012-04-01 01:00–2014-07-01 00:00 |
| `da_lmp_prices.csv` | MISO System day-ahead LMP | 9,624 | 2013-03-26 00:00–2014-04-30 23:00 |
| `rt_lmp_prices.csv` | MISO System real-time LMP | 9,624 | 2013-03-26 00:00–2014-04-30 23:00 |

`predictors15.csv`에는 Zone 1–3과 여러 기상변수가 함께 들어 있습니다. 전처리 코드는 원본을 읽은 뒤 Zone 1만 선택하며, 모형에는 다음 변수만 사용합니다.

| 원자료 열 | 사용 방법 |
|---|---|
| `POWER` | 정규화된 태양광 발전량 |
| `VAR169` | 누적 SSRD. 24시간 예보 묶음 안에서 차분하여 `dSSRD` 생성 |
| `VAR178` | 누적 TSR. 24시간 예보 묶음 안에서 차분하여 `dTSR` 생성 |
| `TIMESTAMP` | 시간대 변환, 날짜 및 시간대 구분 |
| `ZONEID` | Zone 1 선택 |

Hour 변수는 원자료의 별도 열을 사용하지 않고 Sydney 현지시각에서 만듭니다.

## 사용 기간

- AR 이력: 2013-03-26, 12개 시간대
- Train: 2013-03-27–2014-01-20, 300일·3,600행
- Test: 2014-01-21–2014-04-30, 100일·1,200행
- 하루 사용 시간: Sydney 현지시각 09:00–20:00, 12개 시간대

최종 train과 test의 4,800개 행에는 `POWER`, DA LMP, RT LMP, `dSSRD`, `dTSR`가 모두 존재합니다. 합성 가격이나 forward fill은 사용하지 않습니다.

## MISO 가격 자료

가격 CSV는 MISO Market Report Archives의 월별 보고서에서 각 일자의 `MISO System` 값을 읽어 만들었습니다.

- Day-Ahead: `Archived Day-Ahead Pricing (zip)`의 2013년 3월–2014년 4월 자료
- Real-Time: `Archived Real-Time Pricing Report (zip)`의 2013년 3월–2014년 5월 자료
- RT는 파일 게시일이 아니라 파일 내부의 `Market Date`를 기준으로 선택

필요한 401개 market date에 대해 DA와 RT의 24시간 가격을 확인했습니다. 이전 자료와 겹치는 구간의 값도 대조했으며, 현재 CSV에는 합성값이나 결측 대체값이 없습니다.


## 시간 처리

`predictors15.csv`와 가격 CSV의 timestamp는 timezone 정보가 없는 형식입니다. 현재 전처리에서는 두 timestamp를 UTC로 간주하고 `Australia/Sydney`로 변환한 뒤 현지시각 09:00–20:00을 선택합니다.


## SSRD와 TSR

`VAR169`와 `VAR178`은 누적값으로 처리합니다. 원자료의 `01, ..., 23, 00` 순서를 하나의 24시간 예보 묶음으로 보고, 묶음 안에서 차분하여 `dSSRD`와 `dTSR`을 만듭니다. 첫 시간의 증분은 해당 누적값 자체를 사용합니다.

차분은 시간대 변환과 낮 시간 선택 전에 수행합니다. 작은 음수 차이는 임의로 0으로 바꾸지 않습니다.

## SHA-256

| 파일 | SHA-256 |
|---|---|
| `predictors15.csv` | `600e3ddd9ce70f9d5086166e608fa1b9383ff6da03f3ae42c79f1cabe47d41e6` |
| `da_lmp_prices.csv` | `3d2d93b17c36c57950593a51ebd07b466daef9330adb5bbf645dd934c840e10e` |
| `rt_lmp_prices.csv` | `bdcd2bc60db98af2033f498143a354f5915de368794738fd72585a9fe391e7c0` |

