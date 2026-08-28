# 데이터 설명

현재 코드를 실행하는 데 필요한 파일은 다음 세 개입니다.

```text
data/
├── predictors15.csv
├── da_lmp_prices.csv
├── rt_lmp_prices.csv
└── README.md
```

별도로 가공한 solar CSV와 weather CSV를 입력으로 사용하지 않습니다. 전처리 코드가 `predictors15.csv`에서 필요한 발전량과 기상변수를 직접 선택합니다.

## 파일별 역할

| 파일 | 코드가 사용하는 내용 | 행 수 | timestamp 범위 |
|---|---|---:|---|
| `predictors15.csv` | GEFCom2014 Zone 1의 발전량과 SSRD·TSR | 59,112 | 2012-04-01 01:00–2014-07-01 00:00 |
| `da_lmp_prices.csv` | MISO System day-ahead LMP | 9,624 | 2013-03-26 00:00–2014-04-30 23:00 |
| `rt_lmp_prices.csv` | MISO System real-time LMP | 9,624 | 2013-03-26 00:00–2014-04-30 23:00 |

`predictors15.csv`에는 Zone 1–3과 여러 기상변수가 함께 들어 있지만, 현재 모형에서 사용하는 열은 다음과 같습니다.

| 원자료 열 | 의미와 사용 방법 |
|---|---|
| `POWER` | 설비용량으로 정규화된 태양광 발전량. AR의 과거 입력이자 네 모형의 예측 대상 |
| `VAR169` | 누적 SSRD. 24시간 예보 묶음 안에서 차분해 시간별 증분 `dSSRD`를 생성 |
| `VAR178` | 누적 TSR. 같은 방식으로 시간별 증분 `dTSR`을 생성 |
| `TIMESTAMP` | UTC 가정, Sydney 시간대 변환, 날짜·시간 구분에 사용 |
| `ZONEID` | Zone 1 행만 선택하는 데 사용 |

MLR의 hour indicator는 원자료의 별도 변수가 아니라, timestamp를 Sydney 현지시간으로 변환한 뒤 코드에서 만듭니다.

## 사용 기간

| 용도 | Sydney 현지 날짜 | 일수 | 행 수 |
|---|---|---:|---:|
| AR 직전 이력 | 2013-03-26 | 1 | 12 |
| Train | 2013-03-27–2014-01-20 | 300 | 3,600 |
| Test | 2014-01-21–2014-04-30 | 100 | 1,200 |

하루에서 Sydney 현지시간 09:00–20:00의 12개 시간대만 사용합니다. AR 직전 이력 12행은 첫 train day를 예측할 때만 입력으로 쓰이며 train target에는 포함되지 않습니다.

최종 train과 test의 4,800개 행에는 `POWER`, DA LMP, RT LMP, `dSSRD`, `dTSR`가 모두 있습니다. 가격을 임의로 만들거나 결측치를 forward fill하지 않습니다.

## 가격 데이터

두 가격 CSV는 2013-03-26부터 2014-04-30까지 401일 × 24시간인 9,624개 값을 각각 포함합니다.

- `da_lmp_prices.csv`: MISO System day-ahead LMP
- `rt_lmp_prices.csv`: MISO System real-time LMP

가격은 코드에서 다른 단위로 변환하지 않고 CSV에 저장된 값을 그대로 사용합니다. CSV 자체에 통화 메타데이터가 없으므로, 결과의 realized profit은 모형 간 비교용으로 보고 특정 통화의 실제 금액으로 해석하지 않습니다.

## Timestamp 처리

`predictors15.csv`와 가격 CSV의 timestamp는 시간대 표시가 없는 source-naive 형식입니다. 현재 전처리는 두 timestamp를 UTC로 간주한 뒤 `Australia/Sydney`로 변환하고, Sydney 현지시간 09:00–20:00을 선택합니다.

예를 들어 Sydney가 UTC+11인 기간의 현지시간 09:00은 전날 UTC 22:00에 해당합니다. 서머타임이 끝나 UTC+10이 되면 같은 현지시간 09:00이 전날 UTC 23:00에 매핑됩니다. 코드는 날짜를 직접 나눠 예외 처리하지 않고 timezone database의 변환 결과를 사용합니다.

## SSRD·TSR 차분

`VAR169`와 `VAR178`은 각 시간의 독립된 값이 아니라 예보 묶음 안에서 누적되는 값으로 처리합니다. 원자료의 `01, 02, ..., 23, 00`을 하나의 24시간 예보 묶음으로 보고, 앞 시간의 누적값을 빼서 시간별 증분을 만듭니다.

예를 들어 한 예보 묶음의 누적값이 `10, 25, 40`이면 시간별 증분은 `10, 15, 15`입니다. 첫 시간은 앞선 누적값이 없으므로 해당 누적값 자체를 증분으로 사용합니다.

차분은 시간대 변환과 낮 시간 선택보다 먼저 수행합니다. 음수 증분이 나와도 코드에서 임의로 0으로 바꾸지 않고 원본 차분값을 그대로 보존합니다.

## SHA-256

아래 해시는 현재 입력 CSV가 같은 파일인지 확인하기 위한 값입니다.

| 파일 | SHA-256 |
|---|---|
| `predictors15.csv` | `600e3ddd9ce70f9d5086166e608fa1b9383ff6da03f3ae42c79f1cabe47d41e6` |
| `da_lmp_prices.csv` | `3d2d93b17c36c57950593a51ebd07b466daef9330adb5bbf645dd934c840e10e` |
| `rt_lmp_prices.csv` | `bdcd2bc60db98af2033f498143a354f5915de368794738fd72585a9fe391e7c0` |

