# APEN 논문 구현 코드

이 저장소는 Karimi and Kwon (2022)의 *Optimization-driven uncertainty forecasting: Application to day-ahead commitment with renewable energy resources*를 바탕으로, 태양광 발전량 예측모형과 경제적 손실을 고려한 학습 방법을 구현한 프로젝트입니다.

 이 프로젝트의 결과는 논문 표의 수치를 그대로 복원한 결과가 아니라, 아래에 명시한 데이터와 가정을 사용해 같은 학습 구조를 구현한 결과입니다.

## 이 프로젝트에서 한 일

전체 흐름은 다음과 같습니다.

1. GEFCom 태양광·기상 데이터와 MISO DA·RT 가격을 timestamp로 결합합니다.
2. 데이터를 train 300일과 test 100일로 나눕니다.
3. AR과 MLR에 대해 기본 모형 학습과 논문 제안 모형 학습을 각각 실행합니다.
4. Test 예측을 `[0,1]` 범위로 보정한 뒤 nRMSE, realized profit, oracle profit, optimality gap을 계산합니다.

이 문서의 기본 모형은 논문의 `Conventional`, 논문 제안 모형은 논문의 `Proposed`에 해당합니다. 기본 모형은 예측오차만 줄이고, 논문 제안 모형은 예측오차와 전력시장 수익 손실을 함께 줄이도록 학습합니다.

## 데이터와 표본

- 태양광·기상: GEFCom2014 `predictors15.csv`의 Zone 1
- 가격: MISO System day-ahead·real-time LMP
- 사용 시간: Sydney 현지시간 09:00–20:00
- AR 직전 이력: 2013-03-26, 12행
- Train: 2013-03-27–2014-01-20, 300일·3,600행
- Test: 2014-01-21–2014-04-30, 100일·1,200행

`predictors15.csv`와 가격 CSV의 timestamp에는 시간대 정보가 없습니다. 현재 코드는 두 timestamp를 UTC로 간주하고 `Australia/Sydney`로 변환합니다. 이는 논문에서 확인한 조건이 아니라 재현을 위해 정한 가정입니다.

또한 GEFCom 발전량과 MISO 가격은 서로 다른 자료 체계에서 가져왔습니다. 따라서 realized profit은 이 표본 안에서 모형을 비교하기 위한 값이며, 특정 태양광 발전소의 실제 운영 수익으로 해석하지 않습니다.

자세한 전처리와 변수 설명은 [`data/README.md`](data/README.md)와 [`docs/01_METHOD_AND_DATA.md`](docs/01_METHOD_AND_DATA.md)에 있습니다.

## 비교한 모형

| 구분 | AR | MLR |
|---|---|---|
| 입력 | 전날 낮 시간 발전량 12개 | SSRD·TSR 시간별 증분과 hour indicator |
| 기본 모형 | 시간대별 bounded LAD 12개 | 전체 시간을 합친 bounded LAD 1개 |
| 논문 제안 모형 | 시간대별 최적화 기반 모형 12개 | 전체 시간을 합친 최적화 기반 모형 1개 |

Bounded LAD는 train 예측을 0과 1 사이로 제한하면서 평균절대오차를 최소화합니다. 논문 제안 모형은 같은 입력과 예측식을 사용하지만, 학습 목적함수에 경제적 손실을 추가합니다.

현재 저장된 결과는 다음 설정 한 조합으로 계산했습니다.

- Shortage penalty 비율: `0.5`, 즉 $PC_i=0.5DP_i$
- 경제적 손실 가중치: `W1=1`
- 예측오차 가중치: `W2=20`

따라서 아래 결과는 이 설정에서의 결과이며, 다른 penalty 비율이나 `W1/W2` 조합에 대한 민감도 분석 결과는 아닙니다.

논문 제안 모형의 상보조건은 모든 관측에 binary를 두는 대신, 수식상 binary가 필요한 관측에만 남겼습니다. 이는 표본을 줄이거나 제약을 완화한 것이 아니라, 공통 surplus·shortage의 목적계수 부호를 이용한 동일한 MILP의 축소식입니다. 현재 train에서는 논문 제안 AR 12개 모형 전체와 논문 제안 MLR에 각각 binary 1개가 남았습니다.

## 결과

| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |
|---|---:|---:|---:|---:|---:|---:|
| 기본 모형 AR | 34.76% | 53.814411% | 15.04% | 51.350243% | +19.054411%p | +36.310243%p |
| 논문 제안 모형 AR | 34.89% | 54.027363% | 13.91% | 51.202635% | +19.137363%p | +37.292635%p |
| 기본 모형 MLR | 21.76% | 43.455282% | 12.59% | 51.656327% | +21.695282%p | +39.066327%p |
| 논문 제안 모형 MLR | 21.92% | 43.855812% | 11.91% | 51.562050% | +21.935812%p | +39.652050%p |

Δ는 `이번 구현 결과 − 논문 결과`로 계산했으며, 단위는 퍼센트포인트(%p)입니다. 따라서 양수는 이번 구현의 nRMSE 또는 optimality gap이 논문보다 높다는 뜻입니다. 논문 수치는 penalty cost rate 50%에서 기본 모형과 `W1=1`, `W2=20`인 논문 제안 모형의 결과입니다.

- nRMSE는 낮을수록 예측오차가 작습니다.
- 이번 구현의 optimality gap은 $100\times(\text{oracle profit}-\text{realized profit})/\text{oracle profit}$으로 계산한 수익 손실 비율이므로 낮을수록 좋습니다. 
- Realized profit은 높을수록 좋습니다.

같은 예측식끼리 비교하면 논문 제안 모형 AR과 논문 제안 모형 MLR은 각각 기본 모형보다 realized profit은 높고 optimality gap은 낮았습니다. 반면 nRMSE는 조금 높았습니다. 즉, 이번 표본에서는 경제성을 함께 고려한 학습이 순수한 예측오차를 조금 희생하면서 수익 지표를 개선한 결과로 나타났습니다.

논문과 현재 구현은 표본 조건이 다르므로 nRMSE 수치를 동일한 재현 결과로 보면 안 됩니다. 비교 방법과 한계는 [`docs/02_RESULTS_AND_LIMITATIONS.md`](docs/02_RESULTS_AND_LIMITATIONS.md)에 정리했습니다.

## 실행 방법

Python 3.12.13에서 전체 실행을 확인했습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 가상환경을 다음과 같이 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

네 모형을 다시 학습하고 `reproduced_results/`에 결과를 저장하려면 다음을 실행합니다.

```bash
python operational_corrected/run_operational_corrected.py --output-dir reproduced_results
```

실행에 필요한 기본 입력은 `data/`의 CSV 세 개입니다. 실행이 끝나면 지정한 폴더에 예측값, train·test 전처리 데이터, 모형별 계수와 평가 지표가 저장됩니다.

## 폴더

- `operational_corrected/`: 전처리, 평가, 모형 학습과 실행 코드
- `data/`: 실행에 사용하는 CSV와 데이터 설명
- `results/`: 현재 저장된 전처리 결과, 예측값과 지표
- `docs/`: 세부 방법, 결과 해석, 교수님께 확인할 질문
- `docs/audit/`: 코드 정리 전 검증 기록과 파일 해시
