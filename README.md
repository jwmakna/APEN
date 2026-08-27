# APEN 논문 구현 코드

본 저장소는 Karimi and Kwon (2022)의 *Optimization-driven uncertainty forecasting: Application to day-ahead commitment with renewable energy resources*를 바탕으로, 논문에서 제안한 AR·MLR 예측모형과 최적화 기반 학습 방법을 구현하고 재현 과정을 정리한 것입니다.

논문에 정확히 적혀 있지 않은 표본 날짜, 시간대와 일부 계산 방식은 별도의 가정을 두었습니다. 따라서 아래 수치는 논문 표를 그대로 복원한 값이 아니라, 이 저장소에 적어 둔 조건으로 계산한 결과입니다.

## 데이터와 표본

- 태양광·기상: GEFCom2014 `predictors15.csv`의 Zone 1
- 가격: MISO System day-ahead·real-time LMP
- 선택 시간: Sydney 현지시간 09:00–20:00
- AR 직전 이력: 2013-03-26, 12행
- Train: 2013-03-27–2014-01-20, 300일·3,600행
- Test: 2014-01-21–2014-04-30, 100일·1,200행

Test에 사용한 DA·RT 가격은 모두 MISO 월별 보고서에서 가져왔습니다. Timestamp 결합은 시각을 UTC로 간주해 `Australia/Sydney`로 변환하였습니다.

## 모형

| 구분 | AR | MLR |
|---|---|---|
| Conventional | 시간대별 bounded LAD 12개 | pooled bounded LAD 1개 |
| Proposed | 시간대별 최적화 기반 모형 12개 | pooled 최적화 기반 모형 1개 |

AR은 전날 daylight POWER 12개를 사용합니다. MLR은 SSRD·TSR의 시간별 증분과 hour indicator를 사용합니다. Proposed 모형의 가중치는 `W1=1`, `W2=20`입니다.

Proposed 모형의 상보조건은 필요한 관측에만 binary variable을 남기는 동일한 MILP로 풀었습니다. 현재 300일 train에서는 Proposed AR 전체와 Proposed MLR에 각각 1개가 필요했습니다. 표본이나 목적함수를 줄인 것은 아닙니다. 자세한 내용은 [`docs/01_METHOD_AND_DATA.md`](docs/01_METHOD_AND_DATA.md)에 적었습니다.

## 결과

| 모델 | 논문 nRMSE | 현재 nRMSE | Realized profit | 현재 gap |
|---|---:|---:|---:|---:|
| Conventional AR | 34.76% | 53.814411% | 499,359.739669 | 51.350243% |
| Proposed AR | 34.89% | 54.027363% | 500,874.846336 | 51.202635% |
| Conventional MLR | 21.76% | 43.455282% | 496,217.977756 | 51.656327% |
| Proposed MLR | 21.92% | 43.855812% | 497,185.675851 | 51.562050% |

MLR이 AR보다 nRMSE가 낮았습니다. 같은 예측모형끼리 비교하면 Proposed AR와 Proposed MLR은 각각 Conventional 모형보다 realized profit이 높고 gap이 낮았지만 nRMSE는 조금 높았습니다.

논문과 현재 구현의 표본 날짜, 데이터 선택과 시간대 가정이 같다고 할 수는 없습니다. 결과 해석은 [`docs/02_RESULTS_AND_LIMITATIONS.md`](docs/02_RESULTS_AND_LIMITATIONS.md)에 정리했습니다.

## 실행

Python 3.12.13에서 확인했습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 가상환경을 다음과 같이 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

테스트:

```bash
python -m unittest discover operational_corrected -p "test_*.py" -v
```

모형 재실행:

```bash
python operational_corrected/run_operational_corrected.py --output-dir reproduced_results
```

기본 입력은 저장소 안의 `data/`입니다. 같은 이름의 결과 파일이 출력 폴더에 있으면 덮어쓰지 않고 실행을 중단합니다. 저장된 실행에서는 62개 테스트와 네 모형 학습, prediction CSV 기반 독립 지표 재계산이 모두 완료되었습니다.

## 폴더

- `operational_corrected/`: 전처리, 평가함수, 모형, 실행 파일과 테스트
- `data/`: 실행 입력과 참고용 추출 파일
- `results/`: 전처리 결과, 예측값과 지표
- `docs/`: 방법, 결과와 남아 있는 질문
- `docs/audit/`: 실행·테스트 로그와 파일 해시

