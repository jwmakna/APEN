# 25. merged_for_simulation_z01.csv 컬럼 설명 — dSSRD, dTSR, local_timestamp, local_date, local_hour

## 질문

`merged_for_simulation_z01.csv`에서 칼럼 dSSRD, dTSR, local_timestamp,
local_date, local_hour의 의미는?

## 답변

`build_merged_for_simulation.py`에서 확인한 정의 그대로 정리한다.

### dSSRD, dTSR — 시간당 일사량 증분

GEFCom2014 원본의 `VAR169`(SSRD, 지표면 태양복사)와 `VAR178`(TSR, 대기
상단 순태양복사)은 **24시간 예보 묶음 안에서 누적되는 값**이다 — 즉
원시값 자체는 "그 시각까지 쌓인 총량"이라, 시간별 변화량을 보려면
직전 시각 값을 빼야 한다.

- **dssrd** = 이번 시간 VAR169 − 직전 시간 VAR169 (묶음의 첫 시간은 누적값 자체를 그대로 사용)
- **dtsr** = 이번 시간 VAR178 − 직전 시간 VAR178 (같은 방식)

원래의 누적값도 `ssrd`/`tsr` 열로 따로 남겨두고, `dssrd`/`dtsr`는 별도
열로 추가한다. 차분은 zone 필터링/구간 필터링보다 먼저, 원본 전체에
대해 수행한다. 이 두 값이 **기본/제안 모형 MLR(Eq.6)의 입력변수**
dSSRD, dTSR이다.

```python
# build_merged_for_simulation.py 의 deaccumulate_ssrd_tsr() 발췌
for accumulated_col, increment_col in [("VAR169", "dssrd"), ("VAR178", "dtsr")]:
    increment = groups[accumulated_col].diff()                 # 묶음 안에서 "이번 값 - 직전 값"
    increment.loc[is_first_step] = df.loc[is_first_step, accumulated_col]  # 첫 시간은 누적값 자체를 증분으로
```

### local_timestamp, local_date, local_hour — 호주 현지시간 변환

원본 `timestamp`는 UTC 기준인데, 논문이 "태양광 발전이 있는 12시간
(호주 현지시간 09:00~21:00)"을 낮 시간대로 정의하므로, UTC만으로는
이 구간을 알 수 없어 **UTC → `Australia/Sydney` 타임존으로 변환**한
열을 별도로 만들어둔다.

- **local_timestamp**: UTC timestamp를 Sydney 현지시간(서머타임 자동 반영)으로 변환한 값. 이후 다루기 쉽게 타임존 정보는 떼어냄(naive datetime)
- **local_date**: local_timestamp의 날짜 부분만
- **local_hour**: local_timestamp의 시(0~23) 부분만 — 각 분석 스크립트가 여기서 `9 ≤ local_hour < 21`인 행만 걸러 12시간 낮 시간대로 씀

```python
# build_merged_for_simulation.py 발췌
local_ts = merged["timestamp"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TIMEZONE)
merged["local_timestamp"] = local_ts.dt.tz_localize(None)   # 이후 다루기 쉽게 tz 정보는 뗌
merged["local_date"] = local_ts.dt.date
merged["local_hour"] = local_ts.dt.hour
```

이 파일 자체는 24시간을 전부 저장해두고, "어느 12시간을 낮으로 볼지"는
각 분석 스크립트가 `local_hour`로 직접 고르는 구조다(파일 생성 단계
에서는 필터링하지 않음).

관련: [[17_timestamp_정렬_EST_UTC_Sydney]], [[06_Data_Preprocessing_Summary]]
