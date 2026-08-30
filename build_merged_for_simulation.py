"""
extract_prices_data.py + extract_solar_weather.py + merge_for_simulation.py 를
하나로 합친 통합 스크립트.

data/ 폴더에 있는 (이미 추출된) 원본들을 읽어서 시뮬레이션용 병합 데이터셋을
루트 디렉토리에 바로 만든다. 중간 산출물(data/prices, data/solar, data/weather)은
따로 만들지 않고 메모리 상에서 바로 병합한다.

입력 (data/):
  data/da_lmp_prices_extended_final.csv   - MISO Day-Ahead LMP, 2012-04-01~2014-07-31 (TIMESTAMP, DA_LMP)
  data/rt_lmp_prices_extended_final.csv   - MISO Real-Time LMP,  2012-04-01~2014-07-31 (TIMESTAMP, RT_LMP)
  data/predictors15.csv                   - GEFCom2014 Solar Task15 원본 (ZONEID, TIMESTAMP, VAR.., POWER)
  (DA/RT 모두 predictors15.csv의 전체 구간(2012-04-01~2014-07-01)을 갭 없이 커버함)

경로 기준:
  BASE는 __file__이 아니라 os.getcwd()(현재 작업 디렉터리)로 잡는다. Jupyter/Colab
  에서 셀 단위로 실행해도(그 환경엔 __file__이 없음) 그대로 동작하도록 하기 위함.
  따라서 이 스크립트는 반드시 data/ 폴더가 있는 프로젝트 루트에서 실행해야 한다
  (다른 위치에서 실행한다면 실행 전에 해당 디렉터리로 os.chdir 할 것).

출력 (ZONEID 1/2/3 별로 각각):
  merged_for_simulation_z01.csv   (프로젝트 루트)
  merged_for_simulation_z02.csv
  merged_for_simulation_z03.csv

타임존 보정 (가격):
  MISO DA/RT 가격은 고정 EST(UTC-5, 서머타임 미적용)로 발표됨
  (원본 엑셀 헤더: "Peak Hour: HE 21 (EST)"). GEFCom2014 태양광/기상은 UTC.
  따라서 가격 timestamp에 +5시간을 더해 UTC로 맞춘 뒤 병합한다.
  (보정 전에는 가격과 태양광/기상이 5시간 어긋난 채로 merge되고 있었음)

타임존 보정 (낮 시간대):
  논문은 "태양광 발전이 있는 12시간(호주 현지시간 09:00~21:00)"을 낮 시간대로 쓴다.
  UTC 기준 timestamp만으로는 이 창을 알 수 없으므로, UTC -> Australia/Sydney 로
  변환한 local_timestamp/local_date/local_hour 열을 추가로 만들어둔다.
  (12개 daylight 시간대만 남기는 필터링은 이 파일에서 하지 않고, 각 분석 스크립트가
  local_hour 로 직접 골라 쓰도록 전체 24시간 그대로 저장한다)

SSRD/TSR 차분(deaccumulation):
  VAR169(SSRD), VAR178(TSR)은 "01,02,...,23,00" 순서로 묶인 24시간 예보 묶음 안에서
  누적되는 값이라, 원시값을 그대로 쓰면 "그 시각까지의 누적 총량"이 된다. 이번 시간
  값에서 바로 직전 시간 값을 빼서 "이 한 시간 동안의 증분"(dssrd, dtsr)을 별도 열로
  만든다. 원래의 누적값도 ssrd/tsr 열로 그대로 남겨둔다 (하위 호환용).
  차분은 zone 필터링/구간 필터링보다 먼저, 원본 전체에 대해 수행한다.
  (D:/03_JiWon/APEN/data/readme.md, operational_corrected 참고)

GEFCom2014 기상변수 익명명 -> 실제 ECMWF 명칭 매핑:
  VAR78  -> tclw  (Total Column Water)
  VAR79  -> tciw  (Total Column Ice Water)
  VAR134 -> sp    (Surface Pressure)
  VAR157 -> r     (Relative Humidity)
  VAR164 -> tcc   (Total Cloud Cover)
  VAR165 -> u10   (10m U Wind)
  VAR166 -> v10   (10m V Wind)
  VAR167 -> t2m   (2m Temperature)
  VAR169 -> ssrd  (Surface Solar Radiation Downwards)
  VAR175 -> strd  (Surface Thermal Radiation Downwards)
  VAR178 -> tsr   (Thermodynamic Surface Radiation)
  VAR228 -> tp    (Total Precipitation)

Usage:
    python build_merged_for_simulation.py
"""

import os
import pandas as pd

# __file__ 대신 os.getcwd() 를 BASE로 쓴다 (Jupyter/Colab 등 __file__이 없는
# 환경에서도 동작하도록). 이 스크립트를 실행하는 현재 작업 디렉터리가
# data/ 폴더가 있는 프로젝트 루트여야 한다.
BASE = os.getcwd()
DATA = os.path.join(BASE, "data")

DA_PATH = os.path.join(DATA, "da_lmp_prices_extended_final.csv")
RT_PATH = os.path.join(DATA, "rt_lmp_prices_extended_final.csv")

PREDICTORS_PATH = os.path.join(DATA, "predictors15.csv")


def out_path(zone_id):
    return os.path.join(BASE, f"merged_for_simulation_z{zone_id:02d}.csv")


# 전체 데이터 구간 = predictors15.csv 의 구간과 동일하게 맞춘다.
# da/rt _final 파일이 이제 predictors15.csv 구간(2012-04-01~2014-07-01)을
# EST->UTC 보정 후에도 전부 커버하도록 확장되었음 (2012-04-01~2014-07-31, 갭 없음).
ZONE_IDS = [1, 2, 3]
START_DATE = "2012-04-01"
END_DATE = "2014-07-01 00:00:00"

LOCAL_TIMEZONE = "Australia/Sydney"   # 논문이 쓰는 낮 시간대(9~21시) 기준 시간대

WEATHER_VARS = [
    "VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",
    "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228",
]

VAR_MAP = {
    "VAR78": "tclw", "VAR79": "tciw", "VAR134": "sp",
    "VAR157": "r", "VAR164": "tcc", "VAR165": "u10",
    "VAR166": "v10", "VAR167": "t2m", "VAR169": "ssrd",
    "VAR175": "strd", "VAR178": "tsr", "VAR228": "tp",
}


# ============================================================
# 1. Solar + Weather : data/predictors15.csv 에서 zone_id 별로 분리
#    (extract_solar_weather.py 로직 + ZONEID 루프)
# ============================================================
def deaccumulate_ssrd_tsr(df):
    """VAR169/VAR178 누적값을 24시간 예보 묶음 안에서 차분해 dssrd/dtsr 증분을 만듦.

    zone/구간 필터링보다 먼저, 원본 전체(df)에 대해 호출해야 한다.
    """
    df = df.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)  # zone별, 시간순 정렬

    bundle_start = df["TIMESTAMP"] - pd.Timedelta(hours=1)   # 이 행 시각에서 1시간을 뺀 임시 시각
    bundle_date = bundle_start.dt.normalize()                  # 그 시각의 날짜만 남김 = 예보 묶음 이름표

    groups = df.groupby(["ZONEID", bundle_date], sort=False)
    step_in_bundle = groups.cumcount() + 1                       # 묶음 안에서 몇 번째 시간인지 (1~24)
    is_first_step = step_in_bundle.eq(1)                          # 묶음의 첫 시간인지 (뺄 대상이 없음)

    for accumulated_col, increment_col in [("VAR169", "dssrd"), ("VAR178", "dtsr")]:
        increment = groups[accumulated_col].diff()                 # 묶음 안에서 "이번 값 - 직전 값"
        increment.loc[is_first_step] = df.loc[is_first_step, accumulated_col]  # 첫 시간은 누적값 자체를 증분으로
        df[increment_col] = increment

    return df


def load_predictors():
    """predictors15.csv 를 한 번만 읽고, 차분 -> 타임스탬프/구간 필터까지 적용."""
    df = pd.read_csv(PREDICTORS_PATH)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d %H:%M")
    df = deaccumulate_ssrd_tsr(df)          # 구간을 자르기 전에, 원본 전체로 먼저 차분
    df = df[(df["TIMESTAMP"] >= START_DATE) & (df["TIMESTAMP"] <= END_DATE)].copy()
    return df


def load_solar_weather(df_all, zone_id):
    df = df_all[df_all["ZONEID"] == zone_id].copy()
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    solar = df[["TIMESTAMP", "POWER"]].rename(
        columns={"TIMESTAMP": "timestamp", "POWER": "solar_power"}
    )
    print(f"[Zone {zone_id}] Solar   {len(solar)}행")

    weather = df[["TIMESTAMP"] + WEATHER_VARS + ["dssrd", "dtsr"]].rename(columns={"TIMESTAMP": "timestamp"})
    weather = weather.rename(columns=VAR_MAP)
    print(f"[Zone {zone_id}] Weather {len(weather)}행, {len(WEATHER_VARS)}변수 + dssrd/dtsr(차분)")

    return solar, weather


# ============================================================
# 2. DA / RT 가격 : data/da_lmp_prices_extended_final.csv, data/rt_lmp_prices_extended_final.csv
#    (extract_prices_data.py 결과물을 그대로 읽음)
# ============================================================
EST_TO_UTC = pd.Timedelta(hours=5)  # MISO는 고정 EST(UTC-5) 발표, DST 미적용


def load_prices():
    da = pd.read_csv(DA_PATH, parse_dates=["TIMESTAMP"])
    da.columns = ["timestamp", "da_price"]
    da["timestamp"] = da["timestamp"] + EST_TO_UTC  # EST -> UTC
    print(f"[DA_LMP]  {len(da)}행 (EST -> UTC 보정)")

    rt = pd.read_csv(RT_PATH, parse_dates=["TIMESTAMP"])
    rt.columns = ["timestamp", "rt_price"]
    rt["timestamp"] = rt["timestamp"] + EST_TO_UTC  # EST -> UTC
    print(f"[RT_LMP]  {len(rt)}행 (EST -> UTC 보정)")

    return da, rt


# ============================================================
# 3. 병합 (merge_for_simulation.py 로직) - zone_id 별로 반복
# ============================================================
def main():
    predictors = load_predictors()
    da, rt = load_prices()  # 가격은 zone과 무관하게 공통 (MISO 시스템 가격)

    for zone_id in ZONE_IDS:
        solar, weather = load_solar_weather(predictors, zone_id)

        merged = pd.merge(solar, da, on="timestamp", how="left")
        merged = pd.merge(merged, rt, on="timestamp", how="left")
        merged = pd.merge(merged, weather, on="timestamp", how="left")

        # RT price 내부 결측 -> forward fill (마지막 날 누락 등 예외적인 경우 대비)
        merged["rt_price"] = merged["rt_price"].ffill()

        # 가격을 EST->UTC로 +5시간 이동시켰기 때문에, 태양광 구간의 맨 앞 몇 시간은
        # 대응하는 가격이 없을 수 있음(과거로 더 당길 원본 EST 데이터가 없음).
        # forward-fill로 채우지 않고 명시적으로 제거한다.
        before = len(merged)
        merged = merged.dropna(subset=["da_price", "rt_price"]).reset_index(drop=True)
        dropped = before - len(merged)
        if dropped:
            print(f"[Zone {zone_id}] 가격 미커버 구간 {dropped}행 제거 (EST->UTC 보정으로 생긴 선두 구간)")

        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # UTC -> Sydney 현지시간 변환 (낮 시간대 09:00~21:00 선택은 각 분석 스크립트가 담당)
        local_ts = merged["timestamp"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TIMEZONE)
        merged["local_timestamp"] = local_ts.dt.tz_localize(None)   # 이후 다루기 쉽게 tz 정보는 뗌
        merged["local_date"] = local_ts.dt.date
        merged["local_hour"] = local_ts.dt.hour

        path = out_path(zone_id)
        merged.to_csv(path, index=False)

        print(f"\n[Zone {zone_id}] 통합 완료: {path}")
        print(f"행: {len(merged)}, 기간: {merged['timestamp'].iloc[0]} ~ {merged['timestamp'].iloc[-1]}")
        print("컬럼:", list(merged.columns))
        print(merged.head(3))
        print("-" * 60)


if __name__ == "__main__":
    main()
