# -*- coding: utf-8 -*-

# =====================================================================
# 기본_모형_AR.py
#
# 목적: "기본 모형 AR" (논문에 원래 있는 AR 모형 - 제안 모형 아님) 의
#       nRMSE 와 optimality gap 을, paper_proposed_block9.py 에서 쓴
#       "bounded LAD"(학습 중 예측값을 [0,1]로 강제하는 linprog 회귀)
#       방식 그대로 다시 구해서 D:\03_JiWon\APEN\Readme.md 결과표의
#       "기본 모형 AR" 행처럼 뽑는다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1, EST->UTC 보정,
#         dssrd/dtsr 차분, Sydney 현지시간 열 포함)
# 구간(블록9): 학습 2012-11-28~2013-09-23(300일),
#             테스트 2013-09-24~2014-01-01(100일)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
from scipy import sparse                     # 희소행렬(linprog 제약식용) 라이브러리
from scipy.optimize import linprog           # 선형계획법(LP) 솔버 - LAD 회귀를 푸는 데 씀


# =====================================================================
# 0. 설정값 (여기 숫자만 바꾸면 동작이 바뀜)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더 경로
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")  # 읽어올 병합 데이터 파일 경로

HOURS_PER_DAY = 12               # 하루 낮 시간대 개수 (Sydney 현지시간 9시~20시)
LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp("2012-11-28")   # 학습 시작일
TRAIN_END = pd.Timestamp("2013-09-23")     # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp("2013-09-24")    # 테스트 시작일
TEST_END = pd.Timestamp("2014-01-01")      # 테스트 마지막일 (100일째)
HISTORY_DATE = pd.Timestamp("2012-11-27")  # 학습 첫날의 "직전 하루" (AR 입력 lag용, target 아님)

LAG_DAYS = 1                     # AR 입력으로 "직전 며칠"을 쓸지 (논문 Eq.3: 직전 하루)

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)
PENALTY_RATE = 0.5                # 약정 부족(shortage) 시 벌금비용률 (일간전 가격의 50%)

PAPER_NRMSE = 34.76                # 논문 Table 3, 기본 모형 AR 의 nRMSE(%) - 비교용
PAPER_GAP = 15.04                  # 논문 Table 3, 기본 모형 AR 의 optimality gap(%) - 비교용


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
# ↑ local_hour 가 9시 이상, 21시 미만(=9~20시)인 행만 True 인 판단 열을 만듦

daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START
# ↑ local_hour(9~20) 를 0~11 로 다시 번호 매김 (hour_idx)


# =====================================================================
# 2. 이력(history) / 학습(train) / 테스트(test) 구간으로 자르기
# =====================================================================
is_history_date = daylight_table["local_date"] == HISTORY_DATE     # 이력 날짜(2012-11-27)인지 판단
history_rows = daylight_table[is_history_date].copy()               # 이력 날짜 행만 골라냄
history_rows = history_rows.sort_values("hour_idx")                  # 시간대(0~11) 순서로 정렬

is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy()                    # 학습 구간 행만 골라냄
train_rows = train_rows.sort_values(["local_date", "hour_idx"])       # 날짜, 시간대 순서로 정렬

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy()                      # 테스트 구간 행만 골라냄
test_rows = test_rows.sort_values(["local_date", "hour_idx"])         # 날짜, 시간대 순서로 정렬

print("이력 날짜:", HISTORY_DATE.date(), "행 수:", len(history_rows))          # 이력 행 수 출력 (12여야 정상)
print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))  # 학습 행 수 출력 (3600 이어야 정상)
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))    # 테스트 행 수 출력 (1200 이어야 정상)


# =====================================================================
# 3. (날짜 x 12시간) 모양의 숫자 배열로 바꾸기
#    - 이미 날짜/시간대 순서로 정렬해뒀으므로, 위에서부터 12개씩 끊어서
#      한 줄(하루)로 채워 넣으면 됨 (naive 하게 순서대로 채움)
# =====================================================================

# --- 3-1. 이력(history) 하루치 발전량 배열 (1, 12) ---
history_solar = np.zeros((1, HOURS_PER_DAY))          # 0으로 채운 빈 배열 준비 (1일 x 12시간)
row_counter = 0                                          # history_rows 를 순서대로 셀 카운터
for _, one_row in history_rows.iterrows():                # history_rows 를 한 줄씩 순서대로 확인
    hour_position = row_counter % HOURS_PER_DAY            # 이 행이 몇 번째 시간대인지 (0~11)
    history_solar[0, hour_position] = one_row["solar_power"]  # 해당 칸에 발전량 값을 채워 넣음
    row_counter = row_counter + 1                            # 카운터를 하나 증가시킴

# --- 3-2. 학습(train) 300일치 발전량 배열 (300, 12) ---
train_dates_sorted = sorted(train_rows["local_date"].unique())  # 학습 구간의 날짜들을 오래된 순으로 정렬한 목록
n_train_days = len(train_dates_sorted)                            # 학습 날짜 수 (300 이어야 정상)
train_solar = np.zeros((n_train_days, HOURS_PER_DAY))              # 0으로 채운 빈 배열 준비 (300일 x 12시간)
row_counter = 0                                                      # train_rows 를 순서대로 셀 카운터
for _, one_row in train_rows.iterrows():                             # train_rows 를 한 줄씩 순서대로 확인
    day_position = row_counter // HOURS_PER_DAY                        # 이 행이 몇 번째 날짜인지 (0~299)
    hour_position = row_counter % HOURS_PER_DAY                        # 이 행이 몇 번째 시간대인지 (0~11)
    train_solar[day_position, hour_position] = one_row["solar_power"]   # 해당 칸에 발전량 값을 채워 넣음
    row_counter = row_counter + 1                                        # 카운터를 하나 증가시킴

# --- 3-3. 테스트(test) 100일치 발전량 / 가격 배열 (100, 12) ---
test_dates_sorted = sorted(test_rows["local_date"].unique())     # 테스트 구간의 날짜들을 오래된 순으로 정렬한 목록
n_test_days = len(test_dates_sorted)                               # 테스트 날짜 수 (100 이어야 정상)
test_solar = np.zeros((n_test_days, HOURS_PER_DAY))                 # 실제 발전량을 담을 빈 배열
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))              # 일간전(DA) 가격을 담을 빈 배열
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))              # 실시간(RT) 가격을 담을 빈 배열
row_counter = 0                                                       # test_rows 를 순서대로 셀 카운터
for _, one_row in test_rows.iterrows():                               # test_rows 를 한 줄씩 순서대로 확인
    day_position = row_counter // HOURS_PER_DAY                          # 이 행이 몇 번째 날짜인지 (0~99)
    hour_position = row_counter % HOURS_PER_DAY                          # 이 행이 몇 번째 시간대인지 (0~11)
    test_solar[day_position, hour_position] = one_row["solar_power"]      # 실제 발전량 값을 채워 넣음
    test_da_price[day_position, hour_position] = one_row["da_price"]      # DA 가격 값을 채워 넣음
    test_rt_price[day_position, hour_position] = one_row["rt_price"]      # RT 가격 값을 채워 넣음
    row_counter = row_counter + 1                                          # 카운터를 하나 증가시킴


# =====================================================================
# 4. AR 학습용 입력(X), 정답(y) 만들기 - 논문 Eq.(3): "직전 하루"의
#    12시간을 입력으로 써서, 다음날 각 시간대를 예측한다.
#    (시간대 h 와 상관없이 입력 X 는 12개 시간대 전부 동일하고, 정답 y 만
#     시간대별로 다르다 - 그래서 X 는 한 번만 만들고, y 는 시간대마다 따로 씀)
# =====================================================================

history_and_train_solar = np.vstack([history_solar, train_solar])   # 이력 하루 + 학습 300일을 위아래로 이어붙임 (301, 12)
# ↑ history_and_train_solar[0] = 이력 하루, history_and_train_solar[1] = 학습 1일째, ...

n_ar_rows = n_train_days                                             # AR 학습 표본 개수 = 학습 날짜 수 (300개)
ar_intercept_column = np.ones((n_ar_rows, 1))                        # 절편(intercept)을 위한 1로만 채운 열 (300, 1)
ar_lag_features = np.zeros((n_ar_rows, HOURS_PER_DAY))                # "직전 하루" 12시간 값을 담을 빈 배열 (300, 12)
for day_index in range(n_ar_rows):                                     # 학습 날짜 0번째부터 299번째까지 순서대로
    previous_day_values = history_and_train_solar[day_index]            # 이 학습일의 "바로 전날" 12시간 값
    ar_lag_features[day_index] = previous_day_values[::-1]               # 시간을 거꾸로 뒤집어서 저장 (논문 식의 순서와 맞춤)

ar_design_matrix = np.hstack([ar_intercept_column, ar_lag_features])   # 절편 열 + lag 12개 열을 옆으로 이어붙임 (300, 13)
# ↑ ar_design_matrix 는 시간대(h)와 무관하게 12개 모델이 전부 공유하는 입력값(X)


# =====================================================================
# 5. 시간대(h=0~11) 마다 따로 "bounded LAD" 회귀를 풀어서 계수를 구함
#    - bounded LAD = 절대오차합(LAD)을 최소화하되, 학습 표본 300개 전부의
#      적합값(예측값)이 [0,1] 범위 안에 있도록 강제하는 회귀
#    - scipy.optimize.linprog 으로 직접 선형계획법을 풂
# =====================================================================

n_features = ar_design_matrix.shape[1]                # 입력 변수 개수 (절편 포함 13개)
coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))   # 시간대별 계수(13개)를 저장할 빈 배열 (12, 13)

for hour in range(HOURS_PER_DAY):                       # 시간대 0부터 11까지 순서대로 하나씩 처리

    y_this_hour = train_solar[:, hour]                    # 이 시간대의 정답값(실제 발전량) 300개

    X_sparse = sparse.csr_matrix(ar_design_matrix)          # 입력행렬을 희소행렬 형태로 변환 (linprog 입력용)
    identity_matrix = sparse.eye(n_ar_rows, format="csr")     # 300x300 단위행렬 (절대값 처리용 보조변수 계수)

    # ---- 절대오차 |y - X@beta| 를 u 라는 보조변수로 표현하기 위한 제약 2묶음
    #      (2026-08-31 부로 bounded LAD 대신 일반 LAD 로 교체 - 논문 Eq.(5)에는
    #      학습 중 [0,1] 강제 제약이 없으므로, 그 조건을 뺀 원문 그대로의 LAD) ----
    constraint_block_1 = sparse.hstack([X_sparse, -identity_matrix])   #  X@beta - u <= y
    constraint_block_2 = sparse.hstack([-X_sparse, -identity_matrix])  # -X@beta - u <= -y  (합치면 u >= |X@beta-y|)

    all_constraints = sparse.vstack([                        # 위 2묶음을 위아래로 합쳐 하나의 제약행렬로 만듦
        constraint_block_1, constraint_block_2
    ], format="csr")

    constraint_limits = np.concatenate([                     # 각 제약식의 우변(<=) 값들을 순서대로 이어붙임
        y_this_hour,                 # 첫 묶음의 우변 = y
        -y_this_hour,                # 두번째 묶음의 우변 = -y
    ])

    objective_coefficients = np.concatenate([                # 목적함수 계수: beta 에는 0, u 에는 1/300
        np.zeros(n_features),          # beta(13개) 는 목적함수에 직접 안 들어감
        np.ones(n_ar_rows) / n_ar_rows,   # u(300개)의 평균을 최소화 = 평균절대오차(LAD) 최소화
    ])

    variable_bounds = [(None, None)] * n_features + [(0.0, None)] * n_ar_rows
    # ↑ beta 는 부호 제한 없음(-무한대~+무한대), u 는 0 이상이어야 함(절대값이니까)

    lp_result = linprog(                                     # 선형계획법을 실제로 풂
        objective_coefficients,
        A_ub=all_constraints,
        b_ub=constraint_limits,
        bounds=variable_bounds,
        method="highs",                                        # highs 라는 빠른 LP 알고리즘 사용
    )

    coefficients_by_hour[hour] = lp_result.x[:n_features]     # 결과에서 beta(13개)만 뽑아 이 시간대의 계수로 저장

    print(f"  시간대 {hour} 회귀 완료 (성공 여부: {lp_result.success})")   # 진행상황 출력


# =====================================================================
# 6. 테스트 100일을 하루씩 순서대로 예측 (rolling one-day-ahead)
#    - 다음날 예측에는 "그 전날의 실제 발전량"을 그대로 씀 (AR의 정의)
# =====================================================================

test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))     # 예측 결과를 담을 빈 배열 (100, 12)
previous_day_actual = train_solar[-1]                        # 테스트 첫날 예측에 쓸 "직전 하루" = 학습 마지막 날

for day_index in range(n_test_days):                          # 테스트 0번째 날부터 99번째 날까지 순서대로

    feature_vector = np.concatenate([[1.0], previous_day_actual[::-1]])
    # ↑ [절편 1] + [직전 하루 12시간을 거꾸로 뒤집은 값] = 학습 때와 똑같은 형태의 입력벡터 (13,)

    for hour in range(HOURS_PER_DAY):                            # 이 날의 시간대 0~11을 하나씩 예측
        raw_prediction = np.dot(coefficients_by_hour[hour], feature_vector)  # 계수와 입력을 곱해서 더함 (내적)
        clipped_prediction = min(max(raw_prediction, 0.0), 1.0)              # 예측값을 0~1 범위로 잘라냄
        test_forecast[day_index, hour] = clipped_prediction                    # 예측 결과 배열에 저장

    previous_day_actual = test_solar[day_index]                # 다음날 예측을 위해 "직전 하루"를 오늘의 실제값으로 갱신


# =====================================================================
# 7. nRMSE 계산 (Eq. 11-12)
# =====================================================================

actual_flat = test_solar.flatten()          # (100,12) 실제값 표를 1200개짜리 한 줄로 펼침
predicted_flat = test_forecast.flatten()     # (100,12) 예측값 표도 1200개짜리 한 줄로 펼침

sum_of_squared_error = 0.0                   # 제곱오차를 누적할 변수, 0에서 시작
for i in range(len(actual_flat)):              # 1200개 값을 하나씩 순서대로
    error_i = actual_flat[i] - predicted_flat[i]   # 이 값의 오차(실제-예측)
    sum_of_squared_error = sum_of_squared_error + error_i * error_i   # 오차의 제곱을 누적

mean_squared_error = sum_of_squared_error / len(actual_flat)   # 누적한 제곱오차의 평균
rmse_value = mean_squared_error ** 0.5                           # 평균제곱오차의 제곱근 = RMSE

sum_of_actual = 0.0                          # 실제값 합계를 누적할 변수
for i in range(len(actual_flat)):              # 1200개 값을 하나씩 순서대로
    sum_of_actual = sum_of_actual + actual_flat[i]   # 실제값을 누적

average_actual = sum_of_actual / len(actual_flat)   # 실제 발전량의 평균값
nrmse_percent = 100.0 * rmse_value / average_actual   # RMSE를 평균으로 나누고 100을 곱해 %로 표현


# =====================================================================
# 8. optimality gap 계산 (논문 Eq.1a 이익함수 3항 + Eq.13 오라클 {0,S})
#    - 이익함수: DA가격*약정량 + RT가격*잉여 - (0.5*DA가격)*부족분  (3항)
#    - 오라클: {0, 실제발전량} 중 더 이익나는 쪽 (설비최대 후보는 안 씀)
# =====================================================================

da_flat = test_da_price.flatten()        # (100,12) DA가격 표를 1200개짜리 한 줄로 펼침
rt_flat = test_rt_price.flatten()        # (100,12) RT가격 표를 1200개짜리 한 줄로 펼침

sum_of_realized_profit = 0.0             # 실제(AR 예측 기반) 총 이익을 누적할 변수
sum_of_oracle_profit = 0.0               # 오라클(사후 최적) 총 이익을 누적할 변수

for i in range(len(actual_flat)):          # 테스트 1200개 관측치를 하나씩 순서대로 처리

    actual_i = actual_flat[i]                # 이 시간의 실제 발전량
    commitment_i = predicted_flat[i]         # 이 시간의 AR 예측값 = 일간전 약정량(commitment)
    da_i = da_flat[i]                        # 이 시간의 DA 가격
    rt_i = rt_flat[i]                        # 이 시간의 RT 가격
    penalty_cost_i = PENALTY_RATE * da_i     # 이 시간의 부족분 벌금단가 (DA가격의 50%)

    # ---- 실제(AR 예측) 커밋에 대한 이익 계산 ----
    mismatch_i = actual_i - commitment_i       # 실제 - 약정 (양수면 잉여, 음수면 부족)
    surplus_i = max(mismatch_i, 0.0)             # 잉여량 (실제가 약정보다 많은 경우만 양수)
    shortage_i = max(-mismatch_i, 0.0)           # 부족량 (약정이 실제보다 많은 경우만 양수)
    realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
        da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
    )                                             # 논문 Eq.(1a) 3항 이익함수
    sum_of_realized_profit = sum_of_realized_profit + realized_profit_i   # 총 이익에 누적

    # ---- 오라클(사후 최적) 이익 계산: {0, 실제발전량} 두 후보 중 더 큰 쪽 ----
    profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
    # ↑ 약정을 0으로 하면: 부족/잉여 없이 실제발전량 전부를 RT가격에 파는 것과 같음

    profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
    # ↑ 약정을 실제발전량만큼 하면: 부족/잉여 없이 전부 DA가격에 파는 것과 같음

    oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual)   # 둘 중 더 이익나는 쪽 선택
    sum_of_oracle_profit = sum_of_oracle_profit + oracle_profit_i             # 총 오라클 이익에 누적

optimality_gap_percent = 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit
# ↑ (오라클 이익 - 실제 이익) / 오라클 이익 * 100 = optimality gap(%)


# =====================================================================
# 9. 결과 출력 (D:\03_JiWon\APEN\Readme.md 결과표와 같은 형식)
# =====================================================================

print()
print("=== 기본 모형 AR (일반 LAD, 2026-08-31부로 bounded LAD 대체) - 블록9 결과 ===")
print(f"nRMSE          = {nrmse_percent:.6f} %   (논문: {PAPER_NRMSE:.2f} %)")
print(f"optimality gap = {optimality_gap_percent:.6f} %   (논문: {PAPER_GAP:.2f} %)")
print()
print("| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| 기본 모형 AR | {PAPER_NRMSE:.2f}% | {nrmse_percent:.6f}% | {PAPER_GAP:.2f}% | {optimality_gap_percent:.6f}% "
    f"| {nrmse_percent - PAPER_NRMSE:+.6f}%p | {optimality_gap_percent - PAPER_GAP:+.6f}%p |"
)
