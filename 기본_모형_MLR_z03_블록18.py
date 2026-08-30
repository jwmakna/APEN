# -*- coding: utf-8 -*-

# =====================================================================
# 기본_모형_MLR.py
#
# 목적: "기본 모형 MLR" (논문에 원래 있는 다중선형회귀 모형 - 제안 모형
#       아님) 의 nRMSE 와 optimality gap 을, "bounded LAD"(학습 중
#       예측값을 [0,1]로 강제하는 linprog 회귀) 방식으로 구해서
#       D:\03_JiWon\APEN\Readme.md 결과표의 "기본 모형 MLR" 행처럼 뽑는다.
#
# MLR 이 AR 과 다른 점: AR 은 시간대(0~11)마다 따로 12개 모델을 만들지만,
# MLR 은 하루 모든 시간대에 "똑같은 계수 하나"만 쓰는 pooled 모델이다
# (논문 Eq.6). 입력변수는 dSSRD, dTSR, Hour(스칼라 하나) 세 개 - 논문이
# backward stepwise 로 최종 선택한 변수 그대로. 또 MLR 은 "내일 날씨
# 예보"를 미리 안다고 가정하는 모델이라, AR처럼 하루씩 굴리며 예측할
# 필요 없이 테스트 구간 전체를 한 번에 예측한다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1)
# 구간(블록9): 학습 2012-11-28~2013-09-23(300일 x 12시간=3600행),
#             테스트 2013-09-24~2014-01-01(100일 x 12시간=1200행)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
from scipy import sparse                     # 희소행렬(linprog 제약식용) 라이브러리
from scipy.optimize import linprog           # 선형계획법(LP) 솔버 - LAD 회귀를 푸는 데 씀


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더 경로
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")  # 읽어올 병합 데이터 파일 경로

LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp("2013-08-25")   # 학습 시작일
TRAIN_END = pd.Timestamp("2013-11-22")     # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp("2013-11-23")    # 테스트 시작일
TEST_END = pd.Timestamp("2013-12-22")      # 테스트 마지막일 (100일째)

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)
PENALTY_RATE = 0.5                # 약정 부족(shortage) 시 벌금비용률 (일간전 가격의 50%)

PAPER_NRMSE = 21.76                # 논문 Table 4, 기본 모형 MLR 의 nRMSE(%) - 비교용
PAPER_GAP = 12.59                  # 논문 Table 4, 기본 모형 MLR 의 optimality gap(%) - 비교용


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
# ↑ local_hour 가 9시 이상, 21시 미만(=9~20시)인 행만 True 인 판단 열을 만듦

daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START
# ↑ local_hour(9~20) 를 0~11 로 다시 번호 매김 (hour_idx) - MLR 의 "Hour" 입력변수로 그대로 씀


# =====================================================================
# 2. 학습(train) / 테스트(test) 구간으로 자르고, 날짜·시간대 순서로 정렬
# =====================================================================
is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy()                    # 학습 구간 행만 골라냄
train_rows = train_rows.sort_values(["local_date", "hour_idx"])       # 날짜, 시간대 순서로 정렬

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy()                      # 테스트 구간 행만 골라냄
test_rows = test_rows.sort_values(["local_date", "hour_idx"])         # 날짜, 시간대 순서로 정렬

print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))  # 3600 이어야 정상
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))    # 1200 이어야 정상


# =====================================================================
# 3. 학습/테스트용 1차원 배열 만들기 (행 하나 = 관측치 하나)
#    - MLR 은 AR 과 달리 (날짜 x 시간대) 표가 필요 없고, 그냥 행 단위로
#      쓸 수 있음 - 정렬해둔 순서 그대로 하나씩 꺼내 배열에 채워 넣음
# =====================================================================

n_train_obs = len(train_rows)                    # 학습 관측치 개수 (3600 이어야 정상)
train_solar = np.zeros(n_train_obs)                # 학습용 실제 발전량을 담을 빈 배열
train_dssrd = np.zeros(n_train_obs)                # 학습용 dSSRD(차분된 태양복사) 값을 담을 빈 배열
train_dtsr = np.zeros(n_train_obs)                 # 학습용 dTSR(차분된 대기상단 복사) 값을 담을 빈 배열
train_hour = np.zeros(n_train_obs)                 # 학습용 Hour(0~11) 값을 담을 빈 배열
row_counter = 0                                      # train_rows 를 순서대로 셀 카운터
for _, one_row in train_rows.iterrows():               # train_rows 를 한 줄씩 순서대로 확인
    train_solar[row_counter] = one_row["solar_power"]    # 실제 발전량 값을 채워 넣음
    train_dssrd[row_counter] = one_row["dssrd"]           # dSSRD 값을 채워 넣음
    train_dtsr[row_counter] = one_row["dtsr"]              # dTSR 값을 채워 넣음
    train_hour[row_counter] = one_row["hour_idx"]           # Hour(0~11) 값을 채워 넣음
    row_counter = row_counter + 1                             # 카운터를 하나 증가시킴

n_test_obs = len(test_rows)                       # 테스트 관측치 개수 (1200 이어야 정상)
test_solar = np.zeros(n_test_obs)                   # 테스트용 실제 발전량을 담을 빈 배열
test_dssrd = np.zeros(n_test_obs)                   # 테스트용 dSSRD 값을 담을 빈 배열
test_dtsr = np.zeros(n_test_obs)                    # 테스트용 dTSR 값을 담을 빈 배열
test_hour = np.zeros(n_test_obs)                    # 테스트용 Hour(0~11) 값을 담을 빈 배열
test_da_price = np.zeros(n_test_obs)                # 테스트용 DA가격을 담을 빈 배열
test_rt_price = np.zeros(n_test_obs)                # 테스트용 RT가격을 담을 빈 배열
row_counter = 0                                       # test_rows 를 순서대로 셀 카운터
for _, one_row in test_rows.iterrows():                # test_rows 를 한 줄씩 순서대로 확인
    test_solar[row_counter] = one_row["solar_power"]     # 실제 발전량 값을 채워 넣음
    test_dssrd[row_counter] = one_row["dssrd"]            # dSSRD 값을 채워 넣음
    test_dtsr[row_counter] = one_row["dtsr"]               # dTSR 값을 채워 넣음
    test_hour[row_counter] = one_row["hour_idx"]            # Hour(0~11) 값을 채워 넣음
    test_da_price[row_counter] = one_row["da_price"]         # DA가격 값을 채워 넣음
    test_rt_price[row_counter] = one_row["rt_price"]          # RT가격 값을 채워 넣음
    row_counter = row_counter + 1                               # 카운터를 하나 증가시킴


# =====================================================================
# 4. 회귀 입력행렬(X) 만들기 - 논문 Eq.(6): 절편 + dSSRD + dTSR + Hour
# =====================================================================

n_features = 4                                          # 절편, dSSRD, dTSR, Hour = 4개 입력변수

X_train = np.zeros((n_train_obs, n_features))              # 학습용 입력행렬 (3600, 4)
for i in range(n_train_obs):                                 # 3600개 학습 행을 하나씩 순서대로
    X_train[i, 0] = 1.0                                        # 첫 열은 절편용 1
    X_train[i, 1] = train_dssrd[i]                              # 둘째 열은 dSSRD
    X_train[i, 2] = train_dtsr[i]                                # 셋째 열은 dTSR
    X_train[i, 3] = train_hour[i]                                 # 넷째 열은 Hour(0~11)

X_test = np.zeros((n_test_obs, n_features))                # 테스트용 입력행렬 (1200, 4)
for i in range(n_test_obs):                                   # 1200개 테스트 행을 하나씩 순서대로
    X_test[i, 0] = 1.0                                          # 첫 열은 절편용 1
    X_test[i, 1] = test_dssrd[i]                                 # 둘째 열은 dSSRD
    X_test[i, 2] = test_dtsr[i]                                   # 셋째 열은 dTSR
    X_test[i, 3] = test_hour[i]                                    # 넷째 열은 Hour(0~11)


# =====================================================================
# 5. "bounded LAD" 회귀 하나를 풀어서 계수 4개를 구함
#    - AR 과 달리 시간대별로 12번 반복할 필요 없이, 딱 한 번만 풀면 됨
#    - 절대오차합(LAD)을 최소화하되, 학습 표본 3600개 전부의 적합값이
#      [0,1] 범위 안에 있도록 강제하는 회귀
# =====================================================================

X_sparse = sparse.csr_matrix(X_train)                     # 입력행렬을 희소행렬 형태로 변환
identity_matrix = sparse.eye(n_train_obs, format="csr")      # 1080x1080 단위행렬 (절대값 처리용 보조변수 계수)

# ---- 일반 LAD: 학습 중 [0,1] 강제 제약 없이 순수 절대오차만 최소화 ----
constraint_block_1 = sparse.hstack([X_sparse, -identity_matrix])   #  X@beta - u <= y
constraint_block_2 = sparse.hstack([-X_sparse, -identity_matrix])  # -X@beta - u <= -y

all_constraints = sparse.vstack([                           # 2묶음을 위아래로 합쳐 하나의 제약행렬로 만듦
    constraint_block_1, constraint_block_2
], format="csr")

constraint_limits = np.concatenate([                        # 각 제약식의 우변(<=) 값들을 순서대로 이어붙임
    train_solar,                  # 첫 묶음의 우변 = y (실제 발전량)
    -train_solar,                 # 두번째 묶음의 우변 = -y
])

objective_coefficients = np.concatenate([                   # 목적함수 계수: beta 에는 0, u 에는 1/3600
    np.zeros(n_features),
    np.ones(n_train_obs) / n_train_obs,
])

variable_bounds = [(None, None)] * n_features + [(0.0, None)] * n_train_obs
# ↑ beta 는 부호 제한 없음, u 는 0 이상

lp_result = linprog(                                        # 선형계획법을 실제로 풂
    objective_coefficients,
    A_ub=all_constraints,
    b_ub=constraint_limits,
    bounds=variable_bounds,
    method="highs",
)

mlr_coefficients = lp_result.x[:n_features]                 # 결과에서 beta(4개)만 뽑아 저장
print(f"MLR 회귀 완료 (성공 여부: {lp_result.success}), 계수: {mlr_coefficients}")   # 진행상황 출력


# =====================================================================
# 6. 테스트 구간 예측 (AR과 달리 한 번에 전체 예측 - 굴릴 필요 없음)
# =====================================================================

test_forecast = np.zeros(n_test_obs)                        # 예측 결과를 담을 빈 배열
for i in range(n_test_obs):                                    # 테스트 1200개 행을 하나씩 순서대로
    raw_prediction = np.dot(mlr_coefficients, X_test[i])          # 계수와 입력을 곱해서 더함 (내적)
    clipped_prediction = min(max(raw_prediction, 0.0), 1.0)         # 예측값을 0~1 범위로 잘라냄
    test_forecast[i] = clipped_prediction                            # 예측 결과 배열에 저장


# =====================================================================
# 7. nRMSE 계산 (Eq. 11-12)
# =====================================================================

sum_of_squared_error = 0.0                   # 제곱오차를 누적할 변수
for i in range(n_test_obs):                    # 1200개 값을 하나씩 순서대로
    error_i = test_solar[i] - test_forecast[i]   # 이 값의 오차(실제-예측)
    sum_of_squared_error = sum_of_squared_error + error_i * error_i   # 오차의 제곱을 누적

mean_squared_error = sum_of_squared_error / n_test_obs   # 누적한 제곱오차의 평균
rmse_value = mean_squared_error ** 0.5                     # 평균제곱오차의 제곱근 = RMSE

sum_of_actual = 0.0                          # 실제값 합계를 누적할 변수
for i in range(n_test_obs):                    # 1200개 값을 하나씩 순서대로
    sum_of_actual = sum_of_actual + test_solar[i]   # 실제값을 누적

average_actual = sum_of_actual / n_test_obs   # 실제 발전량의 평균값
nrmse_percent = 100.0 * rmse_value / average_actual   # RMSE를 평균으로 나누고 100을 곱해 %로 표현


# =====================================================================
# 8. optimality gap 계산 (논문 Eq.1a 이익함수 3항 + Eq.13 오라클 {0,S})
# =====================================================================

sum_of_realized_profit = 0.0             # 실제(MLR 예측 기반) 총 이익을 누적할 변수
sum_of_oracle_profit = 0.0               # 오라클(사후 최적) 총 이익을 누적할 변수

for i in range(n_test_obs):                # 테스트 1200개 관측치를 하나씩 순서대로 처리

    actual_i = test_solar[i]                 # 이 시간의 실제 발전량
    commitment_i = test_forecast[i]          # 이 시간의 MLR 예측값 = 일간전 약정량
    da_i = test_da_price[i]                  # 이 시간의 DA 가격
    rt_i = test_rt_price[i]                  # 이 시간의 RT 가격
    penalty_cost_i = PENALTY_RATE * da_i     # 이 시간의 부족분 벌금단가

    mismatch_i = actual_i - commitment_i       # 실제 - 약정
    surplus_i = max(mismatch_i, 0.0)             # 잉여량
    shortage_i = max(-mismatch_i, 0.0)           # 부족량
    realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
        da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
    )                                             # 논문 Eq.(1a) 3항 이익함수
    sum_of_realized_profit = sum_of_realized_profit + realized_profit_i

    profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)     # 약정 0일 때 이익
    profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)    # 약정=실제발전량일 때 이익
    oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual)          # 둘 중 더 큰 쪽
    sum_of_oracle_profit = sum_of_oracle_profit + oracle_profit_i

optimality_gap_percent = 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit


# =====================================================================
# 9. 결과 출력
# =====================================================================

print()
print("=== 기본 모형 MLR (일반 LAD) - z03/블록18 결과 ===")
print(f"nRMSE          = {nrmse_percent:.6f} %   (논문: {PAPER_NRMSE:.2f} %)")
print(f"optimality gap = {optimality_gap_percent:.6f} %   (논문: {PAPER_GAP:.2f} %)")
print()
print("| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| 기본 모형 MLR | {PAPER_NRMSE:.2f}% | {nrmse_percent:.6f}% | {PAPER_GAP:.2f}% | {optimality_gap_percent:.6f}% "
    f"| {nrmse_percent - PAPER_NRMSE:+.6f}%p | {optimality_gap_percent - PAPER_GAP:+.6f}%p |"
)
