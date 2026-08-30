# -*- coding: utf-8 -*-

# =====================================================================
# 논문_제안_모형_MLR.py
#
# 목적: "논문 제안 모형 MLR" (논문 Eq.10 - 발전계획 최적화를 MLR 학습에
#       통합한 모형) 의 nRMSE 와 optimality gap 을 구해서
#       D:\03_JiWon\APEN\Readme.md 결과표의 "논문 제안 모형 MLR" 행처럼
#       뽑는다. 기본_모형_MLR.py 와 입력/평가 방식은 같고, 회귀를 푸는
#       방법만 "MILP(혼합정수계획법)" 로 바뀐다.
#
# MLR 이 AR 과 다른 점(논문_제안_모형.py 참고): MLR 은 시간대별로 12번
# 반복하지 않고, 학습 표본 3600개(300일 x 12시간) 전체를 하나의 MILP로
# 한 번에 푼다 - 계수(beta) 하나만 나오고, 그 계수를 모든 시간대에
# 똑같이 쓴다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1)
# 구간(블록9): 학습 2012-11-28~2013-09-23(300일 x 12시간=3600행),
#             테스트 2013-09-24~2014-01-01(100일 x 12시간=1200행)
# 가중치: W1=1, W2=20, 벌금비용률=50% (논문 Table 4 비교 지점)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
from scipy import sparse                     # 희소행렬(제약식용) 라이브러리
from scipy.optimize import Bounds, LinearConstraint, milp   # 혼합정수계획법(MILP) 솔버


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더 경로
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")  # 읽어올 병합 데이터 파일 경로

LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp("2012-11-28")   # 학습 시작일
TRAIN_END = pd.Timestamp("2013-09-23")     # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp("2013-09-24")    # 테스트 시작일
TEST_END = pd.Timestamp("2014-01-01")      # 테스트 마지막일 (100일째)

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)
PENALTY_RATE = 0.5                # 약정 부족(shortage) 시 벌금비용률 (일간전 가격의 50%)

W1 = 1.0                          # 정규화 파라미터 W1 (최적화 격차 가중치)
W2 = 20.0                         # 정규화 파라미터 W2 (예측오차 가중치) - 논문 Table 4 비교 지점

PAPER_NRMSE = 21.92                # 논문 Table 4, 논문 제안 모형 MLR 의 nRMSE(%) - 비교용
PAPER_GAP = 11.91                  # 논문 Table 4, 논문 제안 모형 MLR 의 optimality gap(%) - 비교용


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
#    - 이번엔 학습용 DA/RT 가격도 필요함 (제안 모형은 학습 때 가격도
#      손실함수에 쓰기 때문)
# =====================================================================

n_train_obs = len(train_rows)                    # 학습 관측치 개수 (3600 이어야 정상)
train_solar = np.zeros(n_train_obs)                # 학습용 실제 발전량을 담을 빈 배열
train_dssrd = np.zeros(n_train_obs)                # 학습용 dSSRD 값을 담을 빈 배열
train_dtsr = np.zeros(n_train_obs)                 # 학습용 dTSR 값을 담을 빈 배열
train_hour = np.zeros(n_train_obs)                 # 학습용 Hour(0~11) 값을 담을 빈 배열
train_da_price = np.zeros(n_train_obs)             # 학습용 DA가격을 담을 빈 배열
train_rt_price = np.zeros(n_train_obs)             # 학습용 RT가격을 담을 빈 배열
row_counter = 0                                      # train_rows 를 순서대로 셀 카운터
for _, one_row in train_rows.iterrows():               # train_rows 를 한 줄씩 순서대로 확인
    train_solar[row_counter] = one_row["solar_power"]    # 실제 발전량 값을 채워 넣음
    train_dssrd[row_counter] = one_row["dssrd"]           # dSSRD 값을 채워 넣음
    train_dtsr[row_counter] = one_row["dtsr"]              # dTSR 값을 채워 넣음
    train_hour[row_counter] = one_row["hour_idx"]           # Hour(0~11) 값을 채워 넣음
    train_da_price[row_counter] = one_row["da_price"]        # DA가격 값을 채워 넣음
    train_rt_price[row_counter] = one_row["rt_price"]         # RT가격 값을 채워 넣음
    row_counter = row_counter + 1                                # 카운터를 하나 증가시킴

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
    row_counter = row_counter + 1                                # 카운터를 하나 증가시킴


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
# 5. "논문 제안 모형 MILP" (Eq.10) 를 딱 한 번 풀어서 계수 4개를 구함
#    - AR 과 달리 시간대별 반복 없이, 3600개 표본 전체로 MILP 한 번만 품
#    - 학습용 오라클({0,실제발전량,설비최대 1.0} 3후보)로 W1/W2 정규화
#      상수를 잡고, 평가는 나중에 {0,S} 오라클로 따로 함 (AR 코드와 동일한 이유)
# =====================================================================

# ---- 5-1. 학습용(training) 오라클 이익 계산: {0, 실제발전량, 설비최대(1.0)} 3후보 ----
oracle_profit_train = np.zeros(n_train_obs)               # 학습용 오라클 이익을 담을 빈 배열
for i in range(n_train_obs):                                 # 3600개 학습 표본을 하나씩 확인
    actual_i = train_solar[i]                                  # 이 표본의 실제 발전량
    da_i = train_da_price[i]                                    # 이 표본의 DA가격
    rt_i = train_rt_price[i]                                    # 이 표본의 RT가격
    penalty_i = PENALTY_RATE * da_i                              # 부족분 벌금단가

    profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)              # 약정 0
    profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)          # 약정=실제발전량

    surplus_if_full = max(actual_i - 1.0, 0.0)                   # 약정 1.0일 때 잉여
    shortage_if_full = max(1.0 - actual_i, 0.0)                  # 약정 1.0일 때 부족량
    profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
        da_i * 1.0 + rt_i * surplus_if_full - penalty_i * shortage_if_full
    )                                                             # 약정 = 설비최대(1.0)

    oracle_profit_train[i] = max(profit_commit_0, profit_commit_actual, profit_commit_1)
    # ↑ 세 후보 중 가장 큰 값 = 이 표본의 학습용 오라클 이익

training_denominator = 0.0                                 # 학습용 오라클 이익의 합계(정규화 상수)
for i in range(n_train_obs):                                 # 3600개를 하나씩 순서대로
    training_denominator = training_denominator + oracle_profit_train[i]   # 누적

# ---- 5-2. 목적함수 계수 계산 (잉여/부족 각각에 대한 비용) ----
scale = CAPACITY_MW * DURATION_HOURS                        # 이익 스케일 상수 (30)
surplus_cost = np.zeros(n_train_obs)                          # 잉여(y_plus) 1단위당 목적함수 계수
shortage_cost = np.zeros(n_train_obs)                          # 부족(y_minus) 1단위당 목적함수 계수
for i in range(n_train_obs):                                    # 3600개를 하나씩 순서대로
    penalty_i = PENALTY_RATE * train_da_price[i]                  # 이 표본의 부족분 벌금단가
    surplus_cost[i] = (-W1 * scale * train_rt_price[i] / training_denominator) + (W2 / n_train_obs)
    shortage_cost[i] = (W1 * scale * penalty_i / training_denominator) + (W2 / n_train_obs)

binary_row_list = []                                          # 이진변수가 필요한 표본 번호를 담을 빈 리스트
for i in range(n_train_obs):                                    # 3600개를 하나씩 순서대로 확인
    if surplus_cost[i] + shortage_cost[i] < 0.0:                  # 두 비용의 합이 음수면
        binary_row_list.append(i)                                    # 이 표본 번호를 이진변수 목록에 추가
binary_rows = np.array(binary_row_list, dtype=int)             # 리스트를 numpy 배열로 변환
n_binary = len(binary_rows)                                    # 이진변수 개수
print("이진변수가 필요한 표본 개수:", n_binary, "/", n_train_obs)   # 몇 개나 필요했는지 출력

# ---- 5-3. 변수 배치: beta(4) + x(3600, 약정량) + y_plus(3600, 잉여) + y_minus(3600, 부족) + z(이진변수) ----
beta_start = 0                                                 # beta 변수 시작 위치
x_start = n_features                                            # x(약정량) 변수 시작 위치
yplus_start = n_features + n_train_obs                          # y_plus(잉여) 변수 시작 위치
yminus_start = n_features + 2 * n_train_obs                     # y_minus(부족) 변수 시작 위치
z_start = n_features + 3 * n_train_obs                          # z(이진변수) 시작 위치
n_variables = n_features + 3 * n_train_obs + n_binary            # 전체 변수 개수

# ---- 5-4. 목적함수(최소화) 계수 벡터 만들기 ----
objective = np.zeros(n_variables)                              # 목적함수 계수를 0으로 채운 빈 배열
for i in range(n_train_obs):                                     # 3600개를 하나씩 순서대로
    objective[x_start + i] = -W1 * scale * train_da_price[i] / training_denominator   # 약정량에 대한 계수
    objective[yplus_start + i] = surplus_cost[i]                                       # 잉여에 대한 계수
    objective[yminus_start + i] = shortage_cost[i]                                     # 부족에 대한 계수

# ---- 5-5. 등식 제약: (a) x = X@beta,  (b) x + y_plus - y_minus = actual ----
X_sparse = sparse.csr_matrix(X_train)                          # 입력행렬을 희소행렬로 변환
identity_n = sparse.eye(n_train_obs, format="csr")                # 3600x3600 단위행렬

equation_a = sparse.lil_matrix((n_train_obs, n_variables))       # (a) 제약을 담을 빈 행렬
equation_a[:, beta_start:beta_start + n_features] = -X_sparse      # -X@beta 부분
equation_a[:, x_start:x_start + n_train_obs] = identity_n           # +x 부분

equation_b = sparse.lil_matrix((n_train_obs, n_variables))       # (b) 제약을 담을 빈 행렬
equation_b[:, x_start:x_start + n_train_obs] = identity_n           # +x 부분
equation_b[:, yplus_start:yplus_start + n_train_obs] = identity_n   # +y_plus 부분
equation_b[:, yminus_start:yminus_start + n_train_obs] = -identity_n  # -y_minus 부분

all_equations = sparse.vstack([equation_a, equation_b], format="csr")   # (a),(b) 를 위아래로 합침
equation_right_side = np.concatenate([np.zeros(n_train_obs), train_solar])   # (a)우변=0, (b)우변=실제발전량

equality_constraint = LinearConstraint(all_equations, equation_right_side, equation_right_side)
all_constraints = [equality_constraint]                          # 제약 목록 (등식 제약부터 넣음)

# ---- 5-6. 복잡성(complementarity) 제약: 잉여/부족이 동시에 켜지지 않도록 ----
if n_binary > 0:                                                   # 이진변수가 하나라도 필요하면
    complementarity_matrix = sparse.lil_matrix((2 * n_binary, n_variables))   # 빈 제약 행렬 준비
    for k in range(n_binary):                                        # 이진변수 하나씩 순서대로
        row = binary_rows[k]                                           # 이 이진변수가 담당하는 표본 번호
        complementarity_matrix[k, yplus_start + row] = 1.0               # y_plus[row] 계수
        complementarity_matrix[k, z_start + k] = 1.0                     # + z[k]  ->  y_plus+z <= 1
        complementarity_matrix[n_binary + k, yminus_start + row] = 1.0    # y_minus[row] 계수
        complementarity_matrix[n_binary + k, z_start + k] = -1.0          # - z[k]  ->  y_minus-z <= 0

    complementarity_upper = np.concatenate([np.ones(n_binary), np.zeros(n_binary)])   # 상한
    complementarity_lower = np.full(2 * n_binary, -np.inf)                              # 하한(-무한대)

    complementarity_constraint = LinearConstraint(
        complementarity_matrix.tocsr(), complementarity_lower, complementarity_upper
    )
    all_constraints.append(complementarity_constraint)                # 제약 목록에 추가

# ---- 5-7. 변수별 상/하한 및 정수(이진) 여부 지정 ----
lower_bounds = np.concatenate([
    np.full(n_features, -np.inf),          # beta: 아래 제한 없음
    np.zeros(3 * n_train_obs + n_binary),    # x, y_plus, y_minus, z: 0 이상
])
upper_bounds = np.concatenate([
    np.full(n_features, np.inf),            # beta: 위 제한 없음
    np.ones(3 * n_train_obs + n_binary),      # x, y_plus, y_minus, z: 1 이하
])
variable_bounds = Bounds(lower_bounds, upper_bounds)

integrality = np.zeros(n_variables, dtype=int)                    # 기본은 전부 실수(0)
for k in range(n_binary):                                          # z 변수들만
    integrality[z_start + k] = 1                                     # 정수(이진)로 지정

# ---- 5-8. MILP 풀기 ----
milp_result = milp(
    c=objective,
    integrality=integrality,
    bounds=variable_bounds,
    constraints=all_constraints,
    options={"mip_rel_gap": 1e-9},
)

mlr_coefficients = milp_result.x[beta_start:beta_start + n_features]   # beta(4개)만 뽑아 저장
print(f"MLR MILP 완료 (성공 여부: {milp_result.success}), 계수: {mlr_coefficients}")


# =====================================================================
# 6. 테스트 구간 예측 (한 번에 전체 예측)
# =====================================================================

test_forecast = np.zeros(n_test_obs)                        # 예측 결과를 담을 빈 배열
for i in range(n_test_obs):                                    # 테스트 1200개 행을 하나씩 순서대로
    raw_prediction = np.dot(mlr_coefficients, X_test[i])          # 계수와 입력을 곱해서 더함
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
# 8. optimality gap 계산 - 평가용 오라클 {0, 실제발전량} 만 사용
#    (5번의 학습용 오라클과는 다른, 논문 Eq.13 그대로의 오라클을 씀)
# =====================================================================

sum_of_realized_profit = 0.0             # 실제(제안모형 예측 기반) 총 이익을 누적할 변수
sum_of_oracle_profit = 0.0               # 오라클(사후 최적) 총 이익을 누적할 변수

for i in range(n_test_obs):                # 테스트 1200개 관측치를 하나씩 순서대로 처리

    actual_i = test_solar[i]                 # 이 시간의 실제 발전량
    commitment_i = test_forecast[i]          # 이 시간의 제안모형 예측값 = 일간전 약정량
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
    oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual)          # 둘 중 더 큰 쪽 (평가용 오라클)
    sum_of_oracle_profit = sum_of_oracle_profit + oracle_profit_i

optimality_gap_percent = 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit


# =====================================================================
# 9. 결과 출력
# =====================================================================

print()
print("=== 논문 제안 모형 MLR (Eq.10 MILP, W1=1, W2=20) - 블록9 결과 ===")
print(f"nRMSE          = {nrmse_percent:.6f} %   (논문: {PAPER_NRMSE:.2f} %)")
print(f"optimality gap = {optimality_gap_percent:.6f} %   (논문: {PAPER_GAP:.2f} %)")
print()
print("| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| 논문 제안 모형 MLR | {PAPER_NRMSE:.2f}% | {nrmse_percent:.6f}% | {PAPER_GAP:.2f}% | {optimality_gap_percent:.6f}% "
    f"| {nrmse_percent - PAPER_NRMSE:+.6f}%p | {optimality_gap_percent - PAPER_GAP:+.6f}%p |"
)
