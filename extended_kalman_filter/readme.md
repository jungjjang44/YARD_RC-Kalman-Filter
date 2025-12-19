# Extended Kalman Filter (EKF) ROS2 Node

이 노드는 **IMU(/imu/data)** 와 **외부 위치·요(/localization_info)** 측정을 융합하여 
로봇의 2D 평면 상태 **[x, y, vx, vy, yaw]** 를 추정하는 **Extended Kalman Filter(EKF)** 구현이다.

---

## 1. 상태(State) 정의

상태 벡터는 다음 5차원으로 정의된다.

- **x, y** : 위치 [m]
- **vx, vy** : 속도 [m/s]
- **yaw** : 헤딩 [rad]

---

## 2. 입·출력(ROS2 I/O)

### Subscriptions
- `/imu/data` (**sensor_msgs/Imu**) 
  - orientation(quaternion) → roll/pitch 추출 
  - linear_acceleration(ax, ay, az) 사용 
  - angular_velocity.z 는 **deg/s로 가정 후 rad/s 변환**
- `/localization_info` (**custom_msgs/Localization**) 
  - px, py, p_yaw(deg) 사용 → yaw는 **deg → rad 변환**

### Publications
- `/filtered_v` (**std_msgs/Float64MultiArray**) 
  - `[vx_est, vy_est]`
- `/filtered_a` (**std_msgs/Float64MultiArray**) 
  - EKF에 실제 입력으로 사용된 보정 가속도 `[ax_used, ay_used]`
- `/raw_a` (**std_msgs/Float64MultiArray**) 
  - 포즈 콜백 구간에서 누적 평균된 원시 가속도(디버그)

---

## 3. 동작 개요(Flow)

이 구현은 **Pose(/localization_info) 콜백을 기준으로 EKF 1스텝을 수행**한다.

1) IMU 콜백이 들어오는 동안, Pose 콜백 사이 구간의 IMU 값을 누적(단순 평균)한다. 
2) Pose 콜백이 들어오면:
   - dt를 계산하고(클리핑 포함), 누적된 IMU 평균값을 꺼낸다.
   - roll/pitch 기반으로 중력 성분을 제거해 **수평 가속도만** 사용한다.
   - EKF Prediction → Measurement Update를 순서대로 수행한다.
---

## 4. 센서 보정/가속도 전처리

### 4.1 중력 성분 제거(수평 투영)
IMU orientation에서 구한 roll/pitch로 중력 방향 단위벡터 `g_hat`를 만들고,
가속도 벡터를 수평면으로 투영하여 중력 성분을 제거한다.

### 4.2 IMU 축 보정(90° 회전 보정)
코드 상에서 IMU가 시계방향 90° 회전된 것으로 가정하여,
- `ax_use = ay_corr`
- `ay_use = -ax_corr`
로 축을 교정한다.

### 4.3 HARDCODING 옵션(바이어스 보정)
`HARDCODING=True`일 때는 특정 상수 오프셋을 빼는 임시 보정 로직이 적용된다.

---

## 5. EKF 모델

### 5.1 Prediction(비선형 상태식)
yaw(psi)에 대해 (c=cos, s=sin)로 두고,
보정 가속도(ax_use, ay_use)로 다음과 같이 상태를 예측한다.

- x, y: 속도 및 가속도 적분
- vx, vy: 가속도 적분
- yaw: `yaw += wz * dt`

### 5.2 공분산 예측
선형화 행렬 A와 입력 행렬 B를 사용하여 공분산을 예측한다.

`P = A P A^T + B Σu B^T + Q`

### 5.3 Measurement Update
측정 z는 `[x_meas, y_meas, yaw_meas]` 이며, H는 x/y/yaw만 관측하는 형태다.
표준 EKF 업데이트를 수행한다.

---

## 6. 주요 파라미터(코드 상수)

- 상태 차원: `NX=5`, 측정 차원: `NZ=3`
- 초기 공분산 대각: `PS=0.1`
- dt 클리핑: `(1e-3, 0.2)`
- 프로세스 노이즈 Q 구성: `sTracker, sVelocity, sYaw`
- IMU 입력 노이즈(Σu): `sigma_ax, sigma_ay, sigma_yawrate`
- 측정 노이즈 R: `varGPS`, `varyaw(deg)` → rad 변환 후 사용 
> 팁: 실제 센서 노이즈 특성에 맞춰 `varGPS`, `varyaw`, `sigma_*`, `s*` 튜닝이 EKF 품질을 가장 크게 좌우한다.

---

## 7. 초기화 로직

첫 Pose 수신 시점에 IMU를 한 번이라도 받았으면,
상태를 `[x, y, 0, 0, yaw]`로 초기화하고 측정 업데이트만 1회 수행한다.

---

## 8. 로그/결과 저장(JSON)

노드 종료 시 EKF 추정치와 측정치를 JSON으로 저장한다.
기본 저장 경로:
- `src/extended_kalman_filter/data/ekf_estimates.json`

저장 항목 예:
- Time, x/y/vx/vy/yaw 추정치
- x/y/yaw 측정치
- ax/ay/wz 사용값

