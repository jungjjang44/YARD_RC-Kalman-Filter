
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === JSON 로드 ===
json_path_candidates = [
    Path("ekf_estimates.json"),
    Path("/mnt/data/ekf_estimates (1).json"),   # 업로드된 파일
    Path("/mnt/data/ekf_estimates.json"),
    Path("src/extended_kalman_filter/data/ekf_estimates.json"),
]
for p in json_path_candidates:
    if p.exists():
        json_path = p
        break
else:
    raise FileNotFoundError("ekf_estimates.json 파일을 찾을 수 없습니다.")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
est = data.get("estimates", {})

# ---- 안전한 키 매핑 ----
def arr(key_list, dtype=float):
    for k in key_list:
        if k in est:
            return np.asarray(est[k], dtype=dtype)
    return np.array([], dtype=dtype)

time  = arr(["Time"])
x     = arr(["x_est","x"])
y     = arr(["y_est","y"])
vx    = arr(["vx_est","vx"])
vy    = arr(["vy_est","vy"])
yaw   = arr(["yaw_est","yaw"])
x_r   = arr(["x_meas","x_real"])
y_r   = arr(["y_meas","y_real"])
yaw_r = arr(["yaw_meas","Yaw_real","yaw_real"])
ax_r  = arr(["ax_used","ax_real"])
ay_r  = arr(["ay_used","ay_real"])
wz_r  = arr(["wz_used","yaw_rate"])  # [rad/s] 가정

# --- 시간 정렬 및 0초 기준화 ---
if time.size > 0:
    order = np.argsort(time)
    def sort_if_same_len(a): return a[order] if a.size == time.size else a
    time = time[order]
    t0 = time[0]
    time = time - t0  # 0초 기준으로 변환

    x, y, vx, vy, yaw, x_r, y_r, yaw_r, ax_r, ay_r, wz_r = map(sort_if_same_len,
        [x, y, vx, vy, yaw, x_r, y_r, yaw_r, ax_r, ay_r, wz_r]
    )

# --- 스케일 14배 적용 ---
scale = 14.0
x, y, x_r, y_r = x*scale, y*scale, x_r*scale, y_r*scale
vx, vy, ax_r, ay_r = vx*scale, vy*scale, ax_r, ay_r
speed_ekf = np.hypot(vx, vy)

if time.size > 1 and x_r.size == time.size and y_r.size == time.size:
    vx_from_pos = np.gradient(x_r, time)
    vy_from_pos = np.gradient(y_r, time)
    speed_from_pos = np.hypot(vx_from_pos, vy_from_pos)
else:
    vx_from_pos = vy_from_pos = speed_from_pos = np.array([])

# --- yaw_rate (d/dt of yaw_est) 및 오차 계산 ---
if time.size > 1 and yaw.size == time.size:
    # unwrap으로 2π 점프 방지
    yaw_unwrapped = np.unwrap(yaw)
    yawrate_from_yaw = np.gradient(yaw_unwrapped, time)  # [rad/s]
else:
    yawrate_from_yaw = np.array([])

if yawrate_from_yaw.size and wz_r.size == time.size:
    yawrate_error = yawrate_from_yaw - wz_r  # 정의: (d/dt yaw_est) - wz_used
    rmse = np.sqrt(np.mean((yawrate_error)**2))
else:
    yawrate_error = np.array([])
    rmse = np.nan

# === 플롯 ===
plt.figure(figsize=(14, 16))

def safe_legend():
    h, lab = plt.gca().get_legend_handles_labels()
    if lab: plt.legend()

# (1) Trajectory
plt.subplot(4, 3, 1)
if x.size and y.size: plt.plot(x, y, label="Trajectory (EKF)", linewidth=1.2)
if x_r.size and y_r.size: plt.plot(x_r, y_r, label="Trajectory (Real)", linewidth=1.0)
plt.xlabel("X [m]"); plt.ylabel("Y [m]")
plt.title("2D Trajectory (scaled ×14)"); plt.axis("equal"); plt.grid(True); safe_legend()

# (2) X
plt.subplot(4, 3, 2)
if x.size: plt.plot(time, x, label="x (EKF)")
if x_r.size: plt.plot(time, x_r, label="x_meas")
plt.title("x vs x_meas (×14)"); plt.xlabel("Time [s]"); plt.ylabel("X [m]")
plt.grid(True); safe_legend()

# (3) Y
plt.subplot(4, 3, 3)
if y.size: plt.plot(time, y, label="y (EKF)")
if y_r.size: plt.plot(time, y_r, label="y_meas")
plt.title("y vs y_meas (×14)"); plt.xlabel("Time [s]"); plt.ylabel("Y [m]")
plt.grid(True); safe_legend()

# (4) Vx
plt.subplot(4, 3, 4)
if vx_from_pos.size: plt.plot(time, vx_from_pos, label="Vx (from x_meas)")
if vx.size: plt.plot(time, vx, label="Vx (EKF)")
plt.title("Vx Comparison (×14)"); plt.xlabel("Time [s]"); plt.ylabel("Vx [m/s]")
plt.grid(True); safe_legend()

# (5) Vy
plt.subplot(4, 3, 5)
if vy_from_pos.size: plt.plot(time, vy_from_pos, label="Vy (from y_meas)")
if vy.size: plt.plot(time, vy, label="Vy (EKF)")
plt.title("Vy Comparison (×14)"); plt.xlabel("Time [s]"); plt.ylabel("Vy [m/s]")
plt.grid(True); safe_legend()

# (6) Yaw
plt.subplot(4, 3, 6)
if yaw.size: plt.plot(time, yaw, label="yaw (EKF)")
if yaw_r.size: plt.plot(time, yaw_r, label="yaw_meas")
plt.title("Yaw Comparison"); plt.xlabel("Time [s]"); plt.ylabel("Yaw [rad]")
plt.grid(True); safe_legend()

# (7) ax
plt.subplot(4, 3, 7)
if ax_r.size: plt.plot(time, ax_r, label="ax_used")
plt.title("ax_used (gravity-comp., ×14)"); plt.xlabel("Time [s]"); plt.ylabel("ax [m/s²]")
plt.grid(True); safe_legend()

# (8) ay
plt.subplot(4, 3, 8)
if ay_r.size: plt.plot(time, ay_r, label="ay_used")
plt.title("ay_used (gravity-comp., ×14)"); plt.xlabel("Time [s]"); plt.ylabel("ay [m/s²]")
plt.grid(True); safe_legend()

# (9) Speed
plt.subplot(4, 3, 9)
if speed_from_pos.size: plt.plot(time, speed_from_pos, label="speed (from pos-deriv)")
if speed_ekf.size: plt.plot(time, speed_ekf, label="speed (EKF)")
plt.title("Speed Comparison (×14)"); plt.xlabel("Time [s]"); plt.ylabel("Speed [m/s]")
plt.grid(True); safe_legend()

# (10) Yaw Rate Error subplot
plt.subplot(4, 3, 10)
if yawrate_from_yaw.size: 
    plt.plot(time, yawrate_from_yaw, label="d/dt(yaw_est)")
if wz_r.size: 
    plt.plot(time, wz_r, label="wz_used")
# if yawrate_error.size:
#     plt.plot(time, yawrate_error, label=f"error = d/dt(yaw_est) - wz_used (RMSE={rmse:.4f})", linewidth=1.2)
plt.title("Yaw Rate & Error"); plt.xlabel("Time [s]"); plt.ylabel("Yaw rate [rad/s]")
plt.grid(True); safe_legend()

plt.tight_layout()
plt.show()
