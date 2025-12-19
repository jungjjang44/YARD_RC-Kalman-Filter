import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === JSON 로드 ===
# 저장 경로가 다를 수 있으니 우선 업로드 파일 경로를 기본값으로 사용
json_path_candidates = [
    Path("ekf_estimates.json"),
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

# ---- 안전한 키 매핑 (신규 키 우선, 없으면 구키 fallback) ----
def arr(key_list, dtype=float):
    for k in key_list:
        if k in est:
            return np.asarray(est[k], dtype=dtype)
    return np.array([], dtype=dtype)

time  = arr(["Time"])

# EKF 추정 (신규 키: *_est)
x     = arr(["x_est","x"])
y     = arr(["y_est","y"])
vx    = arr(["vx_est","vx"])
vy    = arr(["vy_est","vy"])
yaw   = arr(["yaw_est","yaw"])

# 실측/로컬라이제이션 (신규 키: *_meas / 과거 키: *_real, Yaw_real)
x_r   = arr(["x_meas","x_real"])
y_r   = arr(["y_meas","y_real"])
yaw_r = arr(["yaw_meas","Yaw_real","yaw_real"])

# 중력보정 가속도 로그 (신규 키: ax_used, ay_used)
ax_r  = arr(["ax_used","ax_real"])
ay_r  = arr(["ay_used","ay_real"])

# (선택) yaw rate
wz_r  = arr(["wz_used","yaw_rate"])

# --- 시간 정렬(비단조 방지) ---
if time.size > 0:
    order = np.argsort(time)
    def sort_if_same_len(a):
        return a[order] if a.size == time.size else a
    time = time[order]
    x    = sort_if_same_len(x)
    y    = sort_if_same_len(y)
    vx   = sort_if_same_len(vx)
    vy   = sort_if_same_len(vy)
    yaw  = sort_if_same_len(yaw)
    x_r  = sort_if_same_len(x_r)
    y_r  = sort_if_same_len(y_r)
    yaw_r= sort_if_same_len(yaw_r)
    ax_r = sort_if_same_len(ax_r)
    ay_r = sort_if_same_len(ay_r)
    wz_r = sort_if_same_len(wz_r)

# --- 위치 미분으로 속도/속력 산출 ---
if time.size > 1 and x_r.size == time.size and y_r.size == time.size:
    vx_from_pos = np.gradient(x_r, time)
    vy_from_pos = np.gradient(y_r, time)
    speed_from_pos = np.hypot(vx_from_pos, vy_from_pos)
else:
    vx_from_pos = np.array([])
    vy_from_pos = np.array([])
    speed_from_pos = np.array([])

# --- EKF 속력 ---
speed_ekf = np.hypot(vx, vy) if vx.size and vy.size else np.array([])

def safe_legend():
    h, lab = plt.gca().get_legend_handles_labels()
    if lab: plt.legend()

# === 플로팅 ===
plt.figure(figsize=(14, 12))

# (1) 궤적
plt.subplot(3, 3, 1)
if x.size and y.size:
    plt.plot(x, y, label="Trajectory (EKF)", linewidth=1.2)
if x_r.size and y_r.size:
    plt.plot(x_r, y_r, label="Trajectory (Real)", linewidth=1.0)
plt.xlabel("X [m]"); plt.ylabel("Y [m]")
plt.title("2D Trajectory"); plt.axis("equal"); plt.grid(True); safe_legend()

# (2) X
plt.subplot(3, 3, 2)
if time.size and x.size:   plt.plot(time, x,   label="x (EKF)")
if time.size and x_r.size: plt.plot(time, x_r, label="x_meas")
plt.title("x vs x_meas"); plt.xlabel("Time [s]"); plt.ylabel("X [m]")
plt.grid(True); safe_legend()

# (3) Y
plt.subplot(3, 3, 3)
if time.size and y.size:   plt.plot(time, y,   label="y (EKF)")
if time.size and y_r.size: plt.plot(time, y_r, label="y_meas")
plt.title("y vs y_meas"); plt.xlabel("Time [s]"); plt.ylabel("Y [m]")
plt.grid(True); safe_legend()

# (4) Vx
plt.subplot(3, 3, 4)
if time.size and vx_from_pos.size: plt.plot(time, vx_from_pos, label="Vx (from x_meas)")
if time.size and vx.size:          plt.plot(time, vx,          label="Vx (EKF)")
plt.title("Vx Comparison"); plt.xlabel("Time [s]"); plt.ylabel("Vx [m/s]")
plt.grid(True); safe_legend()

# (5) Vy
plt.subplot(3, 3, 5)
if time.size and vy_from_pos.size: plt.plot(time, vy_from_pos, label="Vy (from y_meas)")
if time.size and vy.size:          plt.plot(time, vy,          label="Vy (EKF)")
plt.title("Vy Comparison"); plt.xlabel("Time [s]"); plt.ylabel("Vy [m/s]")
plt.grid(True); safe_legend()

# (6) Yaw
plt.subplot(3, 3, 6)
if time.size and yaw.size:   plt.plot(time, yaw,   label="yaw (EKF)")
if time.size and yaw_r.size: plt.plot(time, yaw_r, label="yaw_meas")
plt.title("Yaw Comparison"); plt.xlabel("Time [s]"); plt.ylabel("Yaw [rad]")
plt.grid(True); safe_legend()

# (7) ax (gravity-compensated)
plt.subplot(3, 3, 7)
if time.size and ax_r.size: plt.plot(time, ax_r, label="ax_used")
plt.title("ax_used (gravity-compensated)"); plt.xlabel("Time [s]"); plt.ylabel("ax [m/s²]")
plt.grid(True); safe_legend()

# (8) ay (gravity-compensated)
plt.subplot(3, 3, 8)
if time.size and ay_r.size: plt.plot(time, ay_r, label="ay_used")
plt.title("ay_used (gravity-compensated)"); plt.xlabel("Time [s]"); plt.ylabel("ay [m/s²]")
plt.grid(True); safe_legend()

# (9) Speed
plt.subplot(3, 3, 9)
if time.size and speed_from_pos.size: plt.plot(time, speed_from_pos, label="speed (from pos-deriv)")
if time.size and speed_ekf.size:      plt.plot(time, speed_ekf,      label="speed (EKF)")
plt.title("Speed Comparison"); plt.xlabel("Time [s]"); plt.ylabel("Speed [m/s]")
plt.grid(True); safe_legend()

plt.tight_layout()
plt.show()
