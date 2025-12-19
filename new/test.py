import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 입력 파일 =====
FILE = "data_log.json"
if not Path(FILE).exists():
    FILE = "/mnt/data/data_log_go_straight.json"

# ===== 파라미터 =====
STATIC_WINDOW_SEC = 2.0      # 초기 정지 구간 기대치
ALT_FRACTION     = 0.2       # 로그가 짧을 때 앞부분 20% 사용
G                = 9.81
DELTA_G_THRESH   = 0.15      # |a|-g 편차 허용(정지/틸트 판정)
USE_MEDIAN       = True      # 평균 대신 median 사용

# ===== JSON 로드 =====
with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

imu = data["imu"]
t  = np.array([r["t"] for r in imu], dtype=float)
ax = np.array([r["ax_mps2"] for r in imu], dtype=float)
ay = np.array([r["ay_mps2"] for r in imu], dtype=float)
az = np.array([r["az_mps2"] for r in imu], dtype=float)

# ===== 시간 정렬/기준화 =====
order = np.argsort(t)
t, ax, ay, az = t[order], ax[order], ay[order], az[order]
t_rel = t - t[0]

# ===== 정지 후보 마스크 구성 =====
if t_rel[-1] >= STATIC_WINDOW_SEC:
    mask0 = (t_rel <= STATIC_WINDOW_SEC)
else:
    n = max(10, int(ALT_FRACTION * len(t_rel)))
    mask0 = np.zeros_like(t_rel, dtype=bool); mask0[:n] = True

a_norm = np.sqrt(ax**2 + ay**2 + az**2)
mask_g = np.abs(a_norm - G) < DELTA_G_THRESH
mask  = mask0 & mask_g       # 초기 구간 + g에 가까운 샘플만 사용

# ===== 틸트 추정(가속도 기반) =====
def robust_avg(v, mask):
    return np.median(v[mask]) if USE_MEDIAN else np.mean(v[mask])

g_est_b = np.array([
    robust_avg(ax, mask),
    robust_avg(ay, mask),
    robust_avg(az, mask),
])

# roll, pitch (yaw 불필요)
roll  = np.arctan2(g_est_b[1], g_est_b[2])
pitch = np.arctan2(-g_est_b[0], np.sqrt(g_est_b[1]**2 + g_est_b[2]**2))
yaw   = 0.0

def Rzyx(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

R_b2w = Rzyx(roll, pitch, yaw)
g_w = np.array([0.0, 0.0, G])
g_b_pred = R_b2w.T @ g_w   # body-frame gravity (예측치)

# ===== 중력 보상 (body frame) =====
a_b = np.vstack([ax, ay, az]).T
a_lin_b = a_b - g_b_pred  # body 선형가속도

# ===== 진단 출력 =====
print("[Tilt deg] roll, pitch =", np.degrees([roll, pitch]))
print("[g_b_pred] =", g_b_pred)  # 이 벡터가 실제 빼는 중력 성분(바디 프레임)
print("Raw mean [m/s^2] =", np.mean(a_b, axis=0))
print("Lin  mean [m/s^2] =", np.mean(a_lin_b, axis=0))
print("Selected static samples =", int(mask.sum()), "/", len(mask))

# ===== 플롯: 보상 전/후(ax, ay, az) =====
labels = ["X", "Y", "Z"]
for i, lbl in enumerate(labels):
    plt.figure(figsize=(10,4))
    plt.plot(t_rel, a_b[:, i], label=f"{lbl}-raw (body)", linewidth=1.0)
    plt.plot(t_rel, a_lin_b[:, i], label=f"{lbl}-gravity-comp (body)", linewidth=1.0)
    plt.title(f"{lbl}-axis Acceleration: Raw vs Gravity-compensated")
    plt.xlabel("Time [s]"); plt.ylabel("Acceleration [m/s²]")
    plt.grid(True); plt.legend()
    plt.show()

# ===== 저장(검증용) =====
out = [{"t": float(t_rel[i]), "ax": float(ax[i]), "ay": float(ay[i]), "az": float(az[i]),
        "ax_lin": float(a_lin_b[i,0]), "ay_lin": float(a_lin_b[i,1]), "az_lin": float(a_lin_b[i,2])}
        for i in range(len(t_rel))]
with open("gravity_compensated_strict.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("→ gravity_compensated_strict.json 저장 완료")
