# anypoint_to_3d.py
# 功能：
# 1. 讀取 loftr_results/anypoint.csv (Anypoint 匹配點)
# 2. 轉換為 Fisheye 座標
# 3. 計算 3D 座標 (X, Y, Z, Depth)
# 4. 輸出 output_3d/3d_raw.csv (包含所有原始數據，不過濾)

import multiprocessing
import os
from typing import Tuple

import numpy as np
import pandas as pd
from moil_3d import Moil3dAlgorithm
from moildev import Moildev

# ====================== 設定區塊 ======================
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANYPOINT_CSV = "loftr_results/anypoint.csv"  # 你的 anypoint 匹配結果
LEFT_JSON = os.path.join(path, "wxsj_7730_4.json")
RIGHT_JSON = os.path.join(path, "wxsj_7730_2.json")

L_CAM_3D = (-30.0, 0.0, 0.0)  # mm
R_CAM_3D = (30.0, 0.0, 0.0)

OUTPUT_DIR = "output_3d"

ANYPOINT_PAN = 0
ANYPOINT_TILT = 0
ANYPOINT_ZOOM = 1

# ====================== 多執行緒全域變數 ======================
global_left_moil = None
global_right_moil = None
global_L_CAM_3D = None
global_R_CAM_3D = None


def init_worker() -> None:
    global global_left_moil, global_right_moil, global_L_CAM_3D, global_R_CAM_3D
    global_left_moil = Moildev(LEFT_JSON)
    global_right_moil = Moildev(RIGHT_JSON)
    global_L_CAM_3D = L_CAM_3D
    global_R_CAM_3D = R_CAM_3D


def worker_compute_depth(
    args: Tuple[int, np.ndarray, np.ndarray],
) -> Tuple[int, float, Tuple[float, float, float]]:
    idx, l_fish, r_fish = args
    try:
        Lu, Lv = l_fish
        Ru, Rv = r_fish
        l_px = (int(round(Lu)), int(round(Lv)))
        r_px = (int(round(Ru)), int(round(Rv)))

        X, Y, Z = Moil3dAlgorithm.single_3d_coordinate(
            global_left_moil,
            global_right_moil,
            global_L_CAM_3D,
            global_R_CAM_3D,
            l_px,
            r_px,
        )
        depth_m = np.sqrt(X**2 + Y**2 + Z**2) / 1000.0

        if not np.isfinite(depth_m):
            depth_m = 0.0
            X = Y = Z = 0.0

        return idx, depth_m, (X, Y, Z)
    except Exception as e:
        print(f"Error at point {idx}: {e}")
        return idx, 0.0, (0.0, 0.0, 0.0)


# ====================== 主程式 ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"載入 {ANYPOINT_CSV}")
    df = pd.read_csv(ANYPOINT_CSV)
    need_cols = ["L_anypoint_x", "L_anypoint_y", "R_anypoint_x", "R_anypoint_y"]
    if not all(c in df.columns for c in need_cols):
        raise ValueError(f"CSV 缺少必要欄位：{need_cols}")

    L_any_x = df["L_anypoint_x"].values.astype(np.float32)
    L_any_y = df["L_anypoint_y"].values.astype(np.float32)
    R_any_x = df["R_anypoint_x"].values.astype(np.float32)
    R_any_y = df["R_anypoint_y"].values.astype(np.float32)

    # 建立 Moildev
    left_moil = Moildev(LEFT_JSON)
    right_moil = Moildev(RIGHT_JSON)

    # anypoint → fisheye 映射表
    mapX_L, mapY_L = left_moil.maps_anypoint_mode1(
        ANYPOINT_PAN, ANYPOINT_TILT, ANYPOINT_ZOOM
    )
    mapX_R, mapY_R = right_moil.maps_anypoint_mode1(
        ANYPOINT_PAN, ANYPOINT_TILT, ANYPOINT_ZOOM
    )

    def any_to_fish(
        mapX: np.ndarray, mapY: np.ndarray, x_any: np.ndarray, y_any: np.ndarray
    ) -> np.ndarray:
        H, W = mapX.shape
        xi = np.clip(np.round(x_any).astype(int), 0, W - 1)
        yi = np.clip(np.round(y_any).astype(int), 0, H - 1)
        return np.stack((mapX[yi, xi], mapY[yi, xi]), axis=1)

    print("轉換 anypoint → fisheye 座標...")
    left_fisheye = any_to_fish(mapX_L, mapY_L, L_any_x, L_any_y)
    right_fisheye = any_to_fish(mapX_R, mapY_R, R_any_x, R_any_y)

    # 多執行緒算 3D
    print("開始 3D 計算 (保留所有點)...")
    with multiprocessing.Pool(
        processes=max(1, multiprocessing.cpu_count() - 1), initializer=init_worker
    ) as pool:
        tasks = [(i, left_fisheye[i], right_fisheye[i]) for i in range(len(df))]
        results = pool.map(worker_compute_depth, tasks)

    # 按照 index 排序
    results.sort(key=lambda x: x[0])

    depths = np.array([r[1] for r in results])
    coords = np.array([r[2] for r in results])

    # 儲存包含"所有"資訊的 RAW CSV
    # 這樣後處理程式 (process_3d.py) 才有辦法畫圖 (需要 Fisheye 座標 和 Anypoint 座標)
    raw_df = pd.DataFrame(
        {
            "L_any_x": L_any_x,
            "L_any_y": L_any_y,
            "R_any_x": R_any_x,  # 雖然主要用左圖畫圖，但保留右圖資訊也好
            "R_any_y": R_any_y,
            "L_fisheye_u": left_fisheye[:, 0],
            "L_fisheye_v": left_fisheye[:, 1],
            "R_fisheye_u": right_fisheye[:, 0],
            "R_fisheye_v": right_fisheye[:, 1],
            "X_mm": coords[:, 0],
            "Y_mm": coords[:, 1],
            "Z_mm": coords[:, 2],
            "Depth_m": depths,
        }
    )

    output_csv = os.path.join(OUTPUT_DIR, "3d_raw.csv")
    raw_df.to_csv(output_csv, index=False, float_format="%.4f")

    print(f"計算完成！原始數據已儲存至: {output_csv}")
    print("請執行 process_3d_results.py 進行過濾與視覺化。")


if __name__ == "__main__":
    main()
