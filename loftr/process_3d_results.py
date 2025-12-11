# process_3d_results.py
# 功能：
# 1. 讀取 output_3d/3d_raw.csv (由 anypoint_to_3d.py 產生)
# 2. 套用 MAX_DEPTH 過濾
# 3. 輸出過濾後的 CSV (3d.csv, fisheye.csv)
# 4. 產生視覺化圖表 (Plotly 2D, Overlay, Side-by-side)

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import cm

# ====================== 設定區塊 ======================
# 重要：這裡控制最大深度！
MAX_DEPTH = 1.0  # 公尺

# 路徑設定
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_RAW_CSV = "output_3d/3d_raw.csv"
OUTPUT_DIR = "output_3d"
LEFT_ANYPOINT_IMAGE = "frame_left_anypoint.png"  # 修正讀取位置
LEFT_FISHEYE_ORIG = "frame_left_org.png"
RIGHT_FISHEYE_ORIG = "frame_right_org.png"
LEFT_JSON = os.path.join(path, "wxsj_7730_4.json")  # 用於取得圖片尺寸


# ====================== 主程式 ======================
def main():
    if not os.path.exists(INPUT_RAW_CSV):
        print(f"找不到 {INPUT_RAW_CSV}，請先執行 anypoint_to_3d.py")
        return

    print(f"讀取 {INPUT_RAW_CSV}...")
    df = pd.read_csv(INPUT_RAW_CSV)

    # 1. 過濾邏輯
    print(f"執行過濾 (MAX_DEPTH = {MAX_DEPTH} m)...")
    # 保留 Depth > 0 且 Depth <= MAX_DEPTH
    valid_mask = (df["Depth_m"] > 0) & (df["Depth_m"] <= MAX_DEPTH)
    df_valid = df[valid_mask].copy()

    # 按照 index 排序 (如果不放心的話，雖然通常已經排好了)
    # df 裡沒有 index 欄位，但它是 dataframe，順序是固定的

    if len(df_valid) == 0:
        print("警告：過濾後沒有任何點剩下！請檢查 MAX_DEPTH 設定。")
        return

    print(f"原始點數: {len(df)}, 過濾後點數: {len(df_valid)}")

    # 2. 重建 numpy array 供後續繪圖使用
    L_any_x = df_valid["L_any_x"].values
    L_any_y = df_valid["L_any_y"].values
    depths = df_valid["Depth_m"].values

    # 3. 輸出過濾後的 CSV (相容舊格式)
    # fisheye.csv
    fisheye_df = pd.DataFrame(
        {
            "L_fisheye_u": df_valid["L_fisheye_u"],
            "L_fisheye_v": df_valid["L_fisheye_v"],
            "R_fisheye_u": df_valid["R_fisheye_u"],
            "R_fisheye_v": df_valid["R_fisheye_v"],
        }
    )
    fisheye_df.to_csv(
        os.path.join(OUTPUT_DIR, "fisheye.csv"), index=False, float_format="%.3f"
    )

    # 3d.csv
    d3_df = pd.DataFrame(
        {
            "X_mm": df_valid["X_mm"],
            "Y_mm": df_valid["Y_mm"],
            "Z_mm": df_valid["Z_mm"],
            "Depth_m": depths,
        }
    )
    d3_df.to_csv(os.path.join(OUTPUT_DIR, "3d.csv"), index=False, float_format="%.3f")
    print("已輸出 fisheye.csv 與 3d.csv")

    # ====================== 視覺化準備 ======================
    # 取得圖片尺寸 (優先從圖片讀取，否則從 moildev 讀取)
    if os.path.exists(LEFT_ANYPOINT_IMAGE):
        base_img = cv2.cvtColor(cv2.imread(LEFT_ANYPOINT_IMAGE), cv2.COLOR_BGR2RGB)
        h, w = base_img.shape[:2]

    # 計算顏色範圍
    vmin = depths.min() if len(depths) > 0 else 0.0
    vmax = depths.max() if len(depths) > 0 else 1.0
    print(f"顏色映射範圍: {vmin:.3f}m ~ {vmax:.3f}m")

    # ====================== Plotly 2D 視覺化 ======================
    print("產生 Plotly 2D 深度圖 (output_3d/depth_sparse_jet.png)...")
    colored_rgb = None  # 初始化
    try:
        fig_2d = px.scatter(
            df_valid,
            x="L_any_x",
            y="L_any_y",
            color="Depth_m",
            color_continuous_scale="jet",
            range_color=[vmin, vmax],
            width=w,
            height=h,
        )
        # 設定固定範圍以確保疊圖對齊 (0,0 在左上角)
        # Y軸: [h, 0] 表示下緣是 h, 上緣是 0 (因為 Plotly 預設 Y 向上，這裡強制反轉座標系並鎖定範圍)
        fig_2d.update_xaxes(showgrid=False, visible=False, range=[0, w])
        fig_2d.update_yaxes(showgrid=False, visible=False, range=[h, 0])

        fig_2d.update_traces(marker=dict(size=12))  # 點的大小加大
        fig_2d.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="black",
            paper_bgcolor="black",
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                tickfont=dict(size=30),
                title=dict(font=dict(size=30)),
            ),
        )

        save_path_jet = os.path.join(OUTPUT_DIR, "depth_sparse_jet.png")
        fig_2d.write_image(save_path_jet)

        # 讀回圖片用於疊加
        colored_rgb = cv2.cvtColor(cv2.imread(save_path_jet), cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Plotly 繪圖失敗: {e}，切換至 OpenCV 繪圖...")
        # Fallback logic defined below

    # ====================== Overlay 視覺化 (透明疊加) ======================
    # 策略：如果 Plotly 成功產生圖片，直接用它來疊加 (保證顏色一樣)
    # 如果失敗，才用 OpenCV 重畫
    print("產生疊加圖 (depth_sparse_overlay.png)...")

    overlay = base_img.copy()

    if colored_rgb is not None:
        # 使用 Plotly 的圖進行疊加
        # 假設背景是全黑 (0,0,0)，只疊加非黑色的點
        # 注意: Plotly 存圖可能因為壓縮產生邊緣雜訊，或因為有點大導致重疊
        # 簡單的 Mask: 只要不是全黑就算有資料
        mask = np.any(colored_rgb > 0, axis=2)  # True if any channel > 0

        # 混合：原圖 0.4 + Plotly圖 0.6
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                (base_img[:, :, c] * 0.4 + colored_rgb[:, :, c] * 0.6).astype(np.uint8),
                base_img[:, :, c],
            )
    else:
        # Fallback: OpenCV 重畫
        depth_vis_ov = np.full((h, w), np.nan, dtype=np.float32)
        sorted_indices = np.argsort(depths)[::-1]
        for i in sorted_indices:
            cx = int(round(L_any_x[i]))
            cy = int(round(L_any_y[i]))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(depth_vis_ov, (cx, cy), 7, float(depths[i]), -1)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colored_mpl = (cm.jet(norm(depth_vis_ov))[:, :, :3] * 255).astype(np.uint8)
        mask = ~np.isnan(depth_vis_ov)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                (base_img[:, :, c] * 0.4 + colored_mpl[:, :, c] * 0.6).astype(np.uint8),
                base_img[:, :, c],
            )

    plt.imsave(os.path.join(OUTPUT_DIR, "depth_sparse_overlay.png"), overlay)

    # ====================== Side-by-Side 魚眼連線圖 ======================
    print("產生魚眼連線圖...")
    if os.path.exists(LEFT_FISHEYE_ORIG) and os.path.exists(RIGHT_FISHEYE_ORIG):
        l_img = cv2.imread(LEFT_FISHEYE_ORIG)
        r_img = cv2.imread(RIGHT_FISHEYE_ORIG)

        # Resize to match height
        th = min(l_img.shape[0], r_img.shape[0])
        if l_img.shape[0] != th:
            l_img = cv2.resize(l_img, (int(l_img.shape[1] * th / l_img.shape[0]), th))
        if r_img.shape[0] != th:
            r_img = cv2.resize(r_img, (int(r_img.shape[1] * th / r_img.shape[0]), th))

        vis = np.hstack((l_img, r_img))

        # 畫線 (一樣只畫 valid points)
        # 注意: 這裡的 df_valid 已經只包含深度 <= MAX_DEPTH 的點了
        pts_l = df_valid[["L_fisheye_u", "L_fisheye_v"]].values
        pts_r = df_valid[["R_fisheye_u", "R_fisheye_v"]].values

        for i in range(len(pts_l)):
            p1 = (int(round(pts_l[i][0])), int(round(pts_l[i][1])))
            p2 = (int(round(pts_r[i][0])) + l_img.shape[1], int(round(pts_r[i][1])))

            cv2.circle(vis, p1, 4, (0, 255, 0), -1)
            cv2.circle(vis, p2, 4, (0, 255, 0), -1)
            cv2.line(vis, p1, p2, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "fisheye_matches_side_by_side.png"), vis)
    else:
        print("找不到原始魚眼圖，跳過連線圖繪製。")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
