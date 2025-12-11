import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =================================================================
# 設定區塊
# =================================================================
# 您的資料檔案路徑
file_path = "output_3d/3d.csv"
# 輸出 HTML 檔案名稱
output_html_file = "docs"
os.makedirs(output_html_file, exist_ok=True)
output_html_file = os.path.join(output_html_file, "3d_point_cloud_jet.html")


# 檢查檔案是否存在
if not os.path.exists(file_path):
    print(f"❌ 錯誤: 找不到檔案 {file_path}")
    print("請檢查檔案路徑是否正確，或取消上方『測試數據生成』區塊的註解並執行一次。")
else:
    try:
        # 1. 讀取 CSV
        print(f"正在讀取 CSV 檔案: {file_path}...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        # 檢查欄位
        required_columns = ["X_mm", "Y_mm", "Z_mm", "Depth_m"]
        if not all(col in df.columns for col in required_columns):
            print(f"❌ 錯誤: CSV 缺少必要的欄位。需要的欄位: {required_columns}")
        else:
            print("正在生成 3D 圖表 (Y/Z 軸互換, Jet 顏色刻度)...")

            # --- 2. 建立主要的點雲圖層 (使用 plotly express) ---
            fig = px.scatter_3d(
                df,
                x="X_mm",
                y="Z_mm",  # 視覺上的 Y 軸顯示 Z 數據 (Y/Z 軸互換)
                z="Y_mm",  # 視覺上的 Z 軸顯示 Y 數據 (Y/Z 軸互換)
                color="Depth_m",
                color_continuous_scale="jet",  # 設定為 Jet 顏色刻度
                title="3D Point Cloud Visualization (Y/Z Swapped, Jet Scale)",
                labels={"Depth_m": "Depth (m)"},
                opacity=0.8,
            )
            fig.update_traces(marker=dict(size=2))  # 調整點的大小

            # --- 3. 建立參考點圖層 ((-30, 0, 0) 和 (30, 0, 0)) ---
            # 參考點座標 (資料 X, 資料 Z, 資料 Y)
            ref_points_x = [-30, 30]
            ref_points_visual_y = [0, 0]
            ref_points_visual_z = [0, 0]

            ref_trace = go.Scatter3d(
                x=ref_points_x,
                y=ref_points_visual_y,
                z=ref_points_visual_z,
                mode="markers",
                marker=dict(
                    size=15,  # 大黑點
                    color="black",
                    symbol="circle",
                ),
                name="Reference Points",
            )
            fig.add_trace(ref_trace)

            # --- 4. 優化顯示與標籤 ---
            fig.update_layout(
                scene=dict(
                    xaxis_title="X_mm",
                    yaxis_title="Z_mm (Visual Y)",  # 軸互換後的新標題
                    zaxis_title="Y_mm (Visual Z)",  # 軸互換後的新標題
                ),
                legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
            )

            # --- 5. 儲存為 HTML 檔案 ---
            fig.write_html(output_html_file)
            print(f"✅ 圖表已成功儲存為 HTML 檔案: {output_html_file}")
            print(
                "\n💡 下一步：請將此 HTML 檔案上傳至您的網路空間（如 GitHub Pages, Dropbox 或網頁伺服器），即可獲得公開分享網址。"
            )

    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")
