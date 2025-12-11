# loftr_to_anypoint_csv.py
import os

import cv2
import kornia
import kornia.geometry.transform as KGT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # ← 新增
import torch
from kornia.feature import LoFTR

# -------------------------- 設定參數 --------------------------
IMG_NAME_LEFT = "frame_left_anypoint.png"
IMG_NAME_RIGHT = "frame_right_anypoint.png"
OUTPUT_DIR = "loftr_results"
MAX_IMAGE_SIZE = 1600
CONF_THRESHOLD = 0.6  # 信心度門檻


# ------------------------------------------------------------
def load_torch_image(fname, device, max_size=MAX_IMAGE_SIZE):
    img = cv2.imread(fname)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {fname}")

    H_orig, W_orig = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_tensor = kornia.image_to_tensor(img_gray, False).float() / 255.0

    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img_tensor = img_tensor.to(device)

    scale_factor = 1.0
    if max(H_orig, W_orig) > max_size:
        scale = max_size / max(H_orig, W_orig)
        new_H = int(H_orig * scale)
        new_W = int(W_orig * scale)
        scale_factor = 1.0 / scale  # 因為我們縮小了，所以還原時要乘回去
        print(f"縮放圖片 {fname}: {H_orig}x{W_orig} → {new_H}x{new_W}")

        img_tensor = KGT.resize(
            img_tensor.unsqueeze(0), (new_H, new_W), interpolation="bilinear"
        ).squeeze(0)

    return (
        img_tensor.unsqueeze(0),
        img_rgb,
        scale_factor,
    )  # scale_factor = 原圖 / 縮小後


def save_matches_result(img1, img2, kp1, kp2, output_dir):
    height = max(img1.shape[0], img2.shape[0])
    width1 = img1.shape[1]
    combined = np.zeros((height, width1 + img2.shape[1], 3), dtype=np.uint8)
    combined[: img1.shape[0], :width1] = img1
    combined[: img2.shape[0], width1:] = img2

    plt.figure(figsize=(20, 10))
    plt.imshow(combined)
    for (x1, y1), (x2, y2) in zip(kp1, kp2):
        plt.plot([x1, x2 + width1], [y1, y2], "c-", linewidth=0.5, alpha=0.7)
        plt.plot(x1, y1, "yo", markersize=3)
        plt.plot(x2 + width1, y2, "yo", markersize=3)
    plt.title(f"LoFTR Matches: {len(kp1)} points (conf > {CONF_THRESHOLD})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "result_vis.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"匹配圖已儲存: {os.path.join(output_dir, 'result_vis.png')}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 載入模型
    print("載入 LoFTR 模型 (outdoor)...")
    matcher = LoFTR(pretrained="outdoor").to(device)
    matcher.eval()

    # 讀圖
    img0_tensor, img0_rgb, scale0 = load_torch_image(IMG_NAME_LEFT, device)
    img1_tensor, img1_rgb, scale1 = load_torch_image(IMG_NAME_RIGHT, device)

    # 推理
    with torch.no_grad():
        output = matcher({"image0": img0_tensor, "image1": img1_tensor})

    # 提取匹配點（還在縮小圖上）
    mkpts0 = output["keypoints0"].cpu().numpy().reshape(-1, 2)
    mkpts1 = output["keypoints1"].cpu().numpy().reshape(-1, 2)
    confidence = output["confidence"].cpu().numpy()

    # 過濾信心度
    good = confidence > CONF_THRESHOLD
    mkpts0 = mkpts0[good]
    mkpts1 = mkpts1[good]
    print(f"過濾後匹配點數: {len(mkpts0)} (信心度 > {CONF_THRESHOLD})")

    # 還原到原始 anypoint 影像尺寸
    mkpts0_orig = mkpts0 * scale0
    mkpts1_orig = mkpts1 * scale1

    # 儲存視覺化
    save_matches_result(img0_rgb, img1_rgb, mkpts0_orig, mkpts1_orig, OUTPUT_DIR)

    # ====================== 輸出 anypoint.csv ======================
    df = pd.DataFrame(
        {
            "L_anypoint_x": mkpts0_orig[:, 0],
            "L_anypoint_y": mkpts0_orig[:, 1],
            "R_anypoint_x": mkpts1_orig[:, 0],
            "R_anypoint_y": mkpts1_orig[:, 1],
        }
    )

    csv_path = os.path.join(OUTPUT_DIR, "anypoint.csv")
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"成功產生 anypoint.csv → {csv_path}")
    print(f"   總點數: {len(df)}")
    print("   可直接用你之前的 3D 計算程式處理！")
    print("全部完成！")


if __name__ == "__main__":
    main()
