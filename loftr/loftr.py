# loftr_tiled_2x2.py
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kornia.feature import LoFTR

# -------------------------- 設定參數 --------------------------
IMG_NAME_LEFT = "frame_left_anypoint.png"
IMG_NAME_RIGHT = "frame_right_anypoint.png"
OUTPUT_DIR = "loftr_results"

# 切塊設定 - 2x2 網格
GRID_ROWS = 2  # 垂直切 2 塊
GRID_COLS = 2  # 水平切 2 塊
OVERLAP_PX = 150  # 切塊間的重疊像素 (增加重疊以避免邊界遺失)
CONF_THRESHOLD = 0.5  # 信心度門檻

# GPU 記憶體管理
MAX_CROP_SIZE = 1600  # 單塊最大尺寸

# ------------------------------------------------------------


def load_image_gray(fname, device):
    """讀取圖片並轉為 Tensor (保留原圖尺寸)"""
    img = cv2.imread(fname)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {fname}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 轉 Tensor: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_gray).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    return img_tensor, img_rgb, img_gray.shape


def run_loftr_on_crop(matcher, crop0, crop1, device, max_crop_size):
    """在單一裁切區域上執行 LoFTR"""

    _, _, h, w = crop0.shape
    scale = 1.0

    # 檢查是否需要縮小
    if max(h, w) > max_crop_size:
        scale = max_crop_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        crop0 = torch.nn.functional.interpolate(
            crop0, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        crop1 = torch.nn.functional.interpolate(
            crop1, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
    else:
        # 確保尺寸能被 8 整除
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            crop0 = torch.nn.functional.pad(crop0, (0, pad_w, 0, pad_h))
            crop1 = torch.nn.functional.pad(crop1, (0, pad_w, 0, pad_h))

    # LoFTR 推理
    with torch.no_grad():
        output = matcher({"image0": crop0, "image1": crop1})

    mkpts0 = output["keypoints0"].cpu().numpy()
    mkpts1 = output["keypoints1"].cpu().numpy()
    conf = output["confidence"].cpu().numpy()

    # 還原縮放
    if scale != 1.0:
        mkpts0 = mkpts0 / scale
        mkpts1 = mkpts1 / scale

    return mkpts0, mkpts1, conf


def remove_duplicates(mkpts0, mkpts1, confs, threshold=5.0):
    """去除重疊區域產生的重複點"""
    if len(mkpts0) == 0:
        return mkpts0, mkpts1, confs

    data = np.hstack((mkpts0, mkpts1, confs.reshape(-1, 1)))

    # 依信心度降序排列
    data = data[data[:, 4].argsort()[::-1]]

    coords0 = data[:, :2]
    keep_mask = np.ones(len(data), dtype=bool)

    for i in range(len(data)):
        if not keep_mask[i]:
            continue

        curr_pt = coords0[i]
        dists = np.linalg.norm(coords0[i + 1 :] - curr_pt, axis=1)
        close_indices = np.where(dists < threshold)[0] + (i + 1)
        keep_mask[close_indices] = False

    data_clean = data[keep_mask]

    return data_clean[:, :2], data_clean[:, 2:4], data_clean[:, 4]


def tiled_loftr_match(img0_tensor, img1_tensor, matcher, device, rows, cols, overlap):
    """2x2 切塊執行 LoFTR 匹配"""
    _, _, H, W = img0_tensor.shape

    # 計算每個區塊的基本大小
    h_block = H // rows
    w_block = W // cols

    all_mkpts0 = []
    all_mkpts1 = []
    all_confs = []

    print(f"\n========== 開始 {rows}x{cols} 切塊匹配 ==========")
    print(f"原圖尺寸: {W}x{H}")
    print(f"重疊像素: {overlap}px")
    print(f"信心度門檻: {CONF_THRESHOLD}\n")

    tile_count = 0

    for r in range(rows):
        for c in range(cols):
            tile_count += 1

            # 計算切塊範圍 (加入重疊)
            y_start = max(0, r * h_block - (overlap if r > 0 else 0))
            y_end = min(H, (r + 1) * h_block + (overlap if r < rows - 1 else 0))

            x_start = max(0, c * w_block - (overlap if c > 0 else 0))
            x_end = min(W, (c + 1) * w_block + (overlap if c < cols - 1 else 0))

            y_start, y_end, x_start, x_end = (
                int(y_start),
                int(y_end),
                int(x_start),
                int(x_end),
            )

            crop0 = img0_tensor[:, :, y_start:y_end, x_start:x_end]
            crop1 = img1_tensor[:, :, y_start:y_end, x_start:x_end]

            crop_h = y_end - y_start
            crop_w = x_end - x_start

            print(
                f"[Tile {tile_count}/4] 位置({r},{c}): x[{x_start}:{x_end}] y[{y_start}:{y_end}] 尺寸:{crop_w}x{crop_h}",
                end=" ",
            )

            # 執行 LoFTR
            kpts0_local, kpts1_local, conf_local = run_loftr_on_crop(
                matcher, crop0, crop1, device, MAX_CROP_SIZE
            )

            # 過濾信心度
            mask = conf_local > CONF_THRESHOLD
            kpts0_local = kpts0_local[mask]
            kpts1_local = kpts1_local[mask]
            conf_local = conf_local[mask]

            print(f"→ 找到 {len(kpts0_local)} 個匹配點")

            if len(kpts0_local) > 0:
                # 還原為全域座標
                kpts0_local[:, 0] += x_start
                kpts0_local[:, 1] += y_start
                kpts1_local[:, 0] += x_start
                kpts1_local[:, 1] += y_start

                all_mkpts0.append(kpts0_local)
                all_mkpts1.append(kpts1_local)
                all_confs.append(conf_local)

    if not all_mkpts0:
        print("\n警告: 未找到任何匹配點!")
        return np.array([]), np.array([]), np.array([])

    all_mkpts0 = np.vstack(all_mkpts0)
    all_mkpts1 = np.vstack(all_mkpts1)
    all_confs = np.concatenate(all_confs)

    print(f"\n切塊合併後原始點數: {len(all_mkpts0)}")

    # 去除重複點
    final_kp0, final_kp1, final_conf = remove_duplicates(
        all_mkpts0, all_mkpts1, all_confs, threshold=8.0
    )
    print(f"去除重複後最終點數: {len(final_kp0)}")
    print("=" * 50 + "\n")

    return final_kp0, final_kp1, final_conf


def save_matches_result(img1_rgb, img2_rgb, kp1, kp2, output_dir):
    """儲存視覺化結果"""
    height = max(img1_rgb.shape[0], img2_rgb.shape[0])
    width1 = img1_rgb.shape[1]
    combined = np.zeros((height, width1 + img2_rgb.shape[1], 3), dtype=np.uint8)
    combined[: img1_rgb.shape[0], :width1] = img1_rgb
    combined[: img2_rgb.shape[0], width1:] = img2_rgb

    # 限制顯示點數以提升效能
    show_limit = 3000
    indices = np.arange(len(kp1))
    if len(kp1) > show_limit:
        np.random.shuffle(indices)
        indices = indices[:show_limit]
        print(
            f"註: 為了繪圖效能,僅顯示隨機 {show_limit} 個點 (CSV 包含全部 {len(kp1)} 個點)"
        )

    plt.figure(figsize=(20, 10))
    plt.imshow(combined)

    kp1_show = kp1[indices]
    kp2_show = kp2[indices]

    for (x1, y1), (x2, y2) in zip(kp1_show, kp2_show):
        plt.plot([x1, x2 + width1], [y1, y2], "c-", linewidth=0.3, alpha=0.5)

    plt.plot(kp1_show[:, 0], kp1_show[:, 1], "r.", markersize=1, alpha=0.8)
    plt.plot(kp2_show[:, 0] + width1, kp2_show[:, 1], "r.", markersize=1, alpha=0.8)

    plt.title(f"2x2 Tiled LoFTR Matches: Total {len(kp1)} points", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "result_vis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 匹配視覺化已儲存: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用裝置: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"
        )

    # 載入模型
    print("載入 LoFTR 模型 (outdoor pretrained)...")
    matcher = LoFTR(pretrained="outdoor").to(device)
    matcher.eval()
    print("✓ 模型載入完成\n")

    # 讀取圖片
    print("讀取圖片...")
    img0_tensor, img0_rgb, shape0 = load_image_gray(IMG_NAME_LEFT, device)
    img1_tensor, img1_rgb, shape1 = load_image_gray(IMG_NAME_RIGHT, device)

    print(f"✓ 左圖: {IMG_NAME_LEFT} ({shape0[1]}x{shape0[0]})")
    print(f"✓ 右圖: {IMG_NAME_RIGHT} ({shape1[1]}x{shape1[0]})")

    # 執行 2x2 切塊匹配
    mkpts0, mkpts1, conf = tiled_loftr_match(
        img0_tensor,
        img1_tensor,
        matcher,
        device,
        rows=GRID_ROWS,
        cols=GRID_COLS,
        overlap=OVERLAP_PX,
    )

    if len(mkpts0) == 0:
        print("錯誤: 沒有找到任何匹配點,請檢查圖片或調整參數")
        return

    # 儲存視覺化
    save_matches_result(img0_rgb, img1_rgb, mkpts0, mkpts1, OUTPUT_DIR)

    # 輸出 CSV
    df = pd.DataFrame(
        {
            "L_anypoint_x": mkpts0[:, 0],
            "L_anypoint_y": mkpts0[:, 1],
            "R_anypoint_x": mkpts1[:, 0],
            "R_anypoint_y": mkpts1[:, 1],
            "Confidence": conf,
        }
    )

    csv_path = os.path.join(OUTPUT_DIR, "anypoint.csv")
    df.to_csv(csv_path, index=False, float_format="%.3f")

    print(f"✓ CSV 檔案已儲存: {csv_path}")
    print(f"  - 總匹配點數: {len(df)}")
    print(f"  - 平均信心度: {conf.mean():.3f}")
    print(f"  - 最小信心度: {conf.min():.3f}")
    print(f"  - 最大信心度: {conf.max():.3f}")
    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
