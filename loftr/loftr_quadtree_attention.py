# loftr_quadtree_accelerated.py
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
OUTPUT_DIR = "loftr_results_quadtree"

# QuadTree 設定
MIN_TILE_SIZE = 400  # 最小切塊尺寸 (避免切太小)
MAX_TILE_SIZE = 1200  # 最大切塊尺寸 (超過就分割)
COMPLEXITY_THRESHOLD = 25  # 複雜度門檻 (越低越容易分割,建議 20-30)
OVERLAP_PX = 100  # 切塊重疊像素
CONF_THRESHOLD = 0.5  # 匹配信心度門檻

# ------------------------------------------------------------


class QuadTreeNode:
    """QuadTree 節點"""

    def __init__(self, x, y, width, height, level=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.level = level
        self.children = []
        self.is_leaf = True
        self.complexity = 0.0

    def __repr__(self):
        return f"Node(x={self.x}, y={self.y}, w={self.width}, h={self.height}, level={self.level}, complex={self.complexity:.1f})"


def calculate_complexity(img_crop):
    """
    計算圖像區域的複雜度
    使用 Laplacian 變異數來估計邊緣/細節豐富程度
    """
    if img_crop.size == 0:
        return 0.0

    # 計算 Laplacian (邊緣檢測)
    laplacian = cv2.Laplacian(img_crop, cv2.CV_64F)
    variance = laplacian.var()

    return variance


def build_quadtree(img_gray, x, y, width, height, level=0, max_level=4):
    """
    遞迴建立 QuadTree
    根據圖像複雜度決定是否繼續分割
    """
    node = QuadTreeNode(x, y, width, height, level)

    # 提取當前區域
    y_end = min(y + height, img_gray.shape[0])
    x_end = min(x + width, img_gray.shape[1])
    crop = img_gray[y:y_end, x:x_end]

    # 計算複雜度
    node.complexity = calculate_complexity(crop)

    # 終止條件
    should_split = (
        width > MIN_TILE_SIZE
        and height > MIN_TILE_SIZE
        and max(width, height) > MAX_TILE_SIZE
        and level < max_level
        and node.complexity > COMPLEXITY_THRESHOLD
    )

    if not should_split:
        node.is_leaf = True
        return node

    # 分割成 4 個子區域
    node.is_leaf = False
    mid_x = width // 2
    mid_y = height // 2

    # 加入重疊以避免邊界問題
    overlap = OVERLAP_PX if level > 0 else 0

    quadrants = [
        (x, y, mid_x + overlap, mid_y + overlap),  # 左上
        (x + mid_x - overlap, y, mid_x + overlap, mid_y + overlap),  # 右上
        (x, y + mid_y - overlap, mid_x + overlap, mid_y + overlap),  # 左下
        (
            x + mid_x - overlap,
            y + mid_y - overlap,
            mid_x + overlap,
            mid_y + overlap,
        ),  # 右下
    ]

    for qx, qy, qw, qh in quadrants:
        # 確保不超出邊界
        qx = max(0, min(qx, img_gray.shape[1] - 1))
        qy = max(0, min(qy, img_gray.shape[0] - 1))
        qw = min(qw, img_gray.shape[1] - qx)
        qh = min(qh, img_gray.shape[0] - qy)

        if qw > 0 and qh > 0:
            child = build_quadtree(img_gray, qx, qy, qw, qh, level + 1, max_level)
            node.children.append(child)

    return node


def get_leaf_nodes(node):
    """獲取所有葉子節點 (實際要處理的切塊)"""
    if node.is_leaf:
        return [node]

    leaves = []
    for child in node.children:
        leaves.extend(get_leaf_nodes(child))
    return leaves


def load_image_gray(fname, device):
    """讀取圖片"""
    img = cv2.imread(fname)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {fname}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(img_gray).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    return img_tensor, img_rgb, img_gray


def run_loftr_on_crop(matcher, crop0, crop1, device):
    """在單一區域執行 LoFTR"""

    _, _, h, w = crop0.shape

    # 確保尺寸能被 8 整除
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        crop0 = torch.nn.functional.pad(crop0, (0, pad_w, 0, pad_h))
        crop1 = torch.nn.functional.pad(crop1, (0, pad_w, 0, pad_h))

    with torch.no_grad():
        output = matcher({"image0": crop0, "image1": crop1})

    mkpts0 = output["keypoints0"].cpu().numpy()
    mkpts1 = output["keypoints1"].cpu().numpy()
    conf = output["confidence"].cpu().numpy()

    return mkpts0, mkpts1, conf


def remove_duplicates(mkpts0, mkpts1, confs, threshold=8.0):
    """去除重複點"""
    if len(mkpts0) == 0:
        return mkpts0, mkpts1, confs

    data = np.hstack((mkpts0, mkpts1, confs.reshape(-1, 1)))
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


def quadtree_loftr_match(
    img0_tensor, img1_tensor, img0_gray, img1_gray, matcher, device
):
    """使用 QuadTree 進行自適應匹配"""

    _, _, H, W = img0_tensor.shape

    print(f"\n{'=' * 60}")
    print("QuadTree 自適應分割匹配")
    print(f"{'=' * 60}")
    print(f"原圖尺寸: {W}x{H}")
    print(f"最小切塊: {MIN_TILE_SIZE}px, 最大切塊: {MAX_TILE_SIZE}px")
    print(f"複雜度門檻: {COMPLEXITY_THRESHOLD}")

    # 建立 QuadTree (基於左圖的複雜度)
    print("\n[1/3] 建立 QuadTree 結構...")
    root = build_quadtree(img0_gray, 0, 0, W, H)
    leaf_nodes = get_leaf_nodes(root)

    print(f"✓ QuadTree 建立完成: {len(leaf_nodes)} 個葉子節點")

    # 統計各層級的節點數
    level_count = {}
    for node in leaf_nodes:
        level_count[node.level] = level_count.get(node.level, 0) + 1

    print("  層級分布:", end=" ")
    for level in sorted(level_count.keys()):
        print(f"L{level}={level_count[level]}", end=" ")
    print()

    # 對每個葉子節點執行 LoFTR
    print("\n[2/3] 執行 LoFTR 匹配...")
    all_mkpts0 = []
    all_mkpts1 = []
    all_confs = []

    for idx, node in enumerate(leaf_nodes, 1):
        x, y, w, h = node.x, node.y, node.width, node.height

        # 裁切區域
        crop0 = img0_tensor[:, :, y : min(y + h, H), x : min(x + w, W)]
        crop1 = img1_tensor[:, :, y : min(y + h, H), x : min(x + w, W)]

        print(
            f"  [{idx:2d}/{len(leaf_nodes)}] Level{node.level} "
            f"[{x:4d},{y:4d}] {w:4d}x{h:4d} "
            f"complex={node.complexity:6.1f}",
            end=" ",
        )

        # 執行匹配
        kpts0, kpts1, conf = run_loftr_on_crop(matcher, crop0, crop1, device)

        # 過濾信心度
        mask = conf > CONF_THRESHOLD
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        conf = conf[mask]

        print(f"→ {len(kpts0):4d} pts")

        if len(kpts0) > 0:
            # 還原全域座標
            kpts0[:, 0] += x
            kpts0[:, 1] += y
            kpts1[:, 0] += x
            kpts1[:, 1] += y

            all_mkpts0.append(kpts0)
            all_mkpts1.append(kpts1)
            all_confs.append(conf)

    if not all_mkpts0:
        print("\n⚠ 警告: 未找到任何匹配點!")
        return np.array([]), np.array([]), np.array([])

    all_mkpts0 = np.vstack(all_mkpts0)
    all_mkpts1 = np.vstack(all_mkpts1)
    all_confs = np.concatenate(all_confs)

    print(f"\n合併後原始點數: {len(all_mkpts0)}")

    # 去除重複
    print("[3/3] 去除重複點...")
    final_kp0, final_kp1, final_conf = remove_duplicates(
        all_mkpts0, all_mkpts1, all_confs
    )
    print(f"✓ 最終點數: {len(final_kp0)}")
    print(f"{'=' * 60}\n")

    return final_kp0, final_kp1, final_conf


def visualize_quadtree(img_rgb, root, output_dir):
    """視覺化 QuadTree 分割結果"""
    plt.figure(figsize=(16, 12))
    plt.imshow(img_rgb)

    leaf_nodes = get_leaf_nodes(root)

    # 定義顏色映射 (按層級)
    colors = ["red", "yellow", "green", "cyan", "magenta"]

    for node in leaf_nodes:
        x, y, w, h = node.x, node.y, node.width, node.height
        color = colors[min(node.level, len(colors) - 1)]

        # 繪製矩形
        rect = plt.Rectangle(
            (x, y), w, h, fill=False, edgecolor=color, linewidth=2, alpha=0.8
        )
        plt.gca().add_patch(rect)

        # 標註複雜度
        plt.text(
            x + 5,
            y + 20,
            f"L{node.level}\n{node.complexity:.0f}",
            color=color,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

    plt.title(f"QuadTree Adaptive Partitioning ({len(leaf_nodes)} tiles)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "quadtree_partition.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("✓ QuadTree 分割視覺化已儲存")


def save_matches_result(img1_rgb, img2_rgb, kp1, kp2, output_dir):
    """儲存匹配結果視覺化"""
    height = max(img1_rgb.shape[0], img2_rgb.shape[0])
    width1 = img1_rgb.shape[1]
    combined = np.zeros((height, width1 + img2_rgb.shape[1], 3), dtype=np.uint8)
    combined[: img1_rgb.shape[0], :width1] = img1_rgb
    combined[: img2_rgb.shape[0], width1:] = img2_rgb

    show_limit = 3000
    indices = np.arange(len(kp1))
    if len(kp1) > show_limit:
        np.random.shuffle(indices)
        indices = indices[:show_limit]
        print(f"註: 繪圖僅顯示 {show_limit} 個點 (CSV 包含全部 {len(kp1)} 個點)")

    plt.figure(figsize=(20, 10))
    plt.imshow(combined)

    kp1_show = kp1[indices]
    kp2_show = kp2[indices]

    for (x1, y1), (x2, y2) in zip(kp1_show, kp2_show):
        plt.plot([x1, x2 + width1], [y1, y2], "c-", linewidth=0.3, alpha=0.5)

    plt.plot(kp1_show[:, 0], kp1_show[:, 1], "r.", markersize=1, alpha=0.8)
    plt.plot(kp2_show[:, 0] + width1, kp2_show[:, 1], "r.", markersize=1, alpha=0.8)

    plt.title(f"QuadTree LoFTR Matches: {len(kp1)} points", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "result_vis.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("✓ 匹配視覺化已儲存")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("QuadTree 加速 LoFTR 匹配系統")
    print(f"{'=' * 60}")
    print(f"裝置: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # 載入模型
    print("\n載入 LoFTR 模型...")
    matcher = LoFTR(pretrained="outdoor").to(device)
    matcher.eval()
    print("✓ 模型載入完成")

    # 讀取圖片
    print("\n讀取圖片...")
    img0_tensor, img0_rgb, img0_gray = load_image_gray(IMG_NAME_LEFT, device)
    img1_tensor, img1_rgb, img1_gray = load_image_gray(IMG_NAME_RIGHT, device)

    print(f"✓ 左圖: {IMG_NAME_LEFT} ({img0_gray.shape[1]}x{img0_gray.shape[0]})")
    print(f"✓ 右圖: {IMG_NAME_RIGHT} ({img1_gray.shape[1]}x{img1_gray.shape[0]})")

    # 使用 QuadTree 進行匹配
    mkpts0, mkpts1, conf = quadtree_loftr_match(
        img0_tensor, img1_tensor, img0_gray, img1_gray, matcher, device
    )

    if len(mkpts0) == 0:
        print("❌ 沒有找到匹配點")
        return

    # 視覺化 QuadTree 分割
    print("\n生成視覺化...")
    root = build_quadtree(img0_gray, 0, 0, img0_gray.shape[1], img0_gray.shape[0])
    visualize_quadtree(img0_rgb, root, OUTPUT_DIR)

    # 儲存匹配結果
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

    print(f"\n{'=' * 60}")
    print("結果統計")
    print(f"{'=' * 60}")
    print(f"✓ CSV: {csv_path}")
    print(f"  總匹配點數: {len(df)}")
    print(f"  平均信心度: {conf.mean():.3f}")
    print(f"  信心度範圍: [{conf.min():.3f}, {conf.max():.3f}]")
    print("\n✅ 全部完成!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
