# loftr_sgbm_fixed.py
import os

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

# ==================== 設定區 ====================
IMG_LEFT = "frame_left.png"
IMG_RIGHT = "frame_right.png"
OUTPUT_DIR = "loftr_results"
MAX_SIZE = 960  # LoFTR 建議最大邊長
CONF_THRESH = 0.6
NUM_DISP = 192  # 必須是 16 的倍數，越大遠處細節越多
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")
print(f"OpenCV 版本: {cv2.__version__}")  # 新增：檢查版本


# ------------------- 讀圖 & LoFTR -------------------
def load_image(fname):
    img = cv2.imread(fname)
    if img is None:
        raise FileNotFoundError(f"找不到 {fname}！請確認檔案存在。")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 縮放到 LoFTR 建議大小
    h, w = img.shape[:2]
    scale = MAX_SIZE / max(h, w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_factor = 1.0 / scale
    else:
        scale_factor = 1.0

    tensor = torch.from_numpy(gray).float()[None, None] / 255.0
    tensor = tensor.to(device)
    return tensor, rgb, gray, scale_factor, (h, w)


# 讀圖
img0_tensor, img0_rgb, img0_gray, sf0, orig_size0 = load_image(IMG_LEFT)
img1_tensor, img1_rgb, img1_gray, sf1, orig_size1 = load_image(IMG_RIGHT)

# LoFTR
matcher = LoFTR(pretrained="outdoor").to(device)
matcher.eval()
with torch.no_grad():
    out = matcher({"image0": img0_tensor, "image1": img1_tensor})

kp0 = out["keypoints0"].cpu().numpy()
kp1 = out["keypoints1"].cpu().numpy()
conf = out["confidence"].cpu().numpy()
mask = conf > CONF_THRESH
kp0, kp1 = kp0[mask], kp1[mask]

# 還原到原圖座標
kp0_orig = kp0 * np.array([sf0, sf0])
kp1_orig = kp1 * np.array([sf1, sf1])

print(f"LoFTR 最終匹配點數: {len(kp0_orig)}")

# ------------------- 高品質 SGBM + WLS 濾波（修正版） -------------------
# 使用原始解析度灰階圖（不再縮小）
orig_left_gray = cv2.imread(IMG_LEFT, 0)
orig_right_gray = cv2.imread(IMG_RIGHT, 0)

# 建立左右 matcher
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISP,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM,
    uniquenessRatio=10,
    speckleWindowSize=200,
    speckleRange=2,
    disp12MaxDiff=1,
    preFilterCap=63,
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

print("正在計算視差圖（可能需要 5~30 秒）...")
dispL = left_matcher.compute(orig_left_gray, orig_right_gray).astype(np.float32) / 16.0
dispR = right_matcher.compute(orig_right_gray, orig_left_gray).astype(np.float32) / 16.0

# WLS 濾波（修正：使用命名參數 + 版本檢查）
try:
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher
    )  # 關鍵修正！
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.3)
    disparity = wls_filter.filter(dispL, orig_left_gray, None, dispR)
    print("WLS 濾波成功啟用（噪點會更少）")
except Exception as e:
    print(f"WLS 濾波失敗（{e}），使用原始 SGBM 視差圖")
    disparity = dispL  # 備用：直接用 left disparity

disparity[disparity <= 0] = np.nan


# ------------------- 儲存所有結果 -------------------
def save_disparity(disp, path):
    valid = ~np.isnan(disp)
    if not valid.any():
        print("警告：視差圖全為無效，請檢查輸入圖像是否為立體對。")
        return
    dmin, dmax = disp[valid].min(), disp[valid].max()
    disp8 = np.clip((disp - dmin) / (dmax - dmin) * 255, 0, 255).astype(np.uint8)
    disp8[~valid] = 0

    # 16bit 原始視差
    disp16 = (disp * 16).astype(np.uint16)
    disp16[~valid] = 0
    cv2.imwrite(path + "_16bit.png", disp16)

    # 彩色圖（用 INFERNO 更適合深度視覺化）
    color = cv2.applyColorMap(disp8, cv2.COLORMAP_INFERNO)
    color[~valid] = 0
    cv2.imwrite(path + "_color.png", color)


# 1. 深度圖
save_disparity(disparity, os.path.join(OUTPUT_DIR, "depth"))


# 2. 匹配圖
def draw_matches():
    h1, w1 = img0_rgb.shape[:2]
    h2, w2 = img1_rgb.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img0_rgb
    canvas[:h2, w1:] = img1_rgb
    for p1, p2 in zip(kp0_orig, kp1_orig):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, tuple(p1.astype(int)), 4, color, -1)
        cv2.circle(canvas, (int(p2[0] + w1), int(p2[1])), 4, color, -1)
        cv2.line(canvas, tuple(p1.astype(int)), (int(p2[0] + w1), int(p2[1])), color, 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "loftr_matches.jpg"), canvas)


draw_matches()

# 3. 極線圖（可視化用）
F, mask = cv2.findFundamentalMat(
    kp0_orig.astype(np.float32), kp1_orig.astype(np.float32), cv2.FM_RANSAC, 1.0, 0.999
)
print(f"Fundamental Matrix RANSAC inliers: {mask.sum() if mask is not None else 0}")

if F is not None and mask is not None:
    lines1 = cv2.computeCorrespondEpilines(
        kp1_orig[mask.ravel() == 1].reshape(-1, 1, 2), 2, F
    )
    lines1 = lines1.reshape(-1, 3)
    epi_img = cv2.cvtColor(orig_left_gray, cv2.COLOR_GRAY2BGR)
    for line, pt in zip(lines1, kp0_orig[mask.ravel() == 1]):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(
            int, [epi_img.shape[1], -(line[2] + line[0] * epi_img.shape[1]) / line[1]]
        )
        cv2.line(epi_img, (x0, y0), (x1, y1), color, 1)
        cv2.circle(epi_img, tuple(pt.astype(int)), 5, color, -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "left_with_epilines.jpg"), epi_img)


# 4. 可選：匯出點雲（ply 格式，用 MeshLab / CloudCompare 打開）
def save_pointcloud():
    focal = orig_left_gray.shape[1]  # 假設焦距 = 圖寬
    baseline = 50.0  # 單位：毫米（請自行修改為你的相機基線）
    Q = np.float32(
        [
            [1, 0, 0, -orig_left_gray.shape[1] / 2],
            [0, -1, 0, orig_left_gray.shape[0] / 2],
            [0, 0, 0, -focal],
            [0, 0, 1 / baseline, 0],
        ]
    )
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(cv2.imread(IMG_LEFT), cv2.COLOR_BGR2RGB)

    mask_valid = ~np.isnan(disparity)
    points = points_3D[mask_valid]
    colors = colors[mask_valid]

    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(os.path.join(OUTPUT_DIR, "pointcloud.ply"), "w") as f:
        f.write(header)
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")
    print("點雲已儲存: pointcloud.ply（記得調整 baseline 值以匹配真實深度）")


save_pointcloud()

print("\n所有檔案已輸出至資料夾:", OUTPUT_DIR)
print("   ├─ loftr_matches.jpg          → LoFTR 匹配視覺化")
print("   ├─ depth_16bit.png            → 原始 16bit 視差（用於後續處理）")
print("   ├─ depth_color.png            → 彩色深度圖（INFERNO 色階，最適合觀看！）")
print("   ├─ left_with_epilines.jpg     → 左圖 + 極線（檢查匹配品質）")
print("   └─ pointcloud.ply             → 3D 點雲（用 MeshLab 開啟）")

# ==================== 輸出匹配點（100% 修正版）===================
print("\n正在儲存所有匹配點座標...")

# 1. TXT 檔案（人類可讀）
with open(os.path.join(OUTPUT_DIR, "loftr_matches.txt"), "w") as f:
    f.write("# LoFTR 高信心匹配點 (原始圖像解析度)\n")
    f.write("# 格式: x_left    y_left    x_right    y_right    confidence\n")
    f.write(f"# 總點數: {len(kp0_orig)}\n")
    for pt1, pt2, c in zip(kp0_orig, kp1_orig, conf[mask]):
        x1, y1 = pt1[0].item(), pt1[1].item()
        x2, y2 = pt2[0].item(), pt2[1].item()
        f.write(f"{x1:9.2f} {y1:9.2f} {x2:9.2f} {y2:9.2f} {c.item():.6f}\n")
print("→ loftr_matches.txt   (已儲存，可用記事本開啟)")

# 2. CSV 檔案（Excel 可用）
matches_array = np.hstack([kp0_orig, kp1_orig, conf[mask].reshape(-1, 1)])
np.savetxt(
    os.path.join(OUTPUT_DIR, "loftr_matches.csv"),
    matches_array,
    delimiter=",",
    header="left_x,left_y,right_x,right_y,confidence",
    comments="",
    fmt="%.4f",
)
print("→ loftr_matches.csv   (Excel 直接開)")

# 3. NPY 檔案（Python 最快讀取）
np.save(
    os.path.join(OUTPUT_DIR, "loftr_matches.npy"),
    {
        "left_keypoints": kp0_orig,
        "right_keypoints": kp1_orig,
        "confidence": conf[mask],
        "num_points": len(kp0_orig),
    },
)
print("→ loftr_matches.npy   (Python 用 np.load 讀取)")
