import os
import json
import math
import numpy as np
from collections import defaultdict

# ================= 只改这一行 =================
TRAIN_JSON = './data/trainjs/annotations.json'
# ==============================================

# 与 Dataset 保持一致：过滤 w<20 或 h<20
MIN_W, MIN_H = 18.0, 18.0

# 面积分层（与 CustomEvaluator 一致）
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2


def load_boxes(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 与 Dataset 一致：只取前 3 个类别
    cat_ids = sorted([c["id"] for c in coco["categories"]])[:3]
    cat_set = set(cat_ids)

    boxes = []
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat_set:
            continue
        x, y, w, h = ann["bbox"]
        if w < MIN_W or h < MIN_H:
            continue
        boxes.append((float(w), float(h)))

    return np.array(boxes, dtype=np.float32)


def area_type(area):
    if area < SMALL_MAX:
        return "small"
    elif area < MEDIUM_MAX:
        return "medium"
    else:
        return "large"


def print_distribution(boxes):
    print(" " + "=" * 60)
    print("📊 GT 框整体分布")
    print("=" * 60)

    ws, hs = boxes[:, 0], boxes[:, 1]
    areas = ws * hs
    ratios = ws / hs

    print(f"有效 GT 总数: {len(boxes)}")
    print(f"宽 w: min={ws.min():.1f}, max={ws.max():.1f}, mean={ws.mean():.1f}, median={np.median(ws):.1f}")
    print(f"高 h: min={hs.min():.1f}, max={hs.max():.1f}, mean={hs.mean():.1f}, median={np.median(hs):.1f}")
    print(f"面积: min={areas.min():.0f}, max={areas.max():.0f}, mean={areas.mean():.0f}, median={np.median(areas):.0f}")
    print(f"宽高比 w/h: min={ratios.min():.2f}, max={ratios.max():.2f}, mean={ratios.mean():.2f}, median={np.median(ratios):.2f}")

    print(" 【S / M / L 分层占比】")
    counts = defaultdict(int)
    for a in areas:
        counts[area_type(a)] += 1
    for k in ["small", "medium", "large"]:
        print(f" {k.upper():6s}: {counts[k]:6d} 个 ({counts[k] / len(boxes) * 100:5.1f}%)")

    print(" 【按分层统计平均尺寸】")
    for k in ["small", "medium", "large"]:
        mask = np.array([area_type(a) == k for a in areas])
        if mask.sum() > 0:
            print(f" {k.upper():6s}: 平均 w×h ≈ {ws[mask].mean():.1f} × {hs[mask].mean():.1f} (n={mask.sum()})")

    print("=" * 60 + " ")


def iou_1_to_n(box, boxes):
    """单个 (w,h) 与一组 (N,2) 的 IoU，中心对齐"""
    bw, bh = box
    ws, hs = boxes[:, 0], boxes[:, 1]
    inter_w = np.minimum(bw, ws)
    inter_h = np.minimum(bh, hs)
    inter = inter_w * inter_h
    union = bw * bh + ws * hs - inter
    return inter / np.maximum(union, 1e-7)


def kmeans_iou(boxes, k=5, iters=100, seed=42):
    """基于 IoU 距离的 K-means，返回 k 个 (w,h) 中心，按面积升序"""
    rng = np.random.RandomState(seed)
    n = len(boxes)

    # K-means++ 初始化（修复版）
    centers = [boxes[rng.randint(n)]]
    for _ in range(1, k):
        # 计算每个 GT 与当前所有中心的 IoU，形状 (N, len(centers))
        ious = np.stack([iou_1_to_n(c, boxes) for c in centers], axis=1)
        # 每个 GT 取最大 IoU，距离 = 1 - IoU
        d = 1.0 - np.max(ious, axis=1)
        d = np.maximum(d, 1e-6)
        probs = d / d.sum()
        centers.append(boxes[rng.choice(n, p=probs)])
    centers = np.array(centers, dtype=np.float32)

    for _ in range(iters):
        # 每个 GT 分配到 IoU 最大的中心
        ious = np.stack([iou_1_to_n(c, boxes) for c in centers], axis=1) # (N, k)
        assign = np.argmax(ious, axis=1)

        new_centers = []
        for j in range(k):
            mask = assign == j
            if mask.sum() == 0:
                new_centers.append(centers[j])
            else:
                # 用该簇的 median w/h 作为新中心，比 mean 更抗离群值
                new_centers.append(np.array([
                    np.median(boxes[mask, 0]),
                    np.median(boxes[mask, 1]),
                ], dtype=np.float32))
        new_centers = np.array(new_centers, dtype=np.float32)

        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers

    # 按面积升序，对应 P2→P6
    order = np.argsort(centers[:, 0] * centers[:, 1])
    centers = centers[order]

    # 输出每个中心的覆盖数量
    ious = np.stack([iou_1_to_n(c, boxes) for c in centers], axis=1)
    assign = np.argmax(ious, axis=1)
    cluster_counts = [int((assign == j).sum()) for j in range(k)]

    return centers, cluster_counts


def mean_iou(boxes, anchors):
    """每个 GT 与所有 anchor 的最大 IoU 的平均值"""
    ious = np.stack([iou_1_to_n(a, boxes) for a in anchors], axis=1)
    return float(np.max(ious, axis=1).mean())


def print_anchor_analysis(boxes, current_anchors):
    print(" " + "=" * 60)
    print("🎯 Anchor 建议（外层固定 5 个，对应 FPN P2~P6）")
    print("=" * 60)

    # 当前 anchor 的匹配质量
    cur = np.array([(s, s) for s in current_anchors], dtype=np.float32)
    cur_miou = mean_iou(boxes, cur)
    print(f" 当前 Anchor {tuple(current_anchors)}")
    print(f" GT 平均最大 IoU = {cur_miou:.4f}")

    # K-means 聚类
    centers, counts = kmeans_iou(boxes, k=5)
    km_miou = mean_iou(boxes, centers)
    print(f" K-means (k=5) 聚类结果，GT 平均最大 IoU = {km_miou:.4f}")
    print(f" 相对当前提升: {km_miou - cur_miou:+.4f}")

    print(" 聚类中心（按面积升序 → P2/P3/P4/P5/P6）:")
    for i, (c, n) in enumerate(zip(centers, counts)):
        level = ["P2", "P3", "P4", "P5", "P6"][i]
        side = math.sqrt(c[0] * c[1])
        print(f" {level}: w={c[0]:6.1f}, h={c[1]:6.1f}, 等效边长≈{side:5.1f}, 覆盖 GT 数={n}")

    # 直接可粘贴的 sizes 元组（每层取等效边长四舍五入，单元素）
    suggested = tuple((int(round(math.sqrt(c[0] * c[1]))),) for c in centers)
    print(" 可直接替换 model.rpn.anchor_generator.sizes 的值（外层 5 个，勿增勿减）:")
    print(f" {suggested} ")

    # 同时给出宽高比建议（如果宽高比偏离 1 较远）
    ratios = boxes[:, 0] / boxes[:, 1]
    p10, p50, p90 = np.percentile(ratios, [10, 50, 90])
    print(f"GT 宽高比 w/h 分位数: P10={p10:.2f}, P50={p50:.2f}, P90={p90:.2f}")
    if abs(p50 - 1.0) > 0.25 or p10 < 0.5 or p90 > 2.0:
        print(" ⚠️ 宽高比分布偏离 1 较远，正方形 Anchor 可能不是最优。")
        print(" 但 aspect_ratios 是每层共享的，改它需要同步确认 RPN head 的 anchor 数量，")
        print(" 建议先只改 sizes，宽高比作为下一步单独实验。")
    else:
        print(" 宽高比集中在 1 附近，正方形 Anchor 与数据形状基本匹配。")

    print("=" * 60 + " ")

    # 额外：k=3 / k=7 作为参考，但不建议直接用于 5 层 FPN
    print("【参考：不同 k 的聚类平均 IoU（仅供探索，k≠5 不能直接用于 5 层 FPN）】")
    for k in [3, 5, 7]:
        c, _ = kmeans_iou(boxes, k=k)
        mi = mean_iou(boxes, c)
        print(f" k={k}: 平均最大 IoU = {mi:.4f}, 中心 = {[(round(math.sqrt(x*y)),) for x, y in c]}")
    print()


if __name__ == "__main__":
    boxes = load_boxes(TRAIN_JSON)
    if len(boxes) == 0:
        print("❌ 没有读到有效 GT，检查 JSON 路径和过滤条件。")
        exit(1)

    print_distribution(boxes)

    # R50 / B4-NS 特调实际使用的 Anchor（外层 5 个，P2→P6）
    current = (32, 56, 112, 256, 512)
    print_anchor_analysis(boxes, current)
