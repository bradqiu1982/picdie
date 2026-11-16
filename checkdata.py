import json
import numpy as np
from collections import Counter

def analyze_dataset(json_path, dataset_name="Train", batch_size=16):
    print(f" {'='*60}")
    print(f"📊 深度体检: {dataset_name} (设定 Batch Size: {batch_size})")
    print(f"{'='*60}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    num_images = len(coco['images'])
    num_anns = len(coco['annotations'])
    
    # 1. 区分有缺陷(正样本)图 与 无缺陷(纯负样本)图
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_to_anns.setdefault(img_id, []).append(ann)

    defect_images_count = len(img_to_anns)
    normal_images_count = num_images - defect_images_count
    
    print(f"🖼️ 图像总数: {num_images} 张")
    print(f"🔴 有缺陷图 (正样本): {defect_images_count} 张 ({(defect_images_count/num_images*100):.1f}%)")
    print(f"🟢 正常图 (负样本): {normal_images_count} 张 ({(normal_images_count/num_images*100):.1f}%)")
    print(f"📦 标注框总数: {num_anns} 个")
    
    # 2. 尺寸分布与平均值 (只统计有缺陷的图)
    size_counts = {'Small (<32x32)': 0, 'Medium (32~96)': 0, 'Large (>96)': 0}
    size_dims = {
        'Small (<32x32)': {'w': [], 'h': []},
        'Medium (32~96)': {'w': [], 'h': []},
        'Large (>96)': {'w': [], 'h': []}
    }
    filtered_small = 0
    valid_boxes = 0
    aspect_ratios = []
    
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        area = w * h
        
        if w < 20 or h < 20:
            filtered_small += 1
            continue
            
        valid_boxes += 1
        if w > 0 and h > 0:
            aspect_ratios.append(max(w, h) / min(w, h))
        
        if area < 32**2:
            size_key = 'Small (<32x32)'
        elif area < 96**2:
            size_key = 'Medium (32~96)'
        else:
            size_key = 'Large (>96)'
            
        size_counts[size_key] += 1
        size_dims[size_key]['w'].append(w)
        size_dims[size_key]['h'].append(h)

    print(f" 📏 【1. 目标尺寸分布】(剔除 w<20 或 h<20 的极小框 {filtered_small} 个)")
    for size, count in size_counts.items():
        if valid_boxes > 0:
            pct = (count / valid_boxes) * 100
            print(f" - {size:15}: {count:6} 个 ({pct:5.1f}%)", end="")
            if count > 0:
                avg_w = np.mean(size_dims[size]['w'])
                avg_h = np.mean(size_dims[size]['h'])
                print(f" | 平均尺寸: 宽 {avg_w:.1f} x 高 {avg_h:.1f}")
            else:
                print()

    # 3. 核心：单图目标数与 Batch 预估
    print(f" 🖼️ 【2. 单图目标数与 Batch 采样预估】")
    counts_per_defect_img = [len(v) for v in img_to_anns.values()]
    
    if counts_per_defect_img:
        avg_boxes_defect = np.mean(counts_per_defect_img)
        max_boxes = max(counts_per_defect_img)
        
        # 全局平均（包含纯背景图）
        global_avg_boxes = num_anns / num_images
        
        print(f" - 【有缺陷图中】平均目标数: {avg_boxes_defect:.2f} 个/张")
        print(f" - 【有缺陷图中】最多目标数: {max_boxes} 个/张")
        print(f" - 【全局（含背景）】平均目标数: {global_avg_boxes:.2f} 个/张")
        
        # 预测一个 Batch 里的情况
        expected_defect_in_batch = batch_size * (defect_images_count / num_images)
        expected_boxes_in_batch = expected_defect_in_batch * avg_boxes_defect
        
        print(f" ⚠️ 【危险预警：当前 Batch={batch_size} 时的期望值】")
        print(f" - 每个 Batch 平均有 {expected_defect_in_batch:.1f} 张有缺陷图")
        print(f" - 每个 Batch 平均只能产出 {expected_boxes_in_batch:.1f} 个真实正样本 🔥")
        print(f" 💡 (如果这个数字小于 20，RPN 采样 512/1024 时全都会被海量背景淹没！)")

    # 4. 长宽比
    if aspect_ratios:
        ar = np.array(aspect_ratios)
        print(f" 📐 【3. 目标长宽比 (Max/Min)】")
        print(f" - 极端长条形 (比例>5): {np.sum(ar > 5)} 个，占比 {np.sum(ar > 5)/len(ar)*100:.1f}%")

    print(f"{'='*60} ")

if __name__ == '__main__':
    #train_json = './data/trainjs/annotations.json'
    train_json = './data/verifyjs/annotations.json'
    analyze_dataset(train_json, "(Verify)", batch_size=16)