import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import cv2
import timm
import torch
torch.cuda.set_per_process_memory_fraction(0.095,device=0)
torch.backends.cudnn.benchmark=False
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch.nn as nn
import numpy as np
import argparse
import math
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from torchvision.ops import box_iou
import torchvision.transforms.functional as F
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

# ================= ⚠️ 固定目录配置区 =================
DEFAULT_MODEL_DIR = r"D:/WUXI_AI/VERIFYMD/model"
DEFAULT_IMAGE_DIR = r"D:/WUXI_AI/VERIFYMD/image/vega"
# ====================================================

class TimmWithManualFPN(nn.Module):
    def __init__(self, timm_model, in_channels_list, out_channels=256):
        super().__init__()
        self.body = timm_model
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.body(x)
        feat_dict = OrderedDict()
        for i, feat in enumerate(feats):
            feat_dict[str(i)] = feat
        return self.fpn(feat_dict)


class StatsTracker:
    def __init__(self):
        self.total = 0
        self.pass_count = 0
        self.risk_count = 0
        self.skipped_count = 0
        self.rules = defaultdict(int)
        self.conf_bins = {"0.4~0.5": 0, "0.5~0.6": 0, "0.6~0.7": 0, "0.7~0.8": 0, "0.8~0.9": 0, "0.9~1.0": 0}

    def add_risk(self, trigger_rules, max_score):
        self.total += 1
        self.risk_count += 1
        for rule in trigger_rules:
            self.rules[rule] += 1
            
        if max_score < 0.5: self.conf_bins["0.4~0.5"] += 1
        elif max_score < 0.6: self.conf_bins["0.5~0.6"] += 1
        elif max_score < 0.7: self.conf_bins["0.6~0.7"] += 1
        elif max_score < 0.8: self.conf_bins["0.7~0.8"] += 1
        elif max_score < 0.9: self.conf_bins["0.8~0.9"] += 1
        else: self.conf_bins["0.9~1.0"] += 1

    def add_pass(self):
        self.total += 1
        self.pass_count += 1

    def add_skip(self):
        self.skipped_count += 1

    def print_report(self):
        print(" " + "=" * 60)
        print("📊 双模型联合仲裁 - 批次分析报告")
        print("=" * 60)
        print(f"📁 扫描图片总数: {self.total + self.skipped_count}")
        print(f"⚠️ 跳过异常图片: {self.skipped_count} 张")
        
        pass_ratio = (self.pass_count / max(1, self.total)) * 100
        risk_ratio = (self.risk_count / max(1, self.total)) * 100
        print(f"✅ 放行 (Pass): {self.pass_count} 张 ({pass_ratio:.2f}%)")
        print(f"🚨 高风险 (Fail): {self.risk_count} 张 ({risk_ratio:.2f}%)")
        
        if self.risk_count > 0:
            print(" 🔍 触发规则分布:")
            for rule, count in self.rules.items():
                print(f" - {rule}: {count} 次")
                
            print(" 📈 高风险图最高置信度分布:")
            for bin_name, count in self.conf_bins.items():
                if count > 0:
                    print(f" - [{bin_name}]: {count} 张 ({(count / self.risk_count) * 100:.1f}%)")
        print("=" * 60 + " ")


# ================= 🚨 仅新增：Score 分析器 =================
class ScoreAnalyzer:
    def __init__(self):
        self.all_scores = []
        self.size_scores = {'small': [], 'medium': [], 'large': []}
        self.consensus_scores_20 = []
        self.consensus_scores_50 = []
        self.consensus_scores_100 = []
        
        self.GLOBAL_MIN_STAT_SCORE = 0.1  
        self.CONSENSUS_MIN_PAIR_SCORE = 0.3 

    def get_size_type(self, area):
        if area < 32**2: return 'small'
        elif area < 96**2: return 'medium'
        else: return 'large'

    def add_frame(self, a_out, b_out):
        for out in [a_out, b_out]:
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            for i in range(len(scores)):
                score = scores[i]
                if score < self.GLOBAL_MIN_STAT_SCORE: continue
                
                w = boxes[i][2] - boxes[i][0]
                h = boxes[i][3] - boxes[i][1]
                area = w * h
                s_type = self.get_size_type(area)
                self.all_scores.append(score)
                self.size_scores[s_type].append(score)

        a_boxes, a_scores = a_out['boxes'].cpu().numpy(), a_out['scores'].cpu().numpy()
        b_boxes, b_scores = b_out['boxes'].cpu().numpy(), b_out['scores'].cpu().numpy()
        for i in range(len(a_scores)):
            if a_scores[i] < self.CONSENSUS_MIN_PAIR_SCORE: continue
            best_dist = float('inf')
            best_b_idx = -1
            
            for j in range(len(b_scores)):
                if b_scores[j] < self.CONSENSUS_MIN_PAIR_SCORE: continue
                cx_a, cy_a = (a_boxes[i][0] + a_boxes[i][2]) / 2, (a_boxes[i][1] + a_boxes[i][3]) / 2
                cx_b, cy_b = (b_boxes[j][0] + b_boxes[j][2]) / 2, (b_boxes[j][1] + b_boxes[j][3]) / 2
                dist = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_b_idx = j
                    
            if best_b_idx >= 0:
                pair_score = max(a_scores[i], b_scores[best_b_idx])
                if best_dist <= 20.0:
                    self.consensus_scores_20.append(pair_score)
                elif best_dist <= 50.0:
                    self.consensus_scores_50.append(pair_score)
                elif best_dist <= 100.0:
                    self.consensus_scores_100.append(pair_score)

    def _print_stats(self, name, scores_list):
        if len(scores_list) > 0:
            cp_scores = np.array(scores_list)
            cp_percentiles = [10, 20, 30, 50, 80]
            cp_p_values = np.percentile(cp_scores, cp_percentiles)
            print(f" 【共识报警 ({name}) 阈值建议】")
            print(f" 总共收集到 {len(cp_scores)} 个共识对")
            for p, val in zip(cp_percentiles, cp_p_values):
                print(f" 👉 第 {p}% 分位数: Score = {val:.4f}")

    def print_analysis(self):
        print(" " + "=" * 60)
        print("📈 大数据 Score 阈值建议分析报告")
        print("=" * 60)
        if len(self.all_scores) == 0:
            print("❌ 数据太少，无法分析。")
            return

        all_scores = np.array(self.all_scores)
        percentiles = [50, 60, 70, 80, 90, 95]
        p_values = np.percentile(all_scores, percentiles)
        print("【全局单边报警 (Solo) 阈值建议】")
        print(f" 总共收集到 {len(all_scores)} 个单边预测框")
        for p, val in zip(percentiles, p_values):
            print(f" 👉 第 {p}% 分位数: Score = {val:.4f}")
        
        print(" 【按尺寸分类的 Solo 阈值建议】")
        for s_type, scores in self.size_scores.items():
            if len(scores) > 0:
                s_scores = np.array(scores)
                p90 = np.percentile(s_scores, 90)
                p80 = np.percentile(s_scores, 80)
                print(f" - {s_type.upper()} ({len(scores)}个): 80%分位={p80:.4f} | 90%分位={p90:.4f}")

        self._print_stats("Dist<20", self.consensus_scores_20)
        self._print_stats("Dist<50", self.consensus_scores_50)
        self._print_stats("Dist<100", self.consensus_scores_100)
        print("=" * 60 + " ")
# ========================================================


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_folder_name", type=str, help="图像文件夹名 (如: Vega)")
    parser.add_argument("model_a_name", type=str, help="模型 A 文件名 (如: r50_tuned_ep30.pth)")
    parser.add_argument("model_b_name", type=str, help="模型 B 文件名 (如: b4_ns_native_ep50.pth)")
    return parser.parse_args()


def setup_output_dir(input_dir):
    parent_dir = os.path.dirname(os.path.abspath(input_dir))
    dir_name = os.path.basename(os.path.normpath(input_dir))
    out_dir = os.path.join(parent_dir, f"{dir_name}_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_unified_model(model_path, device, expected_name=""):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型: {model_path}")
        
    print(f" ⏳ 正在加载模型 [{expected_name}]: {os.path.basename(model_path)}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # ✅ 已修复：优先提取断点中的 ema_model
    if isinstance(checkpoint, dict) and 'ema_model' in checkpoint:
        ema = checkpoint['ema_model']
        model = ema.module if hasattr(ema, 'module') else ema
        print(f" ✅ 已从断点提取 EMA 影子模型权重")
    elif isinstance(checkpoint, dict) and 'full_model' in checkpoint:
        fm = checkpoint['full_model']
        model = fm.module if hasattr(fm, 'module') else fm
        print(f" ✅ 已加载 full_model（best.pth 中即 EMA）")
    else:
        model = checkpoint
        print(f" ✅ 已加载裸模型对象")
    
    backbone_cls = type(model.backbone).__name__ if hasattr(model, 'backbone') else 'N/A'
    total_params = sum(p.numel() for p in model.parameters())
    print(f" ✅ 模型类: {type(model).__name__} | backbone: {backbone_cls} | 参数量: {total_params / 1e6:.1f}M")
    model.to(device).eval()
    return model


def gamma_correction(img, gamma=1.0):
    img_normalized = img.astype(np.float32) / 255.0
    corrected = np.power(img_normalized, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return corrected

def gamma_correction_lab(img, gamma=1.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_corrected = gamma_correction(l, gamma)
    lab_corrected = cv2.merge((l_corrected, a, b))
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def CLAHE(cimg,clipLimit,tSize):
    lab = cv2.cvtColor(cimg,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit,tileGridSize=(tSize,tSize))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe,a,b))
    return cv2.cvtColor(lab_clahe,cv2.COLOR_LAB2BGR)

def preprocess_image(img_bgr, device):
    
    # 🚨 显式尺寸检查，跳过非 1280x1280 的图片
    if img_bgr.shape[:2] != (1280, 1280):
        raise ValueError(f"❌ 输入图像尺寸异常: {img_bgr.shape[:2]}，期望 (1280, 1280)")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)
    img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img_tensor


# 🌟 修复了去重 Bug：全遍历、无 break、防断档、严格择优
def deduplicate_boxes(boxes_with_scores):
    final_boxes = []
    for box, score, area, label_id in boxes_with_scores:
        bw, bh = box[2] - box[0], box[3] - box[1]
        bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        
        overlap_indices = []
        for idx, (exist_box, exist_score, exist_area, exist_label) in enumerate(final_boxes):
            if label_id != exist_label: continue
            
            ecx, ecy = (exist_box[0] + exist_box[2]) / 2, (exist_box[1] + exist_box[3]) / 2
            ew, eh = exist_box[2] - exist_box[0], exist_box[3] - exist_box[1]
            dist = math.sqrt((ecx - bcx)**2 + (ecy - bcy)**2)
            
            if dist < 15.0 and abs(ew - bw) < 15.0 and abs(eh - bh) < 15.0:
                overlap_indices.append(idx)
                
        if not overlap_indices:
            final_boxes.append((box, score, area, label_id))
        else:
            max_exist_score = max(final_boxes[i][1] for i in overlap_indices)
            if score > max_exist_score or (abs(score - max_exist_score) < 1e-4 and area > max(final_boxes[i][2] for i in overlap_indices)):
                for i in sorted(overlap_indices, reverse=True):
                    del final_boxes[i]
                final_boxes.append((box, score, area, label_id))
                
    return [(b, s, l) for b, s, a, l in final_boxes]


# 🌟 移出 main 函数，遵守无嵌套铁律
def get_class_2(output):
    mask = output['labels'] == 2
    return {'boxes': output['boxes'][mask], 'scores': output['scores'][mask], 'labels': output['labels'][mask]}


# =====================================================================
# 🚨 规则功能块：完全平铺的 if，注释掉任意一行即可关闭该规则
# =====================================================================
def get_solo_thresh(box,filename):
    t = 0.6 # 基础兜底默认值
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    
    # 规则 A：Waveguide 出光口区域降阈值 (限定 Y < 500)
    if "_WAVEGUIDE_" in filename.upper() and cy < 500:
        t = 0.55
        
    # 规则 B：边缘惩罚升阈值 (凡触碰边缘 15 像素内，强制拉高到 0.8)
    if box[0] <= 15 or box[1] <= 15 or box[2] >= 1265 or box[3] >= 1265:
        t = 0.80
        
    return t


def get_consensus_thresh(box_a, box_b, dist,filename):
    t_a, t_b = 0.45, 0.45 # 基础兜底默认值
    cx_a, cy_a = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    
    # 规则 A：Waveguide 出光口区域降低共识阈值 (限定 Y < 500)
    if "_WAVEGUIDE_" in filename.upper() and cy_a < 500:
        t_a, t_b = 0.40, 0.40
        
    # 规则 B：边缘框单边惩罚
    if box_a[0] <= 15 or box_a[1] <= 15 or box_a[2] >= 1265 or box_a[3] >= 1265:
        t_a = 0.80
    if box_b[0] <= 15 or box_b[1] <= 15 or box_b[2] >= 1265 or box_b[3] >= 1265:
        t_b = 0.80
        
    # 规则 C：按中心点距离分级惩罚
    if 20.0 < dist <= 50.0:
        t_a += 0.05
        t_b += 0.05
    if 50.0 < dist <= 100.0:
        t_a += 0.15
        t_b += 0.15
        
    return t_a, t_b


def apply_arbitration_logic(model_a_out, model_b_out, filename=""):
    """
    核心仲裁逻辑：无条件独狼 + 中心点距离判定 + 保底机制 + 去重择优
    """
    MIN_ARBITRATION_SCORE = 0.3
    a_boxes, a_scores = model_a_out['boxes'].cpu().numpy(), model_a_out['scores'].cpu().numpy()
    b_boxes, b_scores = model_b_out['boxes'].cpu().numpy(), model_b_out['scores'].cpu().numpy()
    
    a_mask, b_mask = a_scores >= MIN_ARBITRATION_SCORE, b_scores >= MIN_ARBITRATION_SCORE
    a_boxes, a_scores = a_boxes[a_mask], a_scores[a_mask]
    b_boxes, b_scores = b_boxes[b_mask], b_scores[b_mask]

    if len(a_boxes) == 0 and len(b_boxes) == 0:
        return False, [], [], 0.0


    is_risk = False
    trigger_rules = set()
    max_score = 0.0
    raw_triggered = []

    # 1. 无条件独狼
    for i in range(len(a_boxes)):
        cx = (a_boxes[i][0] + a_boxes[i][2]) / 2
        cy = (a_boxes[i][1] + a_boxes[i][3]) / 2
        
        # 🚨 如果是 waveguide 且 Y >= 500，直接跳过不处理
        if "_WAVEGUIDE_" in filename.upper() and cy >= 500:
            continue
            
        solo_thresh = get_solo_thresh(a_boxes[i], filename)
        if a_scores[i] >= solo_thresh:
            is_risk = True
            trigger_rules.add(f"A Solo (>={solo_thresh:.2f})")
            area = (a_boxes[i][2] - a_boxes[i][0]) * (a_boxes[i][3] - a_boxes[i][1])
            raw_triggered.append((a_boxes[i], a_scores[i], area, 2))
            max_score = max(max_score, a_scores[i])

    for j in range(len(b_boxes)):
        cx = (b_boxes[j][0] + b_boxes[j][2]) / 2
        cy = (b_boxes[j][1] + b_boxes[j][3]) / 2
        
        # 🚨 如果是 waveguide 且 Y >= 500，直接跳过不处理
        if "_WAVEGUIDE_" in filename.upper() and cy >= 500:
            continue
            
        solo_thresh = get_solo_thresh(b_boxes[j], filename)
        if b_scores[j] >= solo_thresh:
            is_risk = True
            trigger_rules.add(f"B Solo (>={solo_thresh:.2f})")
            area = (b_boxes[j][2] - b_boxes[j][0]) * (b_boxes[j][3] - b_boxes[j][1])
            raw_triggered.append((b_boxes[j], b_scores[j], area, 2))
            max_score = max(max_score, b_scores[j])

    # 2. 双模型中心点距离共识
    if len(a_boxes) > 0 and len(b_boxes) > 0:
        for i in range(len(a_boxes)):
            for j in range(len(b_boxes)):
                cx_a, cy_a = (a_boxes[i][0] + a_boxes[i][2]) / 2, (a_boxes[i][1] + a_boxes[i][3]) / 2
                cx_b, cy_b = (b_boxes[j][0] + b_boxes[j][2]) / 2, (b_boxes[j][1] + b_boxes[j][3]) / 2
                
                # 🚨 如果是 waveguide 且 Y >= 500，直接跳过不处理
                if  "_WAVEGUIDE_" in filename.upper() and cy_a >= 500:
                    continue
                    
                dist = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
                
                s_a, s_b = a_scores[i], b_scores[j]
                
                t_a, t_b = get_consensus_thresh(a_boxes[i], b_boxes[j], dist, filename)
                
                rule_hit = False
                if dist <= 20.0:
                    if s_a >= t_a and s_b >= t_b:
                        is_risk = True; rule_hit = True; trigger_rules.add("Consensus Dist<20")
                elif dist <= 50.0:
                    if s_a >= t_a and s_b >= t_b:
                        is_risk = True; rule_hit = True; trigger_rules.add("Consensus Dist<50")
                elif dist <= 100.0:
                    if s_a >= t_a and s_b >= t_b:
                        is_risk = True; rule_hit = True; trigger_rules.add("Guarantee Dist<100")

                if rule_hit:
                    area_a = (a_boxes[i][2] - a_boxes[i][0]) * (a_boxes[i][3] - a_boxes[i][1])
                    area_b = (b_boxes[j][2] - b_boxes[j][0]) * (b_boxes[j][3] - b_boxes[j][1])
                    raw_triggered.append((a_boxes[i], s_a, area_a, 2))
                    raw_triggered.append((b_boxes[j], s_b, area_b, 2))
                    max_score = max(max_score, s_a, s_b)

    # 3. 调用通用去重函数
    output_triggered = deduplicate_boxes(raw_triggered)
    output_triggered = [(box, score) for box, score, lbl in output_triggered]

    return is_risk, output_triggered, list(trigger_rules), max_score


def draw_box(img_bgr, box, score, label_id, color=(0, 0, 255), pad=2, thickness=1):
    img_h, img_w = img_bgr.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    draw_x1, draw_y1 = max(0, x1 - pad), max(0, y1 - pad)
    draw_x2, draw_y2 = min(img_w - 1, x2 + pad), min(img_h - 1, y2 + pad)
    cv2.rectangle(img_bgr, (draw_x1, draw_y1), (draw_x2, draw_y2), color, thickness)
    
    prefix = "Defect" if label_id == 2 else f"Loc_{label_id}"
    label = f"{prefix} S:{int(score * 100)}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.6, thickness)
    
    text_x = draw_x1
    text_y = draw_y1 - 5 if draw_y1 - 5 - text_h > 0 else draw_y1 + text_h + 5
    if text_x + text_w > img_w:
        text_x = img_w - text_w - 5
        
    # cv2.rectangle(img_bgr, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + baseline), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (text_x, text_y), font, 0.6, color, thickness)


def main():
    args = parse_arguments()
    input_dir = os.path.join(DEFAULT_IMAGE_DIR, args.img_folder_name)
    a_model_path = os.path.join(DEFAULT_MODEL_DIR, args.model_a_name)
    b_model_path = os.path.join(DEFAULT_MODEL_DIR, args.model_b_name)

    if not os.path.isdir(input_dir):
        print(f"❌ 错误: 找不到图像目录 {input_dir}")
        sys.exit(1)

    out_dir = setup_output_dir(input_dir)
    print(f"📁 输出目录: {out_dir}")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_a = load_unified_model(a_model_path, DEVICE, "Model A")
    model_b = load_unified_model(b_model_path, DEVICE, "Model B")

    valid_exts = ( '.JPG', '.JPEG')
    img_files = [f.upper() for f in os.listdir(input_dir) if f.upper().endswith(valid_exts)]
    print(f"🔍 找到 {len(img_files)} 张图片... ")

    tracker = StatsTracker()
    analyzer = ScoreAnalyzer() # 🚨 新增

    for img_name in tqdm(img_files, desc="双模型仲裁分析中", unit="张"):
        img_path = os.path.join(input_dir, img_name)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️ 无法读取图片，跳过: {img_name}")
            tracker.add_skip()
            continue

        img_tensor = None
        try:
            hg,wd,ch = img_bgr.shape
            if hg != 1280:
                img_bgr = cv2.resize(img_bgr,(1280,1280))
                gamma = 0.9
                if '_WAVEGUIDE_' in img_name:
                    gamma = 0.7
                img_bgr = CLAHE(img_bgr,2.1,16)
                img_bgr = gamma_correction_lab(img_bgr,gamma)

            img_tensor = preprocess_image(img_bgr, DEVICE)
        except ValueError as e:
            print(f"⚠️ 跳过异常图片 {img_name}: {e}")
            tracker.add_skip()
            continue

        with torch.no_grad():
            a_out = model_a([img_tensor])[0]
            b_out = model_b([img_tensor])[0]

        # 画定位框 (Class 1 & 3，分数 >= 0.9，绿色)
        loc_boxes_to_draw = []
        for out in [a_out, b_out]:
            for i in range(len(out['boxes'])):
                s = out['scores'][i].item()
                lbl = out['labels'][i].item()
                if lbl in [1, 3] and s >= 0.9:
                    box = out['boxes'][i].cpu().numpy()
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    loc_boxes_to_draw.append((box, s, area, lbl))
                    
        deduped_loc_boxes = deduplicate_boxes(loc_boxes_to_draw)
        
        for box, s, lbl in deduped_loc_boxes:
            draw_box(img_bgr, box, s, lbl, color=(0, 255, 0))

        a_out_defect = get_class_2(a_out)
        b_out_defect = get_class_2(b_out)

        analyzer.add_frame(a_out_defect, b_out_defect) # 🚨 新增

        is_risk, triggered_boxes, trigger_rules, max_score = apply_arbitration_logic(a_out_defect, b_out_defect, filename=img_name)
        base_name, ext = os.path.splitext(img_name)

        # 最终判定与保存
        if is_risk:
            tracker.add_risk(trigger_rules, max_score)
            for (box, score) in triggered_boxes:
                draw_box(img_bgr, box, score, label_id=2, color=(0, 0, 255))
            cv2.imwrite(os.path.join(out_dir, f"{base_name}_risk{ext}"), img_bgr)
        else:
            tracker.add_pass()
            cv2.imwrite(os.path.join(out_dir, img_name), img_bgr)

    tracker.print_report()
    analyzer.print_analysis() # 🚨 新增


if __name__ == '__main__':
    main()