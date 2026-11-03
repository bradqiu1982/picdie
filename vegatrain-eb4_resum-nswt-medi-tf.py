import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import time
import json
import math
import glob
import torch
import torch.nn as nn
import timm
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import FasterRCNN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes

def print_env_info():
    print("="*60)
    print("🚀 启动训练环境检查...")
    print(f"🖥️ 操作系统: {sys.platform}")
    print(f"🔥 PyTorch 版本: {torch.__version__}")
    print(f"👁️ torchvision 版本: {torch.__version__}")
    print(f"🎮 CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💻 当前显卡: {torch.cuda.get_device_name(0)}")
    print("="*60 + " ")


class Config:
    def __init__(self):
        # ⚠️ 请务必替换为你自己的真实路径
        self.train_img_dir = './data/trainimg'
        self.train_json = './data/trainjs/annotations.json'
        self.val_img_dir = './data/verifyimg'
        self.val_json = './data/verifyjs/annotations.json'
        self.checkpoint_dir = './MODEL/trained_model_eb4_anch47'

        # 💡 预训练权重路径（请确保已经下载到本地）
        # self.local_timm_weights = "./data/weight/efficientnet_b4_ra2_320-7eb33cd5.pth"
        # 💡 加载预训练权重后，学习率调低，Warmup 拉长
        self.batch_size = 16            
        self.epochs = 60
        self.num_workers = 8            
        self.lr = 5e-6                  
        self.weight_decay = 1e-4
        self.warmup_epochs = 10          
        
        self.num_classes = 4            
        self.use_amp = True

        self.diagnose_interval = 200     
        self.max_checkpoints = 30        
        self.best_filename = "effnet_b4_frcnn_best.pth"

class CocoBackgroundDataset(Dataset):
    def __init__(self, img_dir, json_path, is_train=True):
        self.img_dir = img_dir
        self.is_train = is_train
        
        if is_train:
            self.transform = T.Compose([
                T.ToImage(),
                T.RandomHorizontalFlip(p=0.5), 
                T.RandomVerticalFlip(p=0.2),
                # 极轻微亮度对比度扰动（保护 CLAHE 特征）
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                # 💡 核心保留：0.8~1.3 动态多尺度缩放
                T.RandomAffine(degrees=0, scale=(0.85, 1.2), interpolation=T.InterpolationMode.BILINEAR, fill=0),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 💡 核心防线：清理缩放后越界或极小的无效框
                T.SanitizeBoundingBoxes(min_size=2.0)
            ])
        else:
            self.transform = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        with open(json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        
        self.cat_map = {c['id']: i + 1 for i, c in enumerate(sorted(coco['categories'], key=lambda x: x['id'])) if i < 3}
        self.reverse_cat_map = {v: k for k, v in self.cat_map.items()}
        
        img_to_anns = {}
        for ann in coco['annotations']:
            if ann['category_id'] not in self.cat_map: continue
            img_to_anns.setdefault(ann['image_id'], []).append(ann)
        
        id_to_fn = {img['id']: img['file_name'] for img in coco['images']}
        self.samples = []
        
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_id = next((k for k, v in id_to_fn.items() if v == fn), None)
            anns = img_to_anns.get(img_id, [])
            self.samples.append((os.path.join(img_dir, fn), anns, img_id))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, anns, img_id = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_w, img_h = img.size
        
        boxes_list, labels_list = [], []
        if anns:
            for a in anns:
                x, y, w, h = a['bbox']
                
                # 💡 核心修改：过滤 20x20 以下极小目标，斩杀噪点
                if w < 18 or h < 18:
                    continue

                x1, y1, x2, y2 = x, y, x + w, y + h
                x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                x2, y2 = min(float(x2), float(img_w)), min(float(y2), float(img_h))
                
                if x2 > x1 and y2 > y1:
                    boxes_list.append([x1, y1, x2, y2])
                    labels_list.append(self.cat_map[a['category_id']])
        
        if boxes_list:
            boxes_tv = BoundingBoxes(torch.tensor(boxes_list, dtype=torch.float32), format="XYXY", canvas_size=(img_h, img_w))
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes_tv = BoundingBoxes(torch.zeros((0, 4)), format="XYXY", canvas_size=(img_h, img_w))
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        img, target = self.transform(img, {"boxes": boxes_tv, "labels": labels_tensor})
        return img, {"boxes": target["boxes"].as_subclass(torch.Tensor), "labels": target["labels"].as_subclass(torch.Tensor)}, img_id


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


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


def build_model(num_classes=4):
    print("⏳ 正在加载 EfficientNet-B4 特征提取器...")
    timm_model = timm.create_model(
        #'efficientnet_b4', 
        'tf_efficientnet_b4.ns_jft_in1k',
        #'tf_efficientnet_b5.ns_jft_in1k',  
        pretrained=False, 
        features_only=True, 
        out_indices=(1, 2, 3, 4)
    )

    # 2. 手动加载预训练权重并过滤多余键
    print("🔗 正在手动加载过滤后的预训练权重...")
    #weight_path = "./data/weight/efficientnet_b4_ra2_320-7eb33cd5.pth"
    weight_path = "./data/weight/tf_efficientnet_b4_ns-d6313a46.pth"
    #weight_path = "./data/weight/tf_efficientnet_b5_ns-6f26d0cf.pth"
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # 提取 state_dict（兼容各种格式）
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    model_dict = timm_model.state_dict()
    filtered_dict = {}
    for k, v in state_dict.items():
        # 只保留模型里确实有的层，且形状一致的
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
            
    timm_model.load_state_dict(filtered_dict, strict=False)
    print(f"✅ 成功加载 {len(filtered_dict)} 个权重张量，忽略了无关的分类头参数。")

    print("⏳ 正在手动组装 FPN...")
    in_channels_list = timm_model.feature_info.channels()
    backbone_with_fpn = TimmWithManualFPN(timm_model, in_channels_list, out_channels=256)

    print("⏳ 正在组装原生 Faster R-CNN...")
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        min_size=1280, max_size=1280,
        rpn_fg_iou_thresh=0.3,
        rpn_bg_iou_thresh=0.2,
        box_fg_iou_thresh=0.45, 
        box_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=512,
        box_batch_size_per_image=1024
    )
	
    model.rpn.anchor_generator.sizes = ((29,), (47,), (75,), (152,), (175,))

    # 防越界崩溃盾牌
    model.rpn.score_thresh = -100.0  
    model.rpn.min_size = 0.0         
    model.roi_heads.detections_per_img = 100
    
    # 放宽 NMS（防止 Medium 误杀）
    model.roi_heads.nms_thresh = 0.6
    print(f"✅ 模型构建完成 | 输入锁定 1280 | RPN IoU 阈值 0.3 | 采样池 1024 | 越界防御开启| Anchor (56,112) | NMS 0.6")
    return model


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        print("⏳ 正在加载数据集...")
        self.train_dataset = CocoBackgroundDataset(self.cfg.train_img_dir, self.cfg.train_json, True)
        self.val_dataset = CocoBackgroundDataset(self.cfg.val_img_dir, self.cfg.val_json, False)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, collate_fn=collate_fn, pin_memory=False, drop_last=True
            ,persistent_workers=True,prefetch_factor=2
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=4, shuffle=False, 
            num_workers=self.cfg.num_workers, collate_fn=collate_fn, pin_memory=False
            ,persistent_workers=True,prefetch_factor=2
        )
        print(f"✅ 数据集加载完成 | 训练集: {len(self.train_dataset)} 张 | 验证集: {len(self.val_dataset)} 张")

        self.model = build_model(self.cfg.num_classes).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.use_amp)
        
        self.best_map = 0.0
        self.start_epoch = 0
        self.collapse_cnt = 0
        self._resume_checkpoint()

    def _create_scheduler(self):
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.cfg.epochs
        warmup_steps = steps_per_epoch * self.cfg.warmup_epochs
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _resume_checkpoint(self):
        files = glob.glob(os.path.join(self.cfg.checkpoint_dir, "checkpoint_epoch_*.pth"))
        if not files:
            print("🚀 未发现历史断点，将从 Epoch 1 开始全新训练。")
            return
            
        files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        checkpoint_path = files[-1]
        
        print(f" 🔄 发现最新断点: {checkpoint_path}，正在恢复训练状态...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
        # 1. 先恢复模型
        if 'full_model' in checkpoint:
            self.model.load_state_dict(checkpoint['full_model'].state_dict())
            self.model.to(self.device)
            print("✅ 模型权重已成功恢复。")
        else:
            print("🚨 严重警告: 断点中未找到模型权重！")
    
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        
        # 2. 恢复优化器（必须在 model.to(device) 之后）
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ 优化器状态已成功恢复。")
            except Exception as e:
                print(f"⚠️ 优化器状态恢复失败，已重置。错误: {e}")
        else:
            print("⚠️ 断点中优化器状态为空，已跳过。")
            
        # 3. 恢复学习率调度器
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ 学习率调度器已恢复。")
            except Exception as e:
                print(f"⚠️ 调度器状态恢复失败，已重置。错误: {e}")
                
        # 4. 💡 核心修复：安全恢复 Scaler，防止内部断言报错
        if 'scaler_state_dict' in checkpoint:
            scaler_state = checkpoint['scaler_state_dict']
            # 只有状态字典非空时才加载
            if scaler_state and len(scaler_state) > 0:
                try:
                    self.scaler.load_state_dict(scaler_state)
                    print("✅ AMP Scaler 状态已成功恢复。")
                except Exception as e:
                    print(f"⚠️ AMP Scaler 恢复失败，已重置。错误: {e}")
            else:
                print("⚠️ 断点中的 AMP Scaler 状态为空（可能之前未发生有效 Step），已安全跳过加载，Scaler 将重新初始化。")
        else:
            print("⚠️ 断点中未找到 AMP Scaler 状态，已安全跳过。")
            
        print(f"✅ 恢复成功！将从 Epoch {self.start_epoch} 继续训练 (历史最高 mAP: {self.best_map:.4f})。 ")

    def _generate_emergency_report(self, error_type, step, epoch, loss_dict, grad_norm):
        print(" " + "="*70)
        print(f"🚨🚨🚨 训练异常熔断: 【{error_type}】 🚨🚨🚨")
        print(f"📍 位置: Epoch {epoch+1}, Step {step}")
        print("="*70)
        print(f"【异常类型】 {error_type}")
        print(f"【Loss 明细】 {loss_dict}")
        print(f"【梯度范数】 {grad_norm}")
        print("="*70 + " ")
        exit(1)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_steps = len(self.train_loader)
        self.optimizer.zero_grad()

        # 💡 纯净耗时累加器
        acc_wait_time = 0.0      
        acc_transfer_time = 0.0  
        acc_compute_time = 0.0   
        last_time = time.time()
        
        # 纯净的 for 循环，彻底没有 tqdm 干扰
        for step, (images, targets, _) in enumerate(self.train_loader):
            if not any(len(t["boxes"]) > 0 for t in targets): 
                continue

            # 1. 等待 DataLoader 耗时
            wait_time = time.time() - last_time
            acc_wait_time += wait_time

            # 2. 传输到 GPU 耗时
            t_transfer_start = time.time()
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
            torch.cuda.synchronize() 
            transfer_time = time.time() - t_transfer_start
            acc_transfer_time += transfer_time

            # 3. 模型计算耗时
            t_compute_start = time.time()
            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            self.scaler.scale(losses).backward()
            
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if scale_before <= self.scaler.get_scale():
                self.scheduler.step()

            torch.cuda.synchronize()
            compute_time = time.time() - t_compute_start
            acc_compute_time += compute_time

            total_loss += losses.detach()
            last_time = time.time() 

            # ==========================================
            # 🚨 严格每 200 步打印一次，不用 no_grad，干掉冗余前向
            # ==========================================
            if step > 0 and step % self.cfg.diagnose_interval == 0:
                loss_details = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                loss_str = " | ".join([f"{k.replace('loss_', '')}: {v}" for k, v in loss_details.items()])
                
                obj_loss = loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
                vram = torch.cuda.max_memory_allocated() / (1024**3)

                print(f" ✅ [Epoch {epoch+1} Step {step}] Loss: {loss_str} | Obj_Loss: {obj_loss:.4f} | VRAM: {vram:.1f}G")

                total_time = acc_wait_time + acc_transfer_time + acc_compute_time
                wait_pct = (acc_wait_time / total_time) * 100
                transfer_pct = (acc_transfer_time / total_time) * 100
                compute_pct = (acc_compute_time / total_time) * 100

                # print(f"📊 ================= [Step {step} 综合性能报告] ================= ")
                # print(f" ⏳ 1. 等待加载数据 (CPU): {acc_wait_time:.2f}s ({wait_pct:.1f}%)")
                # print(f" 📦 2. 数据搬运到GPU (PCIe): {acc_transfer_time:.2f}s ({transfer_pct:.1f}%)")
                # print(f" 🔥 3. GPU纯模型计算: {acc_compute_time:.2f}s ({compute_pct:.1f}%)")
                # print(f" ─────────────────────────────────────────────────────────")
                
                # if (wait_pct + transfer_pct) > 50:
                #     print(" 🚨 【诊断结论：数据加载瓶颈】")
                #     print(" 👉 真实病因：CPU 喂数据太慢了，或者 Windows 下进程间通信太慢。GPU 超过一半的时间在空转。")
                #     print(" 👉 解决方案：必须转为 .npy 离线缓存，或者换更快的 SSD。")
                # elif compute_pct > 60:
                #     print(" ✅ 【诊断结论：模型计算瓶颈】")
                #     print(" 👉 真实病因：数据供应充足，显卡正在全力干活，这属于非常健康的状态。")
                # else:
                #     print(" ⚖️ 【诊断结论：负荷均衡或混合瓶颈】")
                # print(" ============================================================= ")
                
                acc_wait_time = 0.0
                acc_transfer_time = 0.0
                acc_compute_time = 0.0

                if torch.isnan(losses) or torch.isinf(losses):
                    self._generate_emergency_report("Loss 爆炸 (NaN/Inf)", step, epoch, loss_dict, grad_norm)
                if obj_loss < 1e-7:
                    self.collapse_cnt = self.collapse_cnt + 1
                    if self.collapse_cnt >= 5:
                        self._generate_emergency_report("模型坍塌 (Objectness极低)", step, epoch, loss_dict, grad_norm)
                else:
                    self.collapse_cnt = 0


        return (total_loss / total_steps).item()

    def _evaluate_map(self):
        print(f" ⏳ 正在计算 COCO mAP (验证集共 {len(self.val_dataset)} 张)...")
        self.model.eval()
        
        coco_gt = COCO(self.cfg.val_json)
        results = []
        
        with torch.no_grad():
            for step, (images, _, img_ids) in enumerate(self.val_loader):
                images = [img.to(self.device).float() for img in images]
                outputs = self.model(images)

                for img_id, output in zip(img_ids, outputs):
                    boxes = output["boxes"].float().cpu().numpy()
                    scores = output["scores"].float().cpu().numpy()
                    labels = output["labels"].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        orig_cat_id = self.val_dataset.reverse_cat_map.get(int(label), int(label))
                        
                        if w > 0 and h > 0 and score > 0.05:
                            results.append({
                                "image_id": int(img_id),
                                "category_id": int(orig_cat_id),
                                "bbox": [float(x1), float(y1), float(w), float(h)],
                                "score": float(score),
                            })

        self.model.train()

        if len(results) == 0:
            print("⚠️ 验证集没有检测到任何目标，mAP = 0.0")
            return 0.0

        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        print(f" 🎯 COCO 评估结果:")
        print(f" mAP@[.5:.95] : {stats[0]:.4f}")
        print(f" mAP@0.50 : {stats[1]:.4f}")
        print(f" mAP@0.75 : {stats[2]:.4f}")
        print(f" mAP (small) : {stats[3]:.4f}")
        print(f" mAP (medium) : {stats[4]:.4f}")
        print(f" mAP (large) : {stats[5]:.4f}")
        
        return stats[0]

    def train(self):
        print(f" 🚀 开始训练 (总计 {self.cfg.epochs} 轮)...")
        for epoch in range(self.start_epoch, self.cfg.epochs):
            avg_loss = self._train_one_epoch(epoch)
            
            # 💡 修改点：每一轮训练完都强制进行 COCO mAP 评估
            current_map = self._evaluate_map()
            
            print(f"📊 Epoch {epoch+1} 总结: Train Loss {avg_loss:.4f} | Val mAP {current_map:.4f} | LR {self.optimizer.param_groups[0]['lr']:.2e}")
            
            is_best = current_map > self.best_map
            if is_best:
                self.best_map = current_map
                torch.save({'epoch': epoch, 'full_model': self.model, 'best_map': self.best_map}, os.path.join(self.cfg.checkpoint_dir, self.cfg.best_filename))
                print(f"🌟 mAP 创新高，已保存最优模型 {self.cfg.best_filename}！")

            # 保存断点
            save_path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint = {
                'epoch': epoch, 
                'full_model': self.model, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_map': self.best_map
            }
            torch.save(checkpoint, save_path)
            print(f"💾 已保存断点: checkpoint_epoch_{epoch+1}.pth")

            # 滚动清理旧断点
            all_ckpts = glob.glob(os.path.join(self.cfg.checkpoint_dir, "checkpoint_epoch_*.pth"))
            if len(all_ckpts) > self.cfg.max_checkpoints:
                all_ckpts.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
                for old_ckpt in all_ckpts[:-self.cfg.max_checkpoints]:
                    try:
                        os.remove(old_ckpt)
                        print(f"🗑️ 已清理旧断点: {os.path.basename(old_ckpt)}")
                    except Exception as e:
                        print(f"⚠️ 无法删除旧断点: {e}")
            print("-" * 60)

if __name__ == '__main__':
    print_env_info()
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()