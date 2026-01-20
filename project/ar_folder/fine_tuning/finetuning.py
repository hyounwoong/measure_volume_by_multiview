"""
Depth-Anything-3 Head Fine-tuning Script

이 스크립트는 Depth-Anything-3 모델의 헤드 부분만 fine-tuning합니다.
Backbone은 freeze하고, DualDPT head만 학습합니다.

사용법:
    python finetuning.py \
        --model_name da3-large \
        --train_image_dir /path/to/train/images \
        --train_depth_dir /path/to/train/depth \
        --train_ray_dir /path/to/train/ray \
        --val_image_dir /path/to/val/images \
        --val_depth_dir /path/to/val/depth \
        --val_ray_dir /path/to/val/ray \
        --batch_size 4 \
        --num_epochs 10 \
        --lr 1e-4 \
        --output_dir ./checkpoints
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import cv2

# Depth-Anything-3 imports
from depth_anything_3.api import DepthAnything3


# ============================================
# 1. 데이터셋 클래스
# ============================================
class DepthRayDataset(Dataset):
    """
    Depth와 Ray ground truth를 포함한 데이터셋
    
    이미지, depth map, ray map을 로드합니다.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        depth_paths: List[str],
        ray_paths: List[str],
        process_res: int = 504,
        normalize_depth: bool = True,
    ):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            depth_paths: GT depth map 파일 경로 리스트 (.npy 또는 .png)
            ray_paths: GT ray map 파일 경로 리스트 (.npy)
            process_res: 이미지 처리 해상도
            normalize_depth: Depth를 정규화할지 여부
        """
        assert len(image_paths) == len(depth_paths) == len(ray_paths), \
            "이미지, depth, ray 경로의 개수가 일치해야 합니다."
        
        self.image_paths = sorted(image_paths)
        self.depth_paths = sorted(depth_paths)
        self.ray_paths = sorted(ray_paths)
        self.process_res = process_res
        self.normalize_depth = normalize_depth
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = self.image_paths[idx]
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))
        
        # Depth 로드
        depth_path = self.depth_paths[idx]
        if depth_path.endswith('.npy'):
            depth_gt = np.load(depth_path)
        else:
            # PNG로 저장된 경우 (16-bit depth)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img.dtype == np.uint16:
                depth_gt = depth_img.astype(np.float32) / 65535.0
            else:
                depth_gt = depth_img.astype(np.float32)
        
        # Ray 로드
        ray_path = self.ray_paths[idx]
        ray_gt = np.load(ray_path)  # [num_patches_y, num_patches_x, 6] 또는 [H, W, 6]
        
        # 이미지 전처리: 해상도 조정
        if image.shape[:2] != (self.process_res, self.process_res):
            image = cv2.resize(image, (self.process_res, self.process_res), interpolation=cv2.INTER_LINEAR)
        
        # Depth 전처리: 해상도 조정
        if depth_gt.shape[:2] != (self.process_res, self.process_res):
            depth_gt = cv2.resize(depth_gt, (self.process_res, self.process_res), interpolation=cv2.INTER_LINEAR)
        
        # Ray 전처리: 해상도 조정
        if len(ray_gt.shape) == 3 and ray_gt.shape[:2] != (self.process_res, self.process_res):
            # Ray map을 이미지 크기로 리사이즈
            h, w = ray_gt.shape[:2]
            ray_gt_resized = np.zeros((self.process_res, self.process_res, 6), dtype=np.float32)
            for i in range(6):
                ray_gt_resized[:, :, i] = cv2.resize(
                    ray_gt[:, :, i],
                    (self.process_res, self.process_res),
                    interpolation=cv2.INTER_LINEAR
                )
            ray_gt = ray_gt_resized
        
        # 정규화
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Depth 정규화 (선택적)
        if self.normalize_depth:
            depth_max = depth_gt.max()
            if depth_max > 0:
                depth_gt = depth_gt / depth_max
        
        # 텐서 변환
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # [3, H, W]
        depth_gt = torch.from_numpy(depth_gt).float()  # [H, W]
        ray_gt = torch.from_numpy(ray_gt).float()  # [H, W, 6]
        
        return {
            'image': image,
            'depth_gt': depth_gt,
            'ray_gt': ray_gt,
            'image_path': image_path,
        }


# ============================================
# 2. Loss 함수 정의
# ============================================
class DepthRayLoss(nn.Module):
    """
    Depth와 Ray를 위한 복합 Loss 함수
    
    - Depth Loss: L1 + Scale-invariant loss
    - Ray Loss: Origin L2 + Direction cosine similarity
    """
    
    def __init__(
        self,
        depth_weight: float = 1.0,
        ray_weight: float = 1.0,
        depth_l1_weight: float = 1.0,
        depth_si_weight: float = 0.1,
        ray_origin_weight: float = 1.0,
        ray_dir_weight: float = 1.0,
        use_conf: bool = True,
    ):
        super().__init__()
        self.depth_weight = depth_weight
        self.ray_weight = ray_weight
        self.depth_l1_weight = depth_l1_weight
        self.depth_si_weight = depth_si_weight
        self.ray_origin_weight = ray_origin_weight
        self.ray_dir_weight = ray_dir_weight
        self.use_conf = use_conf
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: {
                'depth': [B, H, W],
                'depth_conf': [B, H, W] (optional),
                'ray': [B, H, W, 6],
                'ray_conf': [B, H, W] (optional),
            }
            gt: {
                'depth_gt': [B, H, W],
                'ray_gt': [B, H, W, 6],
            }
        
        Returns:
            {
                'total_loss': scalar,
                'depth_loss': scalar,
                'ray_loss': scalar,
                'depth_l1': scalar,
                'depth_si': scalar,
                'ray_origin_loss': scalar,
                'ray_dir_loss': scalar,
            }
        """
        depth_pred = pred['depth']  # [B, H, W]
        depth_gt = gt['depth_gt']  # [B, H, W]
        
        # Depth confidence 가중치 (있는 경우)
        if self.use_conf and 'depth_conf' in pred:
            depth_conf = pred['depth_conf'].sigmoid()  # [B, H, W]
            depth_weight_mask = depth_conf
        else:
            depth_weight_mask = torch.ones_like(depth_pred)
        
        # Depth L1 Loss
        depth_diff = (depth_pred - depth_gt) * depth_weight_mask
        depth_l1 = self.l1_loss(depth_pred, depth_gt)
        
        # Scale-invariant depth loss
        depth_si = torch.mean(depth_diff ** 2) - 0.5 * torch.mean(depth_diff) ** 2
        
        depth_loss = self.depth_l1_weight * depth_l1 + self.depth_si_weight * depth_si
        
        # Ray Loss
        ray_pred = pred['ray']  # [B, H, W, 6]
        ray_gt = gt['ray_gt']  # [B, H, W, 6]
        
        # Ray origin loss (first 3 dims)
        ray_origin_pred = ray_pred[..., :3]  # [B, H, W, 3]
        ray_origin_gt = ray_gt[..., :3]  # [B, H, W, 3]
        ray_origin_loss = self.mse_loss(ray_origin_pred, ray_origin_gt)
        
        # Ray direction loss (last 3 dims)
        ray_dir_pred = ray_pred[..., 3:]  # [B, H, W, 3]
        ray_dir_gt = ray_gt[..., 3:]  # [B, H, W, 3]
        
        # Normalize directions
        ray_dir_pred_norm = ray_dir_pred / (ray_dir_pred.norm(dim=-1, keepdim=True) + 1e-8)
        ray_dir_gt_norm = ray_dir_gt / (ray_dir_gt.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Cosine similarity loss (1 - cosine similarity)
        cosine_sim = (ray_dir_pred_norm * ray_dir_gt_norm).sum(dim=-1)  # [B, H, W]
        ray_dir_loss = 1.0 - cosine_sim.mean()
        
        ray_loss = self.ray_origin_weight * ray_origin_loss + self.ray_dir_weight * ray_dir_loss
        
        # Total loss
        total_loss = self.depth_weight * depth_loss + self.ray_weight * ray_loss
        
        return {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'ray_loss': ray_loss,
            'depth_l1': depth_l1,
            'depth_si': depth_si,
            'ray_origin_loss': ray_origin_loss,
            'ray_dir_loss': ray_dir_loss,
        }


# ============================================
# 3. 모델 설정 (헤드만 학습)
# ============================================
def setup_model_for_head_training(
    model_name: str = "da3-large",
    device: str = "cuda",
    freeze_backbone: bool = True,
    freeze_cam: bool = True,
) -> DepthAnything3:
    """
    모델을 로드하고 헤드만 학습 가능하도록 설정
    
    Args:
        model_name: 모델 이름 (da3-large, da3-giant 등)
        device: 디바이스 ('cuda' 또는 'cpu')
        freeze_backbone: Backbone을 freeze할지 여부
        freeze_cam: Camera encoder/decoder를 freeze할지 여부
    
    Returns:
        설정된 모델
    """
    print(f"모델 로드 중: {model_name}")
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
    model = model.to(device)
    
    # Backbone freeze
    if freeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")
    
    # Camera encoder/decoder freeze
    if freeze_cam:
        if model.model.cam_enc is not None:
            for param in model.model.cam_enc.parameters():
                param.requires_grad = False
            print("✓ Camera encoder frozen")
        if model.model.cam_dec is not None:
            for param in model.model.cam_dec.parameters():
                param.requires_grad = False
            print("✓ Camera decoder frozen")
    
    # Head만 학습 가능
    for param in model.model.head.parameters():
        param.requires_grad = True
    print("✓ Head trainable")
    
    # GS adapter/head (있는 경우)
    if model.model.gs_adapter is not None:
        for param in model.model.gs_adapter.parameters():
            param.requires_grad = False
        print("✓ GS adapter frozen")
    if model.model.gs_head is not None:
        for param in model.model.gs_head.parameters():
            param.requires_grad = False
        print("✓ GS head frozen")
    
    # 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n파라미터 통계:")
    print(f"  학습 가능: {trainable_params:,} / {total_params:,}")
    print(f"  비율: {trainable_params/total_params*100:.2f}%")
    
    return model


# ============================================
# 4. 학습 함수
# ============================================
def train_epoch(
    model: DepthAnything3,
    train_loader: DataLoader,
    criterion: DepthRayLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """한 에포크 학습"""
    model.train()
    model.model.train()  # 내부 모델도 train 모드
    
    losses = {
        'total': [],
        'depth': [],
        'ray': [],
        'depth_l1': [],
        'depth_si': [],
        'ray_origin': [],
        'ray_dir': [],
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)  # [B, 3, H, W]
        depth_gt = batch['depth_gt'].to(device)  # [B, H, W]
        ray_gt = batch['ray_gt'].to(device)  # [B, H, W, 6]
        
        # Forward pass
        # 모델 입력 형식: [B, N, 3, H, W] (N=1 for single view)
        images_batch = images.unsqueeze(1)  # [B, 1, 3, H, W]
        
        # Gradient 계산 활성화
        # use_ray_pose=True로 설정해야 ray 출력이 유지됨 (cam_dec가 있으면 False일 때 ray 삭제됨)
        with torch.set_grad_enabled(True):
            output = model.model(
                images_batch,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=None,
                infer_gs=False,
                use_ray_pose=True,  # ray loss를 위해 True로 설정
                ref_view_strategy="saddle_balanced",
            )
        
        # Loss 계산
        # output은 addict.Dict이므로 output['key'] 또는 output.key 둘 다 가능
        pred = {
            'depth': output['depth'].squeeze(1),  # [B, 1, H, W] -> [B, H, W]
        }
        
        # Ray가 있는지 확인 (use_ray_pose=True이면 항상 있음)
        if 'ray' in output:
            pred['ray'] = output['ray'].squeeze(1)  # [B, 1, H, W, 6] -> [B, H, W, 6]
        else:
            raise ValueError("Ray output이 없습니다. use_ray_pose=True로 설정했는지 확인하세요.")
        
        # Confidence 추가 (있는 경우)
        if 'depth_conf' in output:
            pred['depth_conf'] = output['depth_conf'].squeeze(1)
        if 'ray_conf' in output:
            pred['ray_conf'] = output['ray_conf'].squeeze(1)
        
        gt = {
            'depth_gt': depth_gt,
            'ray_gt': ray_gt,
        }
        
        loss_dict = criterion(pred, gt)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping (선택적)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        for key in losses.keys():
            if key in loss_dict:
                losses[key].append(loss_dict[key].item())
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'depth': f"{loss_dict['depth_loss'].item():.4f}",
            'ray': f"{loss_dict['ray_loss'].item():.4f}",
        })
        
        # TensorBoard logging
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss_dict['total_loss'].item(), global_step)
            writer.add_scalar('train/batch_depth_loss', loss_dict['depth_loss'].item(), global_step)
            writer.add_scalar('train/batch_ray_loss', loss_dict['ray_loss'].item(), global_step)
    
    # 평균 loss 계산
    avg_losses = {key: np.mean(values) for key, values in losses.items()}
    return avg_losses


def validate(
    model: DepthAnything3,
    val_loader: DataLoader,
    criterion: DepthRayLoss,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """검증"""
    model.eval()
    model.model.eval()
    
    losses = {
        'total': [],
        'depth': [],
        'ray': [],
        'depth_l1': [],
        'depth_si': [],
        'ray_origin': [],
        'ray_dir': [],
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in pbar:
            images = batch['image'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            ray_gt = batch['ray_gt'].to(device)
            
            # Forward
            images_batch = images.unsqueeze(1)
            output = model.model(
                images_batch,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=None,
                infer_gs=False,
                use_ray_pose=True,  # ray loss를 위해 True로 설정
                ref_view_strategy="saddle_balanced",
            )
            
            # Loss
            pred = {
                'depth': output['depth'].squeeze(1),
            }
            if 'ray' in output:
                pred['ray'] = output['ray'].squeeze(1)
            else:
                raise ValueError("Ray output이 없습니다.")
            if 'depth_conf' in output:
                pred['depth_conf'] = output['depth_conf'].squeeze(1)
            if 'ray_conf' in output:
                pred['ray_conf'] = output['ray_conf'].squeeze(1)
            
            gt = {
                'depth_gt': depth_gt,
                'ray_gt': ray_gt,
            }
            
            loss_dict = criterion(pred, gt)
            
            # Logging
            for key in losses.keys():
                if key in loss_dict:
                    losses[key].append(loss_dict[key].item())
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'depth': f"{loss_dict['depth_loss'].item():.4f}",
                'ray': f"{loss_dict['ray_loss'].item():.4f}",
            })
    
    # 평균 loss 계산
    avg_losses = {key: np.mean(values) for key, values in losses.items()}
    
    # TensorBoard logging
    if writer is not None:
        writer.add_scalar('val/loss', avg_losses['total'], epoch)
        writer.add_scalar('val/depth_loss', avg_losses['depth'], epoch)
        writer.add_scalar('val/ray_loss', avg_losses['ray'], epoch)
    
    return avg_losses


def train(
    model: DepthAnything3,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "./checkpoints",
    resume_from: Optional[str] = None,
) -> DepthAnything3:
    """
    전체 학습 루프
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: 에포크 수
        lr: 학습률
        weight_decay: Weight decay
        device: 디바이스
        output_dir: 체크포인트 저장 디렉토리
        resume_from: 재개할 체크포인트 경로
    """
    # Loss 함수
    criterion = DepthRayLoss(
        depth_weight=1.0,
        ray_weight=1.0,
        depth_l1_weight=1.0,
        depth_si_weight=0.1,
        ray_origin_weight=1.0,
        ray_dir_weight=1.0,
        use_conf=True,
    )
    
    # Optimizer (학습 가능한 파라미터만)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from is not None and os.path.exists(resume_from):
        print(f"체크포인트 로드: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  에포크 {start_epoch}부터 재개")
    
    # 학습 루프
    print(f"\n학습 시작 (에포크 {start_epoch+1}~{num_epochs})")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print("-" * 60)
        
        # 학습
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 검증
        val_losses = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Learning rate 업데이트
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 결과 출력
        print(f"\n학습 Loss:")
        print(f"  Total: {train_losses['total']:.4f}")
        print(f"  Depth: {train_losses['depth']:.4f} (L1: {train_losses['depth_l1']:.4f}, SI: {train_losses['depth_si']:.4f})")
        print(f"  Ray: {val_losses['ray']:.4f} (Origin: {val_losses['ray_origin']:.4f}, Dir: {val_losses['ray_dir']:.4f})")
        
        print(f"\n검증 Loss:")
        print(f"  Total: {val_losses['total']:.4f}")
        print(f"  Depth: {val_losses['depth']:.4f}")
        print(f"  Ray: {val_losses['ray']:.4f}")
        
        print(f"\nLearning Rate: {current_lr:.6f}")
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'lr': current_lr,
        }
        
        # 최신 체크포인트
        torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))
        
        # Best 모델 저장
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))
            print(f"\n✓ Best model saved! (val_loss: {best_val_loss:.4f})")
        
        # 주기적 체크포인트 (매 5 에포크)
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch+1}.pth'))
    
    writer.close()
    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model


# ============================================
# 5. 메인 실행
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Depth-Anything-3 Head Fine-tuning")
    
    # 모델 설정
    parser.add_argument('--model_name', type=str, default='da3-large',
                        help='모델 이름 (da3-large, da3-giant 등)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스 (cuda 또는 cpu)')
    
    # 데이터 경로
    parser.add_argument('--train_image_dir', type=str, required=True,
                        help='학습 이미지 디렉토리')
    parser.add_argument('--train_depth_dir', type=str, required=True,
                        help='학습 depth map 디렉토리')
    parser.add_argument('--train_ray_dir', type=str, required=True,
                        help='학습 ray map 디렉토리')
    parser.add_argument('--val_image_dir', type=str, required=True,
                        help='검증 이미지 디렉토리')
    parser.add_argument('--val_depth_dir', type=str, required=True,
                        help='검증 depth map 디렉토리')
    parser.add_argument('--val_ray_dir', type=str, required=True,
                        help='검증 ray map 디렉토리')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='에포크 수')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker 수')
    parser.add_argument('--process_res', type=int, default=504,
                        help='이미지 처리 해상도')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='재개할 체크포인트 경로')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 설정 저장
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"설정 저장: {config_path}")
    
    # 데이터 경로 수집
    def collect_paths(image_dir, depth_dir, ray_dir, pattern='*'):
        image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, pattern)))
        ray_paths = sorted(glob.glob(os.path.join(ray_dir, pattern)))
        
        # 파일명 매칭 (확장자 제외)
        def get_basename(path):
            return os.path.splitext(os.path.basename(path))[0]
        
        # 이미지 기준으로 매칭
        matched_pairs = []
        for img_path in image_paths:
            img_base = get_basename(img_path)
            depth_path = next((d for d in depth_paths if get_basename(d) == img_base), None)
            ray_path = next((r for r in ray_paths if get_basename(r) == img_base), None)
            
            if depth_path and ray_path:
                matched_pairs.append((img_path, depth_path, ray_path))
        
        return matched_pairs
    
    print("\n데이터 경로 수집 중...")
    train_pairs = collect_paths(args.train_image_dir, args.train_depth_dir, args.train_ray_dir)
    val_pairs = collect_paths(args.val_image_dir, args.val_depth_dir, args.val_ray_dir)
    
    if len(train_pairs) == 0:
        raise ValueError("학습 데이터를 찾을 수 없습니다!")
    if len(val_pairs) == 0:
        raise ValueError("검증 데이터를 찾을 수 없습니다!")
    
    print(f"  학습 데이터: {len(train_pairs)}개")
    print(f"  검증 데이터: {len(val_pairs)}개")
    
    # 데이터셋 생성
    train_dataset = DepthRayDataset(
        image_paths=[p[0] for p in train_pairs],
        depth_paths=[p[1] for p in train_pairs],
        ray_paths=[p[2] for p in train_pairs],
        process_res=args.process_res,
    )
    
    val_dataset = DepthRayDataset(
        image_paths=[p[0] for p in val_pairs],
        depth_paths=[p[1] for p in val_pairs],
        ray_paths=[p[2] for p in val_pairs],
        process_res=args.process_res,
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    # 모델 설정
    model = setup_model_for_head_training(
        model_name=args.model_name,
        device=args.device,
        freeze_backbone=True,
        freeze_cam=True,
    )
    
    # 학습
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    
    print("\n완료!")


if __name__ == "__main__":
    import glob
    main()
