import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

# Import DINOv3 official components
from dinov3.eval.utils import ModelWithIntermediateLayers
from dinov3.eval.setup import ModelConfig, load_model_and_context
import dinov3.distributed as distributed

# ==========================================
# 1. Command line argument parsing
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="DINOv3 Linear Probing for BRACS")
    
    # Required arguments aligned with user specifications
    parser.add_argument("--repo_dir", type=str, required=True, help="Repo root directory")
    parser.add_argument("--arch", type=str, default="dinov3_vit7b16", help="Model architecture")
    parser.add_argument("--weights", type=str, required=True, help="Path to pretrained weights")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output path")
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()

# ==========================================
# 2. Dataset loading
# ==========================================
def get_bracs_datasets(data_dir):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.511375, 0.598449, 0.683452], std=[0.340017, 0.306132, 0.284308])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.511375, 0.598449, 0.683452], std=[0.340017, 0.306132, 0.284308])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val)
    
    return train_ds, val_ds

# ==========================================
# 3. Linear classification head
# ==========================================
class BRACSLinearHead(nn.Module):
    def __init__(self, input_dim=4096, num_classes=7):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)
    
# ==========================================
# 4. Evaluation logic
# ==========================================
@torch.no_grad()
def run_evaluation(feature_model, head, loader, device, autocast_ctx, num_classes):
    feature_model.eval()
    head.eval()
    
    all_probs = []
    all_targets = []
    
    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(device)
        with autocast_ctx():

            feats = feature_model(imgs)[0][1].float()
            logits = head(feats)
            probs = torch.softmax(logits, dim=1)
            
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())

    probs_arr = np.concatenate(all_probs)
    targets_arr = np.concatenate(all_targets)
    preds_bin = np.argmax(probs_arr, axis=1)

    acc = accuracy_score(targets_arr, preds_bin)
    f1 = f1_score(targets_arr, preds_bin, average='macro', zero_division=0)
    
    # targets_one_hot = label_binarize(targets_arr, classes=range(num_classes))
    # auprc = average_precision_score(targets_one_hot, probs_arr, average='macro')
    if num_classes == 2:
        auprc = average_precision_score(targets_arr, probs_arr[:, 1])
    else:
        targets_one_hot = label_binarize(targets_arr, classes=range(num_classes))
        auprc = average_precision_score(targets_one_hot, probs_arr, average='macro')
    
    return acc, f1, auprc
    

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not distributed.is_enabled():
        distributed.enable()

    # --- 1. Determine feature dimension based on architecture ---
    # DINOv3 feature dimensions: 7B=4096, L=1024, B=768, S=384
    embed_dims = {"dinov3_vits16": 384, "dinov3_vitb16": 768, "dinov3_vitl16": 1024, "vit7b": 4096}
    feat_dim = 4096 # default
    for k, v in embed_dims.items():
        if k in args.arch.lower():
            feat_dim = v
            break

    # --- 2. Load model from local hub ---
    # model_cfg = ModelConfig(pretrained_weights=args.weights)
    print(f"Loading model {args.arch} from local hub...")

    model = torch.hub.load(
        args.repo_dir, 
        args.arch, 
        source='local', 
        pretrained=False  
    )

    # Load pretrained weights
    print(f"Loading weights from: {args.weights}")
    full_data = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = full_data["model"] if "model" in full_data else full_data

    # --- Automatically find the best matching prefix ---
    preferred_prefixes = ["model_ema.backbone.", "teacher.backbone.", "student.backbone.", "backbone.", ""]
    best_prefix = ""
    for p in preferred_prefixes:
        matches = sum(1 for k in state_dict if k.startswith(p) and "blocks" in k)
        if matches > 0:
            best_prefix = p
            print(f"Selected prefix: {p} (matched {matches} layers)")
            break

    # Construct clean state_dict adapted to current architecture
    clean_state_dict = {}
    is_vit_s = "vits" in args.arch.lower()
    model_state = model.state_dict()

    print(f"Detected architecture type: {'ViT-S (SwiGLU)' if is_vit_s else 'ViT-B/L (Standard)'}")

    for k, v in state_dict.items():
        if k.startswith(best_prefix):
            new_k = k[len(best_prefix):]
            
            if is_vit_s:
                # --- ViT-S specific: map SwiGLU (w1, w2, w3) to (fc1, fc2) ---
                if "mlp.w1" in new_k:
                    new_k = new_k.replace("mlp.w1", "mlp.fc1")
                elif "mlp.w3" in new_k:
                    new_k = new_k.replace("mlp.w3", "mlp.fc2")
                elif "mlp.w2" in new_k:
                    # Skip w2 entirely as it has no counterpart in standard architectures
                    continue
            else:
                # --- ViT-B/L standard logic: w1->fc1, w2->fc2 ---
                new_k = new_k.replace("mlp.w1", "mlp.fc1").replace("mlp.w2", "mlp.fc2")

            # --- Shape fixing for Linear layer weight transpositions ---
            if new_k in model_state:
                target_shape = model_state[new_k].shape
                if v.shape != target_shape:
                    # Case A: Weight needs transposition [out, in] <-> [in, out]
                    if v.ndim == 2 and v.t().shape == target_shape:
                        v = v.t()
                    # Case B: Final defense for S architecture bias mismatches
                    elif is_vit_s and v.ndim == 1 and v.shape[0] != target_shape[0]:
                        print(f"  [Ignored] Bias dimension mismatch: {new_k} {list(v.shape)} -> {list(target_shape)}")
                        continue

            clean_state_dict[new_k] = v

    # Load weights with strict=False for S architecture (w2 layer was skipped)
    load_strict = not is_vit_s 
    
    # Record weights before loading for verification
    test_param_name = "blocks.0.attn.qkv.weight"
    if test_param_name in model_state:
        before = model.state_dict()[test_param_name].clone()
    else:
        before = None

    msg = model.load_state_dict(clean_state_dict, strict=load_strict)
    
    print("\n===== Loading Results Report =====")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    if before is not None:
        after = model.state_dict()[test_param_name]
        changed = not torch.allclose(before, after)
        print(f"Weight update successful (core parameters changed): {changed}")
        if not changed:
            print("Warning: Model weights appear unchanged - check prefix matching!")

    model.eval().to(args.device)

    # Setup automatic mixed precision context
    autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    
    # Wrap model to extract intermediate layer features
    feature_model = ModelWithIntermediateLayers(model, 1, autocast_ctx).to(args.device)
    
    # Freeze all backbone parameters
    for p in model.parameters():
        p.requires_grad = False

    # --- 3. Prepare data loaders ---
    train_ds, val_ds = get_bracs_datasets(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)
    num_classes = len(train_ds.classes)

    # --- 4. Linear head and optimizer ---
    head = BRACSLinearHead(input_dim=feat_dim, num_classes=num_classes).to(args.device)
    
    optimizer = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # --- 5. Training loop ---
    print(f"Starting training on {num_classes} classes...")
    start_epoch = 0
    best_acc = 0.0
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
    
    # Resume from checkpoint if requested
    if args.resume and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        head.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        head.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            
            with autocast_ctx():
                # Extract features (frozen backbone, no gradients)
                with torch.no_grad():
                    feats = feature_model(imgs)[0][1].float()
                logits = head(feats)
                loss = criterion(logits, labels)
            
            # Backpropagation with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6g}"})
        
        scheduler.step()

        # Validation (every 2 epochs or at the end)
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            acc, f1, auprc = run_evaluation(feature_model, head, val_loader, args.device, autocast_ctx, num_classes)
            print(f" >> [Val] Epoch {epoch+1}: Acc: {acc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}")
            current_best = max(acc, best_acc)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': current_best,
            }, ckpt_path)
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                torch.save(head.state_dict(), os.path.join(args.output_dir, "best_linear_head.pth"))

    # --- 6. Final evaluation ---
    acc, f1, auprc = run_evaluation(feature_model, head, val_loader, args.device, autocast_ctx, num_classes)

    print(f"\nFinal Performance for BRACS ({num_classes} classes):")
    print(f"{'='*45}")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"F1 (Macro):      {f1 * 100:.2f}%")
    print(f"AUPRC (Macro):   {auprc * 100:.2f}%")
    print(f"{'='*45}")

    # Save metrics to file
    with open(os.path.join(args.output_dir, "bracs_final_metrics.txt"), "w") as f:
        f.write(f"Acc: {acc:.4f}\nF1: {f1:.4f}\nAUPRC: {auprc:.4f}\n")

if __name__ == "__main__":
    main()
