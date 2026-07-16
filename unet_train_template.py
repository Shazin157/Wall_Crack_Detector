import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# Dataset
class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, fname + ".png")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img, mask = np.flipud(img).copy(), np.flipud(mask).copy()

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t


# U-Net model

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LightUNet(nn.Module):
    
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = conv_block(3, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# Loss function

def dice_bce_loss(pred, target, smooth=1.0):
    bce = F.binary_cross_entropy(pred, target)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return bce + dice


# Training loop

def train(image_dir, mask_dir, epochs=50, batch_size=8, lr=1e-3, img_size=256,
          checkpoint_path="crack_unet.pt", init_checkpoint=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = CrackDataset(image_dir, mask_dir, img_size=img_size, augment=True)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LightUNet().to(device)
    if init_checkpoint:
        print(f"Loading initial weights from {init_checkpoint}")
        model.load_state_dict(torch.load(init_checkpoint, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = dice_bce_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += dice_bce_loss(preds, masks).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved new best checkpoint -> {checkpoint_path}")

    return model


class UNetSegmenter:
    def __init__(self, checkpoint_path="crack_unet.pt", img_size=256, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightUNet().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.img_size = img_size

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        orig_h, orig_w = image_bgr.shape[:2]
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        pred = self.model(tensor)[0, 0].cpu().numpy()
        mask = (pred > threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

def pretrain_on_public_dataset(public_images_dir, public_masks_dir, epochs=30,
                                checkpoint_path="pretrained_unet.pt"):
    print("=== Stage 1: pretraining on public labeled dataset ===")
    return train(
        public_images_dir, public_masks_dir,
        epochs=epochs, checkpoint_path=checkpoint_path,
    )

def generate_pseudo_labels(robot_images_dir, output_mask_dir,
                            confidence_threshold=85.0, include_negatives=True):
    
    from crack_inspector import run_inspection  # local import avoids torch<->cv2 import-order issues

    os.makedirs(output_mask_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(robot_images_dir, "*.jpg")) +
                          glob.glob(os.path.join(robot_images_dir, "*.png")))

    kept, skipped, negatives = 0, 0, 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        mask_out_path = os.path.join(output_mask_dir, stem + ".png")

        report = run_inspection(img)

        if report["detected"] and report["confidence"] >= confidence_threshold:
            cv2.imwrite(mask_out_path, report["mask"])
            kept += 1
        elif not report["detected"] and include_negatives:
            blank = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.imwrite(mask_out_path, blank)
            negatives += 1
        else:
            skipped += 1  # low-confidence detection — don't trust it as a label

    print(f"Pseudo-labels: {kept} kept, {negatives} negative (no-crack), "
          f"{skipped} skipped as low-confidence out of {len(image_paths)} images")
    return kept, negatives, skipped

def finetune_on_robot_data(pretrained_checkpoint, robot_images_dir, pseudo_mask_dir,
                            epochs=20, lr=1e-4, checkpoint_path="crack_unet_finetuned.pt"):
    print("=== Stage 3: fine-tuning on robot's own pseudo-labeled photos ===")
    return train(
        robot_images_dir, pseudo_mask_dir,
        epochs=epochs, lr=lr,
        checkpoint_path=checkpoint_path,
        init_checkpoint=pretrained_checkpoint,
    )

def full_pipeline(public_images_dir, public_masks_dir, robot_images_dir,
                   pretrain_epochs=30, finetune_epochs=20,
                   pseudo_mask_dir="robot_pseudo_masks",
                   pretrained_checkpoint="pretrained_unet.pt",
                   final_checkpoint="crack_unet_finetuned.pt"):
    pretrain_on_public_dataset(
        public_images_dir, public_masks_dir,
        epochs=pretrain_epochs, checkpoint_path=pretrained_checkpoint,
    )
    generate_pseudo_labels(robot_images_dir, pseudo_mask_dir)
    finetune_on_robot_data(
        pretrained_checkpoint, robot_images_dir, pseudo_mask_dir,
        epochs=finetune_epochs, checkpoint_path=final_checkpoint,
    )
    print(f"\nDone. Use UNetSegmenter(checkpoint_path='{final_checkpoint}') "
          f"in crack_inspector.py in place of segment_crack_classical().")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)

    p_pre = sub.add_parser("pretrain", help="Stage 1: train on public labeled dataset")
    p_pre.add_argument("--images", required=True)
    p_pre.add_argument("--masks", required=True)
    p_pre.add_argument("--epochs", type=int, default=30)
    p_pre.add_argument("--checkpoint", default="pretrained_unet.pt")

    p_lbl = sub.add_parser("pseudo-label", help="Stage 2: auto-generate labels from robot photos")
    p_lbl.add_argument("--robot-images", required=True)
    p_lbl.add_argument("--out-masks", required=True)
    p_lbl.add_argument("--confidence", type=float, default=85.0)

    p_ft = sub.add_parser("finetune", help="Stage 3: fine-tune pretrained model on pseudo-labels")
    p_ft.add_argument("--pretrained", required=True)
    p_ft.add_argument("--robot-images", required=True)
    p_ft.add_argument("--pseudo-masks", required=True)
    p_ft.add_argument("--epochs", type=int, default=20)
    p_ft.add_argument("--checkpoint", default="crack_unet_finetuned.pt")

    p_full = sub.add_parser("full", help="Run all 3 stages back to back")
    p_full.add_argument("--public-images", required=True)
    p_full.add_argument("--public-masks", required=True)
    p_full.add_argument("--robot-images", required=True)
    p_full.add_argument("--pretrain-epochs", type=int, default=30)
    p_full.add_argument("--finetune-epochs", type=int, default=20)

    args = parser.parse_args()

    if args.stage == "pretrain":
        pretrain_on_public_dataset(args.images, args.masks, epochs=args.epochs,
                                    checkpoint_path=args.checkpoint)
    elif args.stage == "pseudo-label":
        generate_pseudo_labels(args.robot_images, args.out_masks,
                                confidence_threshold=args.confidence)
    elif args.stage == "finetune":
        finetune_on_robot_data(args.pretrained, args.robot_images, args.pseudo_masks,
                                epochs=args.epochs, checkpoint_path=args.checkpoint)
    elif args.stage == "full":
        full_pipeline(args.public_images, args.public_masks, args.robot_images,
                      pretrain_epochs=args.pretrain_epochs,
                      finetune_epochs=args.finetune_epochs)
