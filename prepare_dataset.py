import os
import re
import shutil
import argparse
import cv2
import numpy as np


def _normalize_stem(filename: str) -> str:

    stem = os.path.splitext(filename)[0].lower()
    stem = re.sub(r'(^gt[_\-]?)|([_\-]?gt$)', '', stem)
    stem = re.sub(r'(^mask[_\-]?)|([_\-]?mask$)', '', stem)
    stem = re.sub(r'(^label[_\-]?)|([_\-]?label$)', '', stem)
    return stem


def prepare_dataset(images_dir, gt_dir, out_images_dir, out_masks_dir,
                     binarize_threshold=1):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    gt_files = [f for f in os.listdir(gt_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    gt_lookup = {_normalize_stem(f): f for f in gt_files}

    matched, unmatched = 0, []

    for img_file in image_files:
        key = _normalize_stem(img_file)
        gt_file = gt_lookup.get(key)

        if gt_file is None:
            unmatched.append(img_file)
            continue

        # Copy image as-is, renamed to the shared stem
        new_stem = f"{matched:05d}"
        img_ext = os.path.splitext(img_file)[1]
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(out_images_dir, new_stem + img_ext),
        )

        # Load mask, force to clean binary 0/255 PNG regardless of source format
        mask = cv2.imread(os.path.join(gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            unmatched.append(img_file)
            continue
        binary_mask = np.where(mask >= binarize_threshold, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_masks_dir, new_stem + ".png"), binary_mask)

        matched += 1

    print(f"Matched {matched} image/GT pairs.")
    if unmatched:
        print(f"WARNING: {len(unmatched)} images had no matching GT file and were skipped:")
        for f in unmatched[:10]:
            print(f"  - {f}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
        print("Check these filenames manually — the auto-matcher may be missing a naming pattern.")

    return matched, unmatched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="folder of raw wall/crack images")
    parser.add_argument("--gt", required=True, help="folder of ground-truth mask files")
    parser.add_argument("--out-images", default="dataset/images")
    parser.add_argument("--out-masks", default="dataset/masks")
    parser.add_argument("--binarize-threshold", type=int, default=1,
                         help="GT pixel values >= this are treated as crack (255)")
    args = parser.parse_args()

    prepare_dataset(args.images, args.gt, args.out_images, args.out_masks,
                     binarize_threshold=args.binarize_threshold)
