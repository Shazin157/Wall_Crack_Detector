import cv2
import numpy as np
from dataclasses import dataclass
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

#config files
@dataclass
class CameraCalibration:
    mm_per_pixel: float = 0.15   # PLACEHOLDER - calibrate for your robot's camera distance


@dataclass
class ClassificationThresholds:
    hairline_max_mm: float = 0.3
    medium_max_mm: float = 1.0
    # anything above medium_max_mm -> Large

#preprocessing
def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE (handles uneven lighting) + denoise."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    return gray

#segmentation
def segment_crack_classical(gray: np.ndarray) -> np.ndarray:
    # Adaptive threshold picks out locally dark, thin structures
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=7,
    )

    # Remove small noise blobs, keep thin elongated shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Filter connected components by elongation (aspect-ratio / solidity),
    # since noise blobs tend to be roughly circular while cracks are thin lines
    mask = np.zeros_like(closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:  # tiny speck, ignore
            continue
        x, y, w, h = cv2.boundingRect(c)
        elongation = max(w, h) / max(min(w, h), 1)
        if elongation >= 2.5 or area > 200:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    return mask


def segment_crack_unet(gray: np.ndarray, model=None) -> np.ndarray:

    raise NotImplementedError(
    )


#measument
@dataclass
class CrackMeasurement:
    detected: bool
    length_mm: float
    avg_width_mm: float
    max_width_mm: float
    confidence: float
    skeleton: np.ndarray
    mask: np.ndarray


def measure_crack(mask: np.ndarray, calib: CameraCalibration) -> CrackMeasurement:
    """
    Length: total skeleton pixel count (with diagonal-step correction),
            converted to mm.
    Width:  distance transform of the mask, sampled along the skeleton.
            Each skeleton pixel's distance-transform value ~= local half-width
            of the crack at that point, so width_at_point = 2 * distance value.
    """
    binary = (mask > 0)

    if binary.sum() == 0:
        empty = np.zeros_like(mask)
        return CrackMeasurement(False, 0.0, 0.0, 0.0, 0.0, empty, mask)

    skeleton = skeletonize(binary)

    # Length: count skeleton pixels, weighting diagonal moves by sqrt(2)
    length_px = _skeleton_length_px(skeleton)
    length_mm = length_px * calib.mm_per_pixel

    # Width: distance transform gives distance-to-nearest-background pixel;
    # doubling it approximates local crack thickness at each skeleton point
    dist_transform = distance_transform_edt(binary)
    widths_px = dist_transform[skeleton] * 2.0

    if widths_px.size == 0:
        avg_width_mm = max_width_mm = 0.0
    else:
        avg_width_mm = float(np.mean(widths_px)) * calib.mm_per_pixel
        max_width_mm = float(np.max(widths_px)) * calib.mm_per_pixel

    confidence = _confidence_score(binary, skeleton)

    return CrackMeasurement(
        detected=True,
        length_mm=round(length_mm, 2),
        avg_width_mm=round(avg_width_mm, 3),
        max_width_mm=round(max_width_mm, 3),
        confidence=round(confidence, 1),
        skeleton=skeleton,
        mask=mask,
    )


def _skeleton_length_px(skeleton: np.ndarray) -> float:
    ys, xs = np.nonzero(skeleton)
    if len(xs) == 0:
        return 0.0
    coords = set(zip(ys.tolist(), xs.tolist()))
    length = 0.0
    visited_edges = set()
    for (y, x) in coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in coords:
                    edge = tuple(sorted([(y, x), (ny, nx)]))
                    if edge in visited_edges:
                        continue
                    visited_edges.add(edge)
                    length += np.hypot(dy, dx)
    return length


def _confidence_score(binary: np.ndarray, skeleton: np.ndarray) -> float:

    num_components, _ = cv2.connectedComponents(binary.astype(np.uint8))
    fragmentation_penalty = min(1.0, (num_components - 1) * 0.15)
    base = 95.0
    score = base - fragmentation_penalty * 40.0
    return max(40.0, min(99.0, score))


#classification

def classify_crack(max_width_mm: float, thresholds: ClassificationThresholds) -> str:
    if max_width_mm <= 0:
        return "No Crack"
    if max_width_mm < thresholds.hairline_max_mm:
        return "Hairline"
    if max_width_mm <= thresholds.medium_max_mm:
        return "Medium"
    return "Large"

#inspection 
def run_inspection(
    image_bgr: np.ndarray,
    calib: CameraCalibration = CameraCalibration(),
    thresholds: ClassificationThresholds = ClassificationThresholds(),
    segmenter=None,
) -> dict:
    gray = preprocess(image_bgr)

    if segmenter is None:
        mask = segment_crack_classical(gray)
    else:
        mask = segmenter(image_bgr)

    result = measure_crack(mask, calib)
    crack_class = classify_crack(result.max_width_mm, thresholds)

    return {
        "detected": result.detected,
        "length_mm": result.length_mm,
        "avg_width_mm": result.avg_width_mm,
        "max_width_mm": result.max_width_mm,
        "classification": crack_class,
        "confidence": result.confidence,
        "mask": result.mask,
        "skeleton": result.skeleton,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python crack_inspector.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    report = run_inspection(img)
    print("Wall Inspection Report")
    print(f"Crack Detected: {'Yes' if report['detected'] else 'No'}")
    print(f"Length: {report['length_mm']} mm")
    print(f"Maximum Width: {report['max_width_mm']} mm")
    print(f"Classification: {report['classification']}")
    print(f"Confidence: {report['confidence']}%")
