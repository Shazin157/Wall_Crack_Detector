
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def build_report_text(report: dict, wall_id: str = "N/A") -> str:
    return (
        "Wall Inspection Report\n"
        f"Wall ID: {wall_id}\n"
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Crack Detected: {'Yes' if report['detected'] else 'No'}\n"
        f"Length: {report['length_mm']} mm\n"
        f"Average Width: {report['avg_width_mm']} mm\n"
        f"Maximum Width: {report['max_width_mm']} mm\n"
        f"Classification: {report['classification']}\n"
        f"Confidence: {report['confidence']}%\n"
    )


def build_report_json(report: dict, wall_id: str = "N/A") -> str:
    payload = {
        "wall_id": wall_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "crack_detected": report["detected"],
        "length_mm": report["length_mm"],
        "avg_width_mm": report["avg_width_mm"],
        "max_width_mm": report["max_width_mm"],
        "classification": report["classification"],
        "confidence_pct": report["confidence"],
    }
    return json.dumps(payload, indent=2)


def render_annotated_figure(image_bgr: np.ndarray, report: dict, save_path: str = None):
    """
    3-panel figure: original | binary crack mask | crack highlighted in red
    on the original image. Returns the matplotlib Figure.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = report["mask"]

    overlay = image_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]
    highlighted = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Detected Crack Mask")
    axes[1].axis("off")

    axes[2].imshow(highlighted)
    title = (
        f"{report['classification']} | {report['max_width_mm']} mm max width\n"
        f"Length {report['length_mm']} mm | Confidence {report['confidence']}%"
    )
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from crack_inspector import run_inspection

    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "test_wall.png")
    report = run_inspection(img)

    print(build_report_text(report, wall_id="ROBOT_RUN_01"))
    render_annotated_figure(img, report, save_path="annotated_report.png")
    print("Saved annotated_report.png")
