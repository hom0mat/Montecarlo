import os
from pathlib import Path

import cv2
import numpy as np


RESOLUTION_M_PER_PX = 0.05
CANVAS_SIZE_PX = 800
MAP_FILENAME = "warehouse_layout.png"


def draw_rotated_block(image, center_px, size_px, angle_deg, color=0):
    rect = (center_px, size_px, angle_deg)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.fillConvexPoly(image, box, color)


def create_known_map(output_dir=None):
    """Create a known occupancy-style map for the MCL exercise."""
    canvas = np.full((CANVAS_SIZE_PX, CANVAS_SIZE_PX), 255, dtype=np.uint8)

    # Outer warehouse boundary
    cv2.rectangle(canvas, (0, 0), (CANVAS_SIZE_PX - 1, CANVAS_SIZE_PX - 1), 0, 16)

    # New interior layout: shelves/cells/corridors
    cv2.rectangle(canvas, (115, 120), (205, 340), 0, -1)   # left vertical shelf
    cv2.rectangle(canvas, (265, 105), (475, 165), 0, -1)   # top horizontal shelf
    cv2.rectangle(canvas, (575, 120), (665, 315), 0, -1)   # right vertical shelf
    cv2.rectangle(canvas, (250, 335), (360, 455), 0, -1)   # central block
    cv2.rectangle(canvas, (455, 375), (625, 445), 0, -1)   # right middle shelf
    cv2.rectangle(canvas, (100, 565), (325, 635), 0, -1)   # lower left shelf
    cv2.rectangle(canvas, (520, 585), (705, 665), 0, -1)   # lower right shelf
    draw_rotated_block(canvas, (418, 610), (150, 42), -23)  # diagonal pallet row

    if output_dir is None:
        output_path = Path.cwd() / MAP_FILENAME
    else:
        output_path = Path(output_dir) / MAP_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), canvas)

    print("Known MCL map created successfully.")
    print(f" - File: {output_path}")
    print(f" - Resolution: {RESOLUTION_M_PER_PX} m/px")
    print(f" - Image size: {CANVAS_SIZE_PX} x {CANVAS_SIZE_PX} px")
    print(" - Simulated area: 40 m x 40 m")


def main():
    package_root = Path(__file__).resolve().parents[1]
    create_known_map(package_root / "maps")


if __name__ == "__main__":
    main()
