import os

import cv2
import numpy as np


def main():
    os.makedirs("data/yourset/images", exist_ok=True)
    os.makedirs("data/yourset/gt_maps", exist_ok=True)
    os.makedirs("data/yourset/fixations", exist_ok=True)

    for i in range(3):
        H, W = 240, 320
        img = np.full((H, W, 3), 255, np.uint8)
        cv2.putText(img, f"IMG{i}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.circle(img, (160, 120), 20, (0, 0, 255), -1)
        cv2.imwrite(
            f"data/yourset/images/{i:03d}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-((yy - 120) ** 2 + (xx - 160) ** 2) / (2 * (18**2))).astype(np.float32)
        g = g / g.sum()
        np.save(f"data/yourset/gt_maps/{i:03d}.npy", g)

        fix = np.zeros((H, W), np.uint8)
        fix[118:122, 158:162] = 1
        np.save(f"data/yourset/fixations/{i:03d}.npy", fix)


if __name__ == "__main__":
    main()

