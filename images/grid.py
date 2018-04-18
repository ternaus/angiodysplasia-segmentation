import cv2
import numpy as np

SZ = 576
H_PAD = 5
V_PAD1 = 50
V_PAD2 = 5
N_COLS = 5
N_ROWS = 3

lesion_path = "data/train/angyodysplasia/images/{}.jpg"
masks_path = "data/train/angyodysplasia/masks/{}_a.jpg"
normal_path = "data/train/normal/images/expert{}.png"

lesion = [1103, 1118, 1122, 1133, 1137]
normal = [1, 2, 3, 4, 5]

img = np.ones((SZ * N_ROWS + V_PAD1 + V_PAD2 * (N_ROWS - 2), SZ * N_COLS + H_PAD * (N_COLS - 1), 3), dtype=np.uint8) * 255

for i, (l, n) in enumerate(zip(lesion, normal)):
    im_l = cv2.imread(lesion_path.format(l))
    im_m = cv2.imread(masks_path.format(l))
    im_n = cv2.imread(normal_path.format(n))

    img[0: SZ, (SZ + H_PAD) * i: (SZ + H_PAD) * i + SZ, :] = im_n
    img[SZ + V_PAD1: 2 * SZ + V_PAD1, (SZ + H_PAD) * i: (SZ + H_PAD) * i + SZ, :] = im_l
    img[2 * SZ + V_PAD1 + V_PAD2: 2 * (SZ + V_PAD1 + V_PAD2) + SZ, (SZ + H_PAD) * i: (SZ + H_PAD) * i + SZ, :] = im_m

cv2.imwrite("images/tbl.png", img)