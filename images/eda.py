import cv2
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

masks = Path("data/train/angyodysplasia/masks").glob("*.jpg")

nb_components = []
areas = []
for mask_path in masks:
    mask = cv2.imread(str(mask_path))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(mask, connectivity=8, stats=cv2.CV_32S)
    n_comp = output[0]
    nb_components.append(n_comp - 1)
    for i in range(n_comp - 1):
        area = output[2][i + 1, cv2.CC_STAT_AREA]
        areas.append(area)
    pass

print(np.histogram(nb_components, bins=range(10)))
print(np.histogram(areas))

plt.figure(figsize=(20, 10))
sns.set(color_codes=True)
sns.set(font_scale=2)
col_list = ["blue", "deep teal", "viridian", "twilight blue", "gunmetal", "cool blue", "warm grey", "dusky blue"]
sns.set_palette(sns.xkcd_palette(col_list))

plt.subplot(121)
plt.title("Distribution of lesions per image")
sns.distplot(nb_components, kde=False, axlabel="Lesion count")

plt.subplot(122)
plt.title("Distribution of lesion area")
sns.distplot(areas, kde=False, axlabel="Lesion area, pixels")
plt.tight_layout()
plt.savefig("hist.png")
plt.show()

