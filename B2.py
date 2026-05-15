import cv2
import numpy as np
import matplotlib.pyplot as plt

img  = cv2.imread("3ab9c933-83ed-4bc2-ab21-7f52cf5e5fbc.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

params = [(1.0, 50), (1.0, -50), (1.5, 0), (0.5, 0)]
labels = ["a=1.0 b=+50", "a=1.0 b=-50", "a=1.5 b=0", "a=0.5 b=0"]

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes[0,0].imshow(gray, cmap="gray"); axes[0,0].set_title("Original"); axes[0,0].axis("off")
axes[1,0].plot(cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()); axes[1,0].set_title("Hist - Original")

for i, ((a, b), label) in enumerate(zip(params, labels)):
    processed = np.clip(a * gray.astype(np.float32) + b, 0, 255).astype(np.uint8)
    axes[0, i+1].imshow(processed, cmap="gray"); axes[0, i+1].set_title(label); axes[0, i+1].axis("off")
    axes[1, i+1].plot(cv2.calcHist([processed],[0],None,[256],[0,256]).ravel())
    axes[1, i+1].set_title(f"Hist - {label}"); axes[1, i+1].set_xlim([0,256])

plt.tight_layout(); plt.savefig("B2_results.png", dpi=150); plt.show()
