import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python pick_coords.py path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
img = plt.imread(img_path)
plt.imshow(img, cmap='gray')
plt.title("Click TOP-LEFT and BOTTOM-RIGHT of your druse, then close this window")
pts = plt.ginput(2)   # blocks until you click twice then close
plt.close()

(x0, y0), (x1, y1) = pts
print(f"--box {int(x0)},{int(y0)},{int(x1)},{int(y1)}")
