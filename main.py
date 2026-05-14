import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def show_image_real_size(title, image):
    h, w = image.shape
    dpi = 100

    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.title(f"{title} - {w}x{h}")
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


def show2comparison(original, result, title_result):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)

    plt.title(f"Imagine originala\n{original.shape[1]}x{original.shape[0]}")
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"{title_result}\n{result.shape[1]}x{result.shape[0]}")
    plt.imshow(result, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def show3comparison(original, nearest, bilinear):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Originala\n{original.shape[1]}x{original.shape[0]}")
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Nearest Neighbor\n{nearest.shape[1]}x{nearest.shape[0]}")
    plt.imshow(nearest, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Bilinear\n{bilinear.shape[1]}x{bilinear.shape[0]}")
    plt.imshow(bilinear, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def show4comparison(original, nearest, bilinear, shrink_avg):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.title(f"Originala\n{original.shape[1]}x{original.shape[0]}")
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title(f"Nearest 0.5\n{nearest.shape[1]}x{nearest.shape[0]}")
    plt.imshow(nearest, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title(f"Bilinear 0.5\n{bilinear.shape[1]}x{bilinear.shape[0]}")
    plt.imshow(bilinear, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title(f"Shrink average 2x2\n{shrink_avg.shape[1]}x{shrink_avg.shape[0]}")
    plt.imshow(shrink_avg, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# NN -> pt fiecare pixel din img noua, cauta pixelu cel mai apropiat din img veche

def resize_nearest(img, scale):
    old_h, old_w = img.shape

    new_h = int(old_h * scale)
    new_w = int(old_w * scale)

    result = np.zeros((new_h, new_w), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            old_y = round(y / scale)
            old_x = round(x / scale)

            if old_y >= old_h:
                old_y = old_h - 1
            if old_x >= old_w:
                old_x = old_w - 1

            result[y, x] = img[old_y, old_x]

    return result

#bilinear ofera rezultat mai neted, mai putin pixelat, dar este si mai lent decat nearest neighb

def resize_bilinear(img, scale):
    old_h, old_w = img.shape

    new_h = int(old_h * scale)
    new_w = int(old_w * scale)

    result = np.zeros((new_h, new_w), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            old_y = y / scale
            old_x = x / scale

            y1 = int(np.floor(old_y))
            x1 = int(np.floor(old_x))

            y2 = min(y1 + 1, old_h - 1)
            x2 = min(x1 + 1, old_w - 1)

            dy = old_y - y1
            dx = old_x - x1

            p1 = img[y1, x1]
            p2 = img[y1, x2]
            p3 = img[y2, x1]
            p4 = img[y2, x2]

            value = (
                p1 * (1 - dx) * (1 - dy) +
                p2 * dx * (1 - dy) +
                p3 * (1 - dx) * dy +
                p4 * dx * dy
            )

            result[y, x] = np.clip(value, 0, 255)

    return result

#pt fiecare bloc de 2x2 pixeli din img originala calcl media si obtinem un pixel nou

def shrink_average_2x2(img):
    old_h, old_w = img.shape

    new_h = old_h // 2
    new_w = old_w // 2

    result = np.zeros((new_h, new_w), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            block = img[2 * y:2 * y + 2, 2 * x:2 * x + 2]
            result[y, x] = int(np.mean(block))

    return result

def main():
    os.makedirs("results", exist_ok=True)

    image_path = "img/testpat1.bmp"

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Imaginea nu a fost gasita!")
        return

    print("Dimensiune originala:", img.shape)
    print("Tip imagine:", img.dtype)

    scale = float(input("Introdu factorul de scalare (2, 0.5, 1.5): "))

    start = time.time()
    nearest_img = resize_nearest(img, scale)
    end = time.time()
    nearest_time = end - start

    print("Dimensiune Nearest Neighbor:", nearest_img.shape)
    print("Timp Nearest Neighbor:", nearest_time, "secunde")

    start = time.time()
    bilinear_img = resize_bilinear(img, scale)
    end = time.time()
    bilinear_time = end - start

    print("Dimensiune Bilinear:", bilinear_img.shape)
    print("Timp Bilinear:", bilinear_time, "secunde")

    cv2.imwrite("results/nearest_result.png", nearest_img)
    cv2.imwrite("results/bilinear_result.png", bilinear_img)

    if scale == 0.5:
        start = time.time()
        shrink_img = shrink_average_2x2(img)
        end = time.time()
        shrink_time = end - start

        print("Dimensiune Shrink average 2x2:", shrink_img.shape)
        print("Timp Shrink average 2x2:", shrink_time, "secunde")

        cv2.imwrite("results/shrink_average_2x2.png", shrink_img)

        show4comparison(img, nearest_img, bilinear_img, shrink_img)
    else:
        show3comparison(img, nearest_img, bilinear_img)

    show_image_real_size("Imagine originala", img)
    show_image_real_size("Nearest Neighbor", nearest_img)
    show_image_real_size("Bilinear", bilinear_img)


if __name__ == "__main__":
    main()