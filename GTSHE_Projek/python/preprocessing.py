import numpy as np
from PIL import Image


def preprocess_image(image, max_size=1024):

    if isinstance(image, Image.Image):
        img = image.copy()
    else:
        img = Image.open(image)

    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")

        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")

    img.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    rgb = np.asarray(img, dtype=np.float64)

    gray = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )

    gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)

    return gray