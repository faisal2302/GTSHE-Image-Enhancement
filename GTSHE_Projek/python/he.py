import numpy as np


def histogram_equalization(image):

    histogram = np.bincount(image.ravel(), minlength=256)

    pdf = histogram / image.size

    cdf = np.cumsum(pdf)

    mapping = np.round(cdf * 255).astype(np.uint8)

    result = mapping[image]

    return result