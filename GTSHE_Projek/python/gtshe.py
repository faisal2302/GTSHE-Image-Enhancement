import numpy as np


def gtshe(image):

    histogram = np.bincount(image.ravel(), minlength=256)

    total_pixels = image.size

    tc = total_pixels / 256

    clipped_histogram = np.minimum(histogram, tc)

    total_clipped = clipped_histogram.sum()

    pdf = clipped_histogram / total_clipped

    cdf = np.cumsum(pdf)

    lnew = np.floor(cdf * 255 + 0.5).astype(np.uint8)

    occurrence = np.zeros(256, dtype=np.uint8)

    for level in range(256):
        if histogram[level] > 0:
            occurrence[lnew[level]] = 1

    total_occurrence = occurrence.sum()

    pdf_second = occurrence / total_occurrence

    cdf_second = np.cumsum(pdf_second)

    final_value = np.floor(cdf_second * 255 + 0.5).astype(np.uint8)

    mapping = final_value[lnew]

    result = mapping[image]

    return result