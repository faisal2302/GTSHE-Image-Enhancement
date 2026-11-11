import numpy as np


def evaluate_image(original, processed):

    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    if original.shape != processed.shape:
        raise ValueError("Ukuran kedua citra harus sama.")

    difference = original - processed

    mse = np.mean(difference ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)

    histogram = np.bincount(
        processed.astype(np.uint8).ravel(),
        minlength=256
    )

    probability = histogram / processed.size

    probability = probability[probability > 0]

    entropy = -np.sum(probability * np.log2(probability))

    standard_deviation = np.std(processed)

    mean_intensity = np.mean(processed)

    unique_levels = np.count_nonzero(histogram)

    return {
        "MSE": mse,
        "PSNR": psnr,
        "Entropy": entropy,
        "Standard Deviation": standard_deviation,
        "Mean Intensity": mean_intensity,
        "Unique Gray Levels": unique_levels
    }