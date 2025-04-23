# restoration_utils.py

import cv2
import numpy as np
from skimage import exposure

def apply_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98)).astype(np.uint8)

def get_dark_channel(image, window_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
    flat_image = image.reshape((-1, 3))
    flat_dark = dark_channel.ravel()
    num_pixels = dark_channel.size
    num_brightest = int(max(num_pixels * top_percent, 1))
    indices = np.argpartition(-flat_dark, num_brightest)[:num_brightest]
    brightest = flat_image[indices]
    return np.mean(brightest, axis=0)

def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    norm_image = image / atmospheric_light
    dark_channel = get_dark_channel(norm_image, window_size)
    return 1 - omega * dark_channel

def recover_image(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.maximum(transmission, t0)
    result = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return np.clip(result, 0, 255).astype(np.uint8)

def dehaze(image):
    dark_channel = get_dark_channel(image)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel)
    transmission = estimate_transmission(image, atmospheric_light)
    return recover_image(image, transmission, atmospheric_light)

def calculate_snr(image):
    mean_signal = np.mean(image)
    std_noise = np.std(image)
    return float('inf') if std_noise == 0 else mean_signal / std_noise

def reduce_noise(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def sharpen_image(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

def adaptive_restoration(image):
    log = []
    brightness = np.mean(image)

    if brightness < 50:
        image = apply_gamma_correction(image, gamma=1.5)
        log.append("Applied gamma correction")

    if np.std(image) < 50:
        image = contrast_stretching(image)
        log.append("Applied contrast stretching")

    if np.mean(image) < 128:
        image = dehaze(image)
        log.append("Applied dehazing")

    snr = calculate_snr(image)
    if snr < 4.0:
        image = reduce_noise(image)
        log.append(f"Applied noise reduction (SNR: {snr:.2f})")

    image = sharpen_image(image)
    log.append("Applied sharpening")

    return image, log
