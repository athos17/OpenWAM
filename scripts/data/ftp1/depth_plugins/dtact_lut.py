"""Calibrated darkening-to-indentation reconstruction for a matched DTact sensor.

This is only a numerical backend. Its caller must supply the correct sensor's
monotonic response lookup and verified zero-contact baseline. RGB camera values,
reference differences, and physical pressure are not interchangeable.

For exUMI xy-representation, a verified RGB gray_channel=2 selects the preserved
current grayscale; channels 0/1 must not be decoded as monotonic response because
source uint8 multiplication may wrap. This option does not establish calibration
or certify that an episode starts without contact.
"""

import cv2
import numpy as np


def reconstruct(*, image, baseline, calibration, options):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected spatial RGB image")
    channel = options.get("gray_channel")
    if channel is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif type(channel) is int and 0 <= channel < 3:
        gray = image[..., channel]
    else:
        raise ValueError("gray_channel must be null or RGB channel 0, 1, 2")
    baseline = np.asarray(baseline, dtype=np.float32)
    if baseline.shape != gray.shape:
        raise ValueError("DTact requires a spatially registered zero-contact baseline")
    response = np.asarray(calibration["response_lut"], dtype=np.float32)
    if response.ndim != 1 or len(response) < 2 or not np.isfinite(response).all() or (np.diff(response) < 0).any():
        raise ValueError("Response lookup must be calibrated, finite and monotonic")
    reference_index = options.get("reference_index")
    if type(reference_index) is not int or not 0 <= reference_index < len(response):
        raise ValueError("A calibrated zero-depth reference_index is required")
    noise_floor = float(options.get("noise_floor", 0))
    blur_kernel = options.get("blur_kernel", 7)
    if not np.isfinite(noise_floor) or noise_floor < 0:
        raise ValueError("Noise floor must be fixed and nonnegative")
    if type(blur_kernel) is not int or blur_kernel < 1 or blur_kernel % 2 != 1:
        raise ValueError("Blur kernel must be an odd positive integer")
    # Signed float arithmetic prevents the source's uint8 subtraction wrap.
    # LUT index zero need not be zero indentation (the published right-sensor
    # LUT even starts negative). Use the explicitly calibrated reference index.
    # With reference_index=2 and noise_floor=0 this implements
    # LUT[max(reference-gray, 2)] - LUT[2], without uint8 wrap.
    index = np.clip(baseline - gray - noise_floor, reference_index, len(response) - 1)
    if blur_kernel > 1:
        index = cv2.GaussianBlur(index, (blur_kernel, blur_kernel), 0)
    depth = np.interp(index, np.arange(len(response)), response) - response[reference_index]
    return np.maximum(depth, 0).astype(np.float32)
