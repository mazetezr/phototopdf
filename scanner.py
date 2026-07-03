import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_contour_in_edges(edged: np.ndarray, img_area: int) -> np.ndarray | None:
    """Try to find a 4-point contour in an edge map."""
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        # Try multiple epsilon values for approxPolyDP
        for eps_mult in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(contour, eps_mult * peri, True)
            if len(approx) == 4:
                contour_area = cv2.contourArea(approx)
                if contour_area > img_area * 0.1:
                    return approx.reshape(4, 2)
    return None


def _find_document_contour(image: np.ndarray) -> np.ndarray | None:
    """Find the largest 4-point contour using multiple detection strategies."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_area = image.shape[0] * image.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Strategy 1: Canny with different thresholds
    for blur_k, canny_lo, canny_hi in [(5, 50, 200), (7, 30, 150), (5, 20, 100)]:
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edged = cv2.Canny(blurred, canny_lo, canny_hi)
        edged = cv2.dilate(edged, kernel, iterations=1)
        result = _find_contour_in_edges(edged, img_area)
        if result is not None:
            return result

    # Strategy 2: Adaptive threshold (works better on compressed/low-contrast images)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    result = _find_contour_in_edges(thresh, img_area)
    if result is not None:
        return result

    # Strategy 3: Morphological close + threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    return _find_contour_in_edges(closed, img_area)


def _warp_perspective(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the document."""
    rect = _order_points(pts.astype("float32"))
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance document readability using CLAHE on the L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def scan_document(image_bytes: bytes) -> bytes:
    """
    Process a photo of a document:
    1. Find document contour
    2. Correct perspective (if contour found)
    3. Enhance contrast

    Returns processed image as PNG bytes.
    Falls back to original if processing fails.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to decode image, returning original")
            return image_bytes

        # Try to find and warp document
        contour = _find_document_contour(image)
        if contour is not None:
            logger.info("Document contour found, applying perspective correction")
            image = _warp_perspective(image, contour)
        else:
            logger.info("No document contour found, skipping perspective correction")

        # Enhance contrast
        image = _enhance_contrast(image)

        # Encode back to PNG
        success, encoded = cv2.imencode(".png", image)
        if not success:
            logger.warning("Failed to encode processed image, returning original")
            return image_bytes

        return encoded.tobytes()
    except Exception as e:
        logger.error("Scanner failed, returning original image: %s", e)
        return image_bytes
