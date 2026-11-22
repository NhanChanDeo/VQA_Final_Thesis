"""
Proposed CrossVQA M3 Architecture:
Curriculum Pseudo-Domain Transform simulating 3D scene perspective homographies,
directional shadow gradients, optical blur, and JPEG artifacts on planar 2D scans.
"""

import io
from typing import List, Optional
import numpy as np
from PIL import Image, ImageFilter


def _find_perspective_coeffs(
    dst_pts: np.ndarray,
    src_pts: np.ndarray,
) -> List[float]:
    """Solves 8-parameter perspective transform matrix mapping dst -> src coordinates."""
    matrix = []
    for p1, p2 in zip(dst_pts, src_pts):
        matrix.append([p1[0], p1[1], 1.0, 0.0, 0.0, 0.0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0.0, 0.0, 0.0, p1[0], p1[1], 1.0, -p2[1] * p1[0], -p2[1] * p1[1]])
    a_mat = np.asarray(matrix, dtype=np.float64)
    b_vec = np.asarray(src_pts, dtype=np.float64).reshape(8)
    coeffs = np.linalg.solve(a_mat, b_vec)
    return coeffs.tolist()


class CurriculumPseudoDomainTransform:
    """
    Applies severity-controlled 3D perspective warp, directional illumination shadow,
    Gaussian blur, and JPEG compression artifacts to planar DocVQA images.
    """

    def __init__(self, severity: float = 0.5, seed: Optional[int] = None) -> None:
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def set_severity(self, severity: float) -> None:
        self.severity = float(np.clip(severity, 0.0, 1.0))

    @staticmethod
    def linear_schedule(step: int, total_steps: int, start: float = 0.2, end: float = 1.0) -> float:
        if total_steps <= 1:
            return float(end)
        frac = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
        return float(start + frac * (end - start))

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        if self.severity <= 0.0:
            return rgb

        w, h = rgb.size
        s = self.severity

        # 1. Perspective Homography Warp (corner jitter up to 12% at severity=1.0)
        max_dx = 0.12 * s * w
        max_dy = 0.12 * s * h
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        jitter = self.rng.uniform(-1.0, 1.0, size=(4, 2))
        jitter[:, 0] *= max_dx
        jitter[:, 1] *= max_dy
        dst_pts = src_pts + jitter
        coeffs = _find_perspective_coeffs(dst_pts, src_pts)
        bg_color = tuple(int(c) for c in self.rng.integers(180, 245, size=3))
        warped = rgb.transform(
            (w, h),
            Image.Transform.PERSPECTIVE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
            fillcolor=bg_color,
        )

        # 2. Directional illumination / soft shadow gradient
        arr = np.asarray(warped, dtype=np.float32)
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        grad = np.cos(angle) * gx + np.sin(angle) * gy
        shadow_strength = 0.30 * s
        illumination = np.clip(1.0 + shadow_strength * grad, 0.55, 1.15)[..., None]
        arr = np.clip(arr * illumination, 0.0, 255.0).astype(np.uint8)
        out_img = Image.fromarray(arr, mode="RGB")

        # 3. Optical Gaussian Blur
        blur_radius = float(self.rng.uniform(0.2, 0.2 + 1.0 * s))
        if blur_radius > 0.25:
            out_img = out_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 4. Mild JPEG Compression Perturbation
        jpeg_quality = int(round(92 - 24 * s))
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=max(50, min(95, jpeg_quality)))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
