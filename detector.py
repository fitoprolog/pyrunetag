import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def scalar_division(matrix, scalar):
    if abs(scalar) > 1e-18:
        matrix /= scalar
    return matrix


def ellipse_center(e):
    e00 = e[0, 0]
    e01 = e[0, 1]
    e11 = e[1, 1]
    e02 = e[0, 2]
    e12 = e[1, 2]
    denom = e01 * e01 - e00 * e11
    centerx = (e11 * e02 - e01 * e12) / denom
    centery = (e00 * e12 - e01 * e02) / denom
    return np.array([centerx, centery], dtype=np.float64)


def transform_to_ellipse(qc, vr, k):
    return (vr @ qc @ vr.T) / k


def transform_to_circle(q, vr):
    target = vr.T @ q @ vr
    nr = (target[0, 0] + target[1, 1]) / 2.0
    return scalar_division(target, nr)


def circle_center_for_vr(ellipse_norm, vr):
    return ellipse_center(transform_to_circle(ellipse_norm, vr))


def circle_r_on_z0(qc):
    a = qc[0, 0]
    b = qc[0, 1]
    c = qc[1, 1]
    d = qc[0, 2]
    f = qc[1, 2]
    g = qc[2, 2]
    num = a * f * f + c * d * d + g * b * b - 2.0 * b * d * f - a * c * g
    den = (b * b - a * c) * (((a + c) * (a + c)) - (((a - c) * (a - c)) + 4.0 * b * b))
    return -2.0 * num * (a + c) / den


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def build_n(vr):
    return normalize_vector(np.array([vr[0, 2], vr[1, 2], vr[2, 2]], dtype=np.float64))


def fit_circle_matrix(a, b, radius_sq):
    return np.array(
        [
            [1.0, 0.0, a],
            [0.0, 1.0, b],
            [a, b, a * a + b * b - radius_sq],
        ],
        dtype=np.float64,
    )


def fit_circles(center1, center2, radius_sq):
    x1, y1 = center1
    x2, y2 = center2
    base = x1 * x1 - 2.0 * x1 * x2 + x2 * x2 + (y1 - y2) * (y1 - y2)
    root_term = -(x1 - x2) * (x1 - x2) * base * (-4.0 * radius_sq + base)
    if root_term < 0.0:
        return None, None
    anum = (
        -x1**4
        + 2.0 * x1**3 * x2
        - 2.0 * x1 * x2**3
        + x2**4
        - x1 * x1 * (y1 - y2) * (y1 - y2)
        + x2 * x2 * (y1 - y2) * (y1 - y2)
    )
    adenom = 2.0 * (x1 - x2) * base
    bnum = -(y1**3 - y1 * y1 * y2 - y1 * y2 * y2 + y2**3 + x1 * x1 * (y1 + y2) - 2.0 * x1 * x2 * (y1 + y2) + x2 * x2 * (y1 + y2))
    bdenom = 2.0 * base
    sqrt_term = math.sqrt(root_term)
    a2 = (anum + (-y1 + y2) * sqrt_term) / adenom
    a1 = (anum + (y1 - y2) * sqrt_term) / adenom
    b1 = (bnum - sqrt_term) / bdenom
    b2 = (bnum + sqrt_term) / bdenom
    return fit_circle_matrix(a1, b1, radius_sq), fit_circle_matrix(a2, b2, radius_sq)


def point_in_ellipse(ellipse, point):
    x = point[0]
    y = point[1]
    return (
        ellipse[0, 0] * x * x
        + 2.0 * ellipse[0, 1] * x * y
        + ellipse[1, 1] * y * y
        + 2.0 * ellipse[0, 2] * x
        + 2.0 * ellipse[1, 2] * y
        + ellipse[2, 2]
        < 0.0
    )


def points_in_ellipse(ellipse, points):
    x = points[:, 0]
    y = points[:, 1]
    return (
        ellipse[0, 0] * x * x
        + 2.0 * ellipse[0, 1] * x * y
        + ellipse[1, 1] * y * y
        + 2.0 * ellipse[0, 2] * x
        + 2.0 * ellipse[1, 2] * y
        + ellipse[2, 2]
        < 0.0
    )


def transform_point_from_circle_to_ellipse(point, vr, focal):
    sample = vr @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    return np.array([sample[0] / sample[2] * focal, sample[1] / sample[2] * focal], dtype=np.float64)


@dataclass
class EllipsePoint:
    rr: tuple
    ellipse: np.ndarray = field(init=False)
    ellipse_norm: np.ndarray = field(init=False)
    vr1: np.ndarray = field(init=False)
    vr2: np.ndarray = field(init=False)
    n1: np.ndarray = field(init=False)
    n2: np.ndarray = field(init=False)
    l2inv: float = field(init=False)
    area: float = field(init=False)
    unassigned: bool = field(default=True)
    slotted: bool = field(default=True)
    _center: np.ndarray = field(default=None)
    _center_norm: np.ndarray = field(default=None)

    def __post_init__(self):
        center, size, angle = self.rr
        cx, cy = center
        width, height = size
        self.area = float(width * height)
        theta = math.radians(angle)
        rinv = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float64,
        )
        tinv = rinv @ np.array([[-cx], [-cy]], dtype=np.float64)
        a = width / 2.0
        b = height / 2.0
        qcan = np.array(
            [
                [1.0 / (a * a), 0.0, 0.0],
                [0.0, 1.0 / (b * b), 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        rtinv = np.array(
            [
                [rinv[0, 0], rinv[0, 1], tinv[0, 0]],
                [rinv[1, 0], rinv[1, 1], tinv[1, 0]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.ellipse = rtinv.T @ qcan @ rtinv

    def calc_vr(self, focal):
        self.ellipse_norm = self.ellipse.copy()
        self.ellipse_norm[0, 2] /= focal
        self.ellipse_norm[1, 2] /= focal
        self.ellipse_norm[2, 0] /= focal
        self.ellipse_norm[2, 1] /= focal
        self.ellipse_norm[2, 2] /= focal * focal
        _, singular_values, vt = np.linalg.svd(self.ellipse_norm)
        v = vt.T
        l1, l2, l3 = singular_values
        self.l2inv = 1.0 / l2
        g = math.sqrt((l2 - l3) / (l1 - l3))
        h = math.sqrt((l1 - l2) / (l1 - l3))
        found = []
        for s1, s2 in ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
            r = np.array(
                [
                    [g, 0.0, s2 * h],
                    [0.0, -s1, 0.0],
                    [s1 * s2 * h, 0.0, -s1 * g],
                ],
                dtype=np.float64,
            )
            vr = v @ r
            if vr[2, 2] > 0.0:
                found.append(vr)
            if len(found) == 2:
                break
        if len(found) != 2:
            raise ValueError("unable to compute ellipse orientations")
        self.vr1, self.vr2 = found
        self.n1 = build_n(self.vr1)
        self.n2 = build_n(self.vr2)

    def to_circle(self, vr):
        self._center_norm = None
        circle = vr.T @ self.ellipse_norm @ vr
        nr = (circle[0, 0] + circle[1, 1]) / 2.0
        self.ellipse_norm = scalar_division(circle, nr)

    def center(self):
        if self._center is None:
            self._center = ellipse_center(self.ellipse)
        return self._center

    def center_norm(self):
        if self._center_norm is None:
            self._center_norm = ellipse_center(self.ellipse_norm)
        return self._center_norm

    def is_assigned(self):
        return not self.unassigned

    def contains(self, point):
        return point_in_ellipse(self.ellipse, point)


class EllipseDetector:
    def __init__(
        self,
        min_contour_points=10,
        max_contour_points=300,
        min_area=10.0,
        max_area=1000.0,
        min_roundness=0.3,
        max_mse=0.24,
        size_compensation=-1.5,
        max_detected=48,
    ):
        self.min_contour_points = min_contour_points
        self.max_contour_points = max_contour_points
        self.min_area = min_area
        self.max_area = max_area
        self.min_roundness = min_roundness
        self.max_mse = max_mse
        self.size_compensation = size_compensation
        self.max_detected = max_detected

    def detect(self, image):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 15)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for contour in contours:
            if len(contour) < self.min_contour_points or len(contour) > self.max_contour_points:
                continue
            rr = cv2.fitEllipse(contour)
            (cx, cy), (w, h), angle = rr
            area = w * h * math.pi
            if area < self.min_area or area > self.max_area:
                continue
            ratio = w / h
            if ratio < self.min_roundness or ratio > 1.0 / self.min_roundness:
                continue
            major = max(w, h)
            minor = min(w, h)
            eval_angle = math.radians(angle + (90.0 if w < h else 0.0))
            focus_len = math.sqrt(max(0.0, major * major / 4.0 - minor * minor / 4.0))
            fx = math.cos(eval_angle) * focus_len
            fy = math.sin(eval_angle) * focus_len
            f1 = np.array([cx - fx, cy - fy], dtype=np.float64)
            f2 = np.array([cx + fx, cy + fy], dtype=np.float64)
            points = contour[:, 0, :].astype(np.float64)
            dk = np.linalg.norm(points - f1, axis=1) + np.linalg.norm(points - f2, axis=1) - major
            mse = math.sqrt(float(np.dot(dk, dk))) / len(contour)
            if mse > self.max_mse:
                continue
            fixed_angle = 180.0 - angle
            while fixed_angle >= 360.0:
                fixed_angle -= 360.0
            detected.append((area, ((cx, cy), (w + self.size_compensation, h + self.size_compensation), fixed_angle)))
        detected.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in detected[: self.max_detected]]


@dataclass
class Slot:
    qmin: np.ndarray
    qmax: np.ndarray
    c: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    slot_center: np.ndarray
    value: bool = False
    discarded: bool = False
    payload: EllipsePoint = None

    def invalidate(self):
        self.value = False
        self.discarded = True

    def check_inside(self, point):
        vt = point - self.c
        if self.v1[0] * vt[1] - self.v1[1] * vt[0] > 0.0:
            return False
        if self.v2[0] * vt[1] - self.v2[1] * vt[0] < 0.0:
            return False
        return point_in_ellipse(self.qmax, point) and not point_in_ellipse(self.qmin, point)

    def set_if_inside(self, point, ellipse_point):
        if self.check_inside(point) and ellipse_point.contains(self.slot_center):
            self.value = True
            self.payload = ellipse_point
            return True
        return False


@dataclass
class MarkerModel:
    name: str
    world_size: float
    world_unit: str
    num_slots: int
    radius_ratio: float
    num_layers: int
    gap_factor: float
    max_distance: int
    idx: int
    bcode: list
    ellipses: dict
    bcode_np: np.ndarray = None
    bcode_rotations: np.ndarray = None
    rotation_offsets: np.ndarray = None


@dataclass
class DetectedMarker:
    code: list
    vr: np.ndarray
    model: MarkerModel = None
    offset: int = 0
    num_errors: int = 0
    num_discarded: int = 0
    filled_slots: int = 0

    def get_slot(self, index):
        return self.code[(index + self.offset) % len(self.code)]

    def invalidate_slot(self, index):
        self.code[(index + self.offset) % len(self.code)].invalidate()

    def num_filled_slots(self):
        return self.filled_slots if self.filled_slots else sum(1 for slot in self.code if slot.payload is not None)


def load_model(path):
    tokens = Path(path).read_text().split()
    pos = 0
    if tokens[pos] != "RUNE_direct":
        raise ValueError("invalid model header")
    pos += 1
    name = tokens[pos]
    pos += 1
    world_size = float(tokens[pos])
    pos += 1
    world_unit = tokens[pos]
    pos += 1
    num_slots = int(tokens[pos])
    pos += 1
    radius_ratio = float(tokens[pos])
    pos += 1
    num_layers = int(tokens[pos])
    pos += 1
    gap_factor = float(tokens[pos])
    pos += 1
    max_distance = int(tokens[pos])
    pos += 1
    idx = int(tokens[pos])
    pos += 1
    bcode = []
    ellipses = {}
    for slot_idx in range(num_slots):
        enabled = int(tokens[pos])
        pos += 1
        bcode.append(enabled == 1)
        if enabled:
            matrix = np.array([float(tokens[pos + i]) for i in range(9)], dtype=np.float64).reshape(3, 3)
            pos += 9
            matrix[1, 2] *= -1.0
            matrix[2, 1] *= -1.0
            ellipses[slot_idx] = matrix
    bcode_np = np.asarray(bcode, dtype=np.bool_)
    rotation_offsets = np.arange(0, len(bcode_np), num_layers, dtype=np.int32)
    bcode_rotations = np.asarray([np.roll(bcode_np, -rotation) for rotation in rotation_offsets], dtype=np.bool_)
    return MarkerModel(
        name,
        world_size,
        world_unit,
        num_slots,
        radius_ratio,
        num_layers,
        gap_factor,
        max_distance,
        idx,
        bcode,
        ellipses,
        bcode_np,
        bcode_rotations,
        rotation_offsets,
    )


class EllipseFitter:
    def __init__(self, e1, e2, intrinsics, coplanar_threshold=0.97, size_err_threshold=0.5):
        self.e1 = e1
        self.e2 = e2
        self.intrinsics = intrinsics
        self.coplanar_threshold = coplanar_threshold
        self.size_err_threshold = size_err_threshold
        self.vr = None
        self.qfit1c = None
        self.qfit2c = None
        self.rad_avg = None
        self.radius = None

    def choose_avg_vr(self):
        pairs = [
            (self.e1.vr1, self.e1.n1, self.e2.vr1, self.e2.n1),
            (self.e1.vr1, self.e1.n1, self.e2.vr2, self.e2.n2),
            (self.e1.vr2, self.e1.n2, self.e2.vr1, self.e2.n1),
            (self.e1.vr2, self.e1.n2, self.e2.vr2, self.e2.n2),
        ]
        candidates = [pair for pair in pairs if float(np.dot(pair[1], pair[3])) > self.coplanar_threshold]
        if not candidates:
            return None
        best = None
        best_err = float("inf")
        for vr1, _, vr2, _ in candidates:
            q1 = rotation_matrix_to_quaternion(vr1)
            q2 = rotation_matrix_to_quaternion(vr2)
            qavg = q1 + q2 if float(np.dot(q1, q2)) > 0.0 else q1 - q2
            qavg = qavg / np.linalg.norm(qavg)
            vr_avg = quaternion_to_rotation_matrix(qavg)
            q1c = transform_to_circle(self.e1.ellipse_norm, vr_avg)
            q2c = transform_to_circle(self.e2.ellipse_norm, vr_avg)
            r1 = circle_r_on_z0(q1c)
            r2 = circle_r_on_z0(q2c)
            err = abs(r1 - r2) / (r1 + r2)
            if err < best_err:
                best_err = err
                best = (vr_avg, r1, r2, q1c, q2c)
        if best is None or best_err > self.size_err_threshold:
            return None
        return best

    def fit_ellipse_avg(self, radius_ratio):
        best = self.choose_avg_vr()
        if best is None:
            return False
        self.vr, r1, r2, q1c, q2c = best
        center1 = ellipse_center(q1c)
        center2 = ellipse_center(q2c)
        rad_avg_sq = (r1 + r2) * 0.5
        radius_sq = rad_avg_sq * radius_ratio * radius_ratio
        qfit1c, qfit2c = fit_circles(center1, center2, radius_sq)
        if qfit1c is None:
            return False
        self.qfit1c = qfit1c
        self.qfit2c = qfit2c
        self.rad_avg = math.sqrt(rad_avg_sq)
        self.radius = math.sqrt(radius_sq)
        return True

    def fit_with_offset(self, which, rad_mul, k):
        focal = (self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2.0
        who = which.copy()
        rz0sq = (self.radius + self.rad_avg * rad_mul) ** 2
        who[2, 2] = -rz0sq + who[0, 2] * who[0, 2] + who[1, 2] * who[1, 2]
        fit = transform_to_ellipse(who, self.vr, k)
        fit[0, 2] *= focal
        fit[1, 2] *= focal
        fit[2, 0] *= focal
        fit[2, 1] *= focal
        fit[2, 2] *= focal * focal
        return fit

    def get_fit1_with_offset(self, rad_mul):
        return self.fit_with_offset(self.qfit1c, rad_mul, self.e1.l2inv)

    def get_fit2_with_offset(self, rad_mul):
        return self.fit_with_offset(self.qfit2c, rad_mul, self.e2.l2inv)


def rotation_matrix_to_quaternion(m):
    m00, m11, m22 = m[0, 0], m[1, 1], -m[2, 2]
    m01, m02, m10, m12, m20, m21 = m[0, 1], -m[0, 2], m[1, 0], -m[1, 2], m[2, 0], m[2, 1]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], dtype=np.float64)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], dtype=np.float64)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], dtype=np.float64)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], dtype=np.float64)


def quaternion_to_rotation_matrix(q):
    w, i, j, k = q
    return np.array(
        [
            [1.0 - 2.0 * j * j - 2.0 * k * k, 2.0 * i * j - 2.0 * k * w, -(2.0 * i * k + 2.0 * j * w)],
            [2.0 * i * j + 2.0 * k * w, 1.0 - 2.0 * i * i - 2.0 * k * k, -(2.0 * j * k - 2.0 * i * w)],
            [2.0 * i * k - 2.0 * j * w, 2.0 * j * k + 2.0 * i * w, -(1.0 - 2.0 * i * i - 2.0 * j * j)],
        ],
        dtype=np.float64,
    )


class SlotFitter:
    def __init__(self, vr, ellipses, fit_min, fit_max, intrinsics, min_support=6, min_symbols_for_layer=3):
        self.vr = vr
        self.ellipses = ellipses
        self.intrinsics = intrinsics
        self.min_symbols_for_layer = min_symbols_for_layer
        self.auto_inlier_centers = []
        self.points_coords = []
        self.refit = None
        for ellipse in ellipses:
            center = ellipse.center()
            self.points_coords.append(center)
            if not point_in_ellipse(fit_min, center) and point_in_ellipse(fit_max, center):
                self.auto_inlier_centers.append(circle_center_for_vr(ellipse.ellipse_norm, vr))
        self.points_coords = np.asarray(self.points_coords, dtype=np.float64)
        self.auto_inlier_centers = np.asarray(self.auto_inlier_centers, dtype=np.float64)
        self.valid = len(self.auto_inlier_centers) >= min_support

    def fit(self, radius_ratio, gap_factor, num_layers):
        return self._fit(self.auto_inlier_centers, radius_ratio, gap_factor, num_layers)

    def _fit(self, ring_centers, radius_ratio, gap_factor, num_layers):
        if len(ring_centers) == 0:
            return []
        xavg, yavg = np.mean(ring_centers, axis=0)
        u = ring_centers[:, 0] - xavg
        v = ring_centers[:, 1] - yavg
        suu = np.sum(u * u)
        suv = np.sum(u * v)
        svv = np.sum(v * v)
        svvv = np.sum(v * v * v)
        suuu = np.sum(u * u * u)
        suvv = np.sum(u * v * v)
        svuu = np.sum(v * u * u)
        a = np.array([[suu, suv], [suv, svv]], dtype=np.float64)
        b = np.array([0.5 * (suuu + suvv), 0.5 * (svvv + svuu)], dtype=np.float64)
        uc, vc = np.linalg.solve(a, b)
        x_center = uc + xavg
        y_center = vc + yavg
        rsqr = uc * uc + vc * vc + (suu + svv) / len(ring_centers)
        r = math.sqrt(rsqr)
        self.refit = np.array(
            [
                [1.0, 0.0, -x_center],
                [0.0, 1.0, -y_center],
                [-x_center, -y_center, x_center * x_center + y_center * y_center - rsqr],
            ],
            dtype=np.float64,
        )
        focal = (self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2.0
        ellysize = 1.0 / radius_ratio
        alpha = ellysize * 2.0 * gap_factor
        slots_per_layer = int(math.floor((2.0 * math.pi) / alpha))
        alpha = (2.0 * math.pi) / slots_per_layer
        center = np.array([x_center, y_center], dtype=np.float64)
        first_center = ring_centers[0]
        e1_angle = math.atan2(first_center[1] - center[1], first_center[0] - center[0])
        if e1_angle < 0.0:
            e1_angle += 2.0 * math.pi
        delta = ring_centers[1:] - center
        current_angles = np.arctan2(delta[:, 1], delta[:, 0])
        current_angles[current_angles < 0.0] += 2.0 * math.pi
        current_i = (current_angles - e1_angle) / alpha
        wrong = np.count_nonzero(np.abs(np.floor(current_i + 0.5) - current_i) > 0.20)
        if wrong > len(ring_centers) / 3.0:
            return []
        refit_center = ellipse_center(self.refit)
        slots_center = transform_point_from_circle_to_ellipse(refit_center, self.vr, focal)
        fit_slots = []
        fit_slot_centers = []
        for i in range(slots_per_layer):
            angle = e1_angle + alpha * i + alpha * 0.5
            current_slot = np.array([refit_center[0] + r * math.cos(angle) * 2.0, refit_center[1] + r * math.sin(angle) * 2.0], dtype=np.float64)
            fit_slots.append(transform_point_from_circle_to_ellipse(current_slot, self.vr, focal))
            level_centers = []
            for j in range(num_layers):
                factor = ((num_layers + j + 1) * 2.0) / (num_layers * 4.0)
                center_point = np.array(
                    [
                        refit_center[0] + r * factor * math.cos(angle - alpha * 0.5),
                        refit_center[1] + r * factor * math.sin(angle - alpha * 0.5),
                    ],
                    dtype=np.float64,
                )
                level_centers.append(transform_point_from_circle_to_ellipse(center_point, self.vr, focal))
            fit_slot_centers.append(level_centers)
        fit_slots = np.asarray(fit_slots, dtype=np.float64)
        markers = []
        for layer in range(num_layers):
            code, filled_slots = self.build_code_for_layer(layer, num_layers, rsqr, fit_slots, fit_slot_centers, slots_center)
            if code is not None:
                markers.append(DetectedMarker(code=code, vr=self.vr.copy(), filled_slots=filled_slots))
        return markers

    def get_fit_with_offset(self, current_layer, num_layers, radius_sq):
        focal = (self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2.0
        fit = self.refit.copy()
        factor = (((num_layers + current_layer + 1) * 2.0) + 1.0) / (num_layers * 4.0)
        rz0sq = radius_sq * factor * factor
        fit[2, 2] = -rz0sq + fit[0, 2] * fit[0, 2] + fit[1, 2] * fit[1, 2]
        fit_off = transform_to_ellipse(fit, self.vr, 134.46)
        fit_off[0, 2] *= focal
        fit_off[1, 2] *= focal
        fit_off[2, 0] *= focal
        fit_off[2, 1] *= focal
        fit_off[2, 2] *= focal * focal
        return fit_off

    def build_code_for_layer(self, layer, num_layers, radius_sq, fit_slots, fit_slot_centers, slots_center):
        slot_count = num_layers * len(fit_slots)
        values = np.zeros(slot_count, dtype=np.bool_)
        payloads = [None] * slot_count
        fit_levels = []
        current_layer = -layer
        for _ in range(num_layers + 1):
            fit_levels.append(self.get_fit_with_offset(current_layer - 1, num_layers, radius_sq))
            current_layer += 1
        candidate_mask = np.ones(len(self.ellipses), dtype=np.bool_)
        filled_slots = 0
        boundaries = fit_slots - slots_center
        next_boundaries = np.roll(boundaries, -1, axis=0)
        for level in range(len(fit_levels) - 1, 0, -1):
            qmax = fit_levels[level]
            qmin = fit_levels[level - 1]
            inside_qmax = points_in_ellipse(qmax, self.points_coords)
            inside_qmin = points_in_ellipse(qmin, self.points_coords)
            annulus_mask = candidate_mask & inside_qmax & (~inside_qmin)
            symbol_count = int(np.count_nonzero(annulus_mask))
            if symbol_count <= self.min_symbols_for_layer:
                return None, 0

            annulus_idx = np.flatnonzero(annulus_mask)
            annulus_points = self.points_coords[annulus_idx]
            vt = annulus_points - slots_center
            cross = boundaries[:, 0][None, :] * vt[:, 1][:, None] - boundaries[:, 1][None, :] * vt[:, 0][:, None]
            sector_mask = (cross <= 0.0) & (np.roll(cross, -1, axis=1) >= 0.0)
            valid_points = np.any(sector_mask, axis=1)
            if not np.any(valid_points):
                return None, 0

            point_slots = np.argmax(sector_mask, axis=1)
            for local_idx, is_valid in enumerate(valid_points):
                if not is_valid:
                    continue
                ellipse_idx = annulus_idx[local_idx]
                slot_idx = int(point_slots[local_idx])
                code_idx = slot_idx * num_layers + level - 1
                ellipse = self.ellipses[ellipse_idx]
                next_slot_idx = 0 if slot_idx + 1 == len(fit_slots) else slot_idx + 1
                slot_center = fit_slot_centers[next_slot_idx][level - 1]
                if ellipse.contains(slot_center):
                    if not values[code_idx]:
                        filled_slots += 1
                    values[code_idx] = True
                    payloads[code_idx] = ellipse

            candidate_mask &= inside_qmin

        code = [None] * slot_count
        for level in range(len(fit_levels) - 1, 0, -1):
            qmax = fit_levels[level]
            qmin = fit_levels[level - 1]
            for i in range(len(fit_slots)):
                next_slot_idx = 0 if i + 1 == len(fit_slots) else i + 1
                idx = i * num_layers + level - 1
                slot = Slot(qmin=qmin, qmax=qmax, c=slots_center, v1=boundaries[i], v2=next_boundaries[i], slot_center=fit_slot_centers[next_slot_idx][level - 1])
                slot.value = bool(values[idx])
                slot.payload = payloads[idx]
                code[idx] = slot

        return code, filled_slots


class MarkerDetector:
    def __init__(
        self,
        intrinsics,
        models,
        min_filled_slots=24,
        max_observed_errors=4,
        min_observed_match_ratio=0.85,
    ):
        self.intrinsics = intrinsics.astype(np.float64)
        self.models = {model.idx: model for model in models}
        self.min_pts_for_level = 4
        self.max_pair_checks = 192
        self.max_area_ratio = 1.8
        self.max_pair_neighbors = 6
        self.max_pair_distance_ratio = 12.0
        self.min_filled_slots = min_filled_slots
        self.max_observed_errors = max_observed_errors
        self.min_observed_match_ratio = min_observed_match_ratio

    def to_ellipse_points(self, ellipses):
        points = []
        focal = (self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2.0
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        for ellipse in ellipses:
            shifted = ((ellipse[0][0] - cx, ellipse[0][1] - cy), ellipse[1], ellipse[2])
            point = EllipsePoint(shifted)
            point.calc_vr(focal)
            points.append(point)
        return points

    def match_model(self, candidate):
        if not self.models:
            return None
        observed = np.fromiter((slot.value for slot in candidate.code), dtype=np.bool_, count=len(candidate.code))
        filled_slots = candidate.num_filled_slots()
        if filled_slots < self.min_filled_slots:
            return None
        best = None
        runner_up = None
        for model in self.models.values():
            if len(model.bcode_np) != len(candidate.code):
                continue
            expected = model.bcode_rotations
            unexpected_by_rotation = np.count_nonzero(observed[None, :] & (~expected), axis=1)
            matched_by_rotation = np.count_nonzero(observed[None, :] & expected, axis=1)
            missing_by_rotation = np.count_nonzero((~observed[None, :]) & expected, axis=1)
            best_rotation_idx = int(np.lexsort((missing_by_rotation, -matched_by_rotation, unexpected_by_rotation))[0])
            discarded = int(unexpected_by_rotation[best_rotation_idx])
            matched = int(matched_by_rotation[best_rotation_idx])
            missing = int(missing_by_rotation[best_rotation_idx])
            errors = discarded + missing
            rotation = int(model.rotation_offsets[best_rotation_idx])
            score = (discarded, -matched, missing)
            if best is None or score < best[0]:
                runner_up = best
                best = (score, model, rotation, errors, discarded)
            elif runner_up is None or score < runner_up[0]:
                runner_up = (score, model, rotation, errors, discarded)
        if best is None:
            return None
        _, model, rotation, errors, discarded = best
        matched = filled_slots - discarded
        if discarded > self.max_observed_errors:
            return None
        if matched / float(filled_slots) < self.min_observed_match_ratio:
            return None
        if runner_up is not None and best[0][0] == runner_up[0][0] and best[0][1] == runner_up[0][1]:
            return None
        if model.max_distance >= 0 and errors > model.max_distance:
            return None
        matched = DetectedMarker(code=list(candidate.code), vr=candidate.vr.copy(), model=model, offset=rotation, num_errors=errors, num_discarded=discarded, filled_slots=filled_slots)
        for i in range(len(matched.code)):
            observed = matched.get_slot(i).value
            expected = model.bcode[(i + rotation) % len(model.bcode)]
            if observed != expected:
                matched.invalidate_slot(i)
        return matched

    def mark_assigned(self, marker):
        for slot in marker.code:
            if slot.value and not slot.discarded and slot.payload is not None:
                slot.payload.unassigned = False

    def try_fit(self, fitter, markers_by_id):
        if not self.models:
            return None
        sample_model = next(iter(self.models.values()))
        possible_markers = fitter.fit(sample_model.radius_ratio, sample_model.gap_factor, sample_model.num_layers)
        for candidate in possible_markers:
            matched = self.match_model(candidate)
            if matched is None:
                continue
            current = markers_by_id.get(matched.model.idx)
            if current is None or matched.num_filled_slots() > current.num_filled_slots():
                markers_by_id[matched.model.idx] = matched
                return matched
            return current
        return None

    def candidate_pairs(self, centers, areas):
        count = len(centers)
        pair_i, pair_j = np.triu_indices(count, k=1)
        if len(pair_i) == 0:
            return pair_i, pair_j

        min_area = np.minimum(areas[pair_i], areas[pair_j])
        max_area = np.maximum(areas[pair_i], areas[pair_j])
        area_ratio = max_area / np.maximum(min_area, 1e-9)
        deltas = centers[pair_i] - centers[pair_j]
        dist2 = np.einsum("ij,ij->i", deltas, deltas)
        avg_diameter = np.sqrt(np.maximum((areas[pair_i] + areas[pair_j]) * 0.5, 1e-9))
        distance_ratio = np.sqrt(dist2) / avg_diameter
        valid = (area_ratio <= self.max_area_ratio) & (distance_ratio <= self.max_pair_distance_ratio)
        if not np.any(valid):
            return pair_i[:0], pair_j[:0]

        pair_i = pair_i[valid]
        pair_j = pair_j[valid]
        area_ratio = area_ratio[valid]
        distance_ratio = distance_ratio[valid]
        score = area_ratio + 0.05 * distance_ratio
        order = np.argsort(score)

        if self.max_pair_neighbors <= 0:
            order = order[: self.max_pair_checks]
            return pair_i[order], pair_j[order]

        neighbor_counts = np.zeros(count, dtype=np.int32)
        selected = []
        for idx in order:
            i = int(pair_i[idx])
            j = int(pair_j[idx])
            if neighbor_counts[i] >= self.max_pair_neighbors or neighbor_counts[j] >= self.max_pair_neighbors:
                continue
            neighbor_counts[i] += 1
            neighbor_counts[j] += 1
            selected.append(idx)
            if len(selected) >= self.max_pair_checks:
                break
        if not selected:
            return pair_i[:0], pair_j[:0]
        selected = np.asarray(selected, dtype=np.intp)
        return pair_i[selected], pair_j[selected]

    def detect_rotated_rects(self, ellipses):
        markers_by_id = {}
        ellipse_points = self.to_ellipse_points(ellipses)
        if not self.models:
            return []
        sample_model = next(iter(self.models.values()))
        count = len(ellipse_points)
        if count < 2:
            return []

        centers = np.asarray([ellipse.center() for ellipse in ellipse_points], dtype=np.float64)
        areas = np.asarray([ellipse.area for ellipse in ellipse_points], dtype=np.float64)
        pair_i, pair_j = self.candidate_pairs(centers, areas)
        if len(pair_i) == 0:
            return []

        for rank in range(len(pair_i)):
            e1 = ellipse_points[int(pair_i[rank])]
            e2 = ellipse_points[int(pair_j[rank])]
            if e1.is_assigned() or e2.is_assigned():
                continue
            fitter = EllipseFitter(e1, e2, self.intrinsics)
            if not fitter.fit_ellipse_avg(sample_model.radius_ratio):
                continue
            fit1_min = fitter.get_fit1_with_offset(-2.3)
            fit1_max = fitter.get_fit1_with_offset(2.3)
            sf1 = SlotFitter(fitter.vr, ellipse_points, fit1_min, fit1_max, self.intrinsics, self.min_pts_for_level)
            matched = self.try_fit(sf1, markers_by_id) if sf1.valid else None
            if matched is not None:
                self.mark_assigned(matched)
                if len(markers_by_id) == len(self.models):
                    break
                continue
            fit2_min = fitter.get_fit2_with_offset(-2.3)
            fit2_max = fitter.get_fit2_with_offset(2.3)
            sf2 = SlotFitter(fitter.vr, ellipse_points, fit2_min, fit2_max, self.intrinsics, self.min_pts_for_level)
            if sf2.valid:
                matched = self.try_fit(sf2, markers_by_id)
                if matched is not None:
                    self.mark_assigned(matched)
                    if len(markers_by_id) == len(self.models):
                        break
        return list(markers_by_id.values())

    def detect_image(self, image, ellipse_detector=None):
        detector = ellipse_detector or EllipseDetector()
        ellipses = detector.detect(image)
        return self.detect_rotated_rects(ellipses), ellipses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unable to read image")
    models = [load_model(path) for path in args.models]
    intrinsics = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    detector = MarkerDetector(intrinsics, models)
    markers, ellipses = detector.detect_image(image)
    print(f"ellipses={len(ellipses)}")
    print(f"markers={len(markers)}")
    for marker in markers:
        print(
            f"idx={marker.model.idx} "
            f"name={marker.model.name} "
            f"errors={marker.num_errors} "
            f"discarded={marker.num_discarded} "
            f"filled={marker.num_filled_slots()}"
        )


if __name__ == "__main__":
    main()
