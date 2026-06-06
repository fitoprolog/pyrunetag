import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RADIUS_RATIO = 17.8
GAP_FACTOR = 1.3
NUM_LAYERS = 3
ELLIPSE_SIZE = 1.0 / RADIUS_RATIO
CODE_LENGTH = 43
NUM_WORDS = 117649
ALIGN_K = 4
ALIGN_P = 173
ALIGN_ROOT = 2
ALIGN_LOGN1 = 89
GENERATOR = [1, 1, 6, 4, 6, 0, 3, 1, 5, 3, 5, 4, 0, 4, 6, 3, 4, 6, 3, 6, 4, 3, 6, 4, 0, 4, 5, 3, 5, 1, 3, 0, 6, 4, 6, 1, 1, 0, 0, 0, 0, 0, 0]


def unpack_code(code):
    bits = []
    for value in code:
        current = value + 1
        if current < 1 or current > 7:
            raise ValueError("invalid code symbol")
        bits.append((current // 4) % 2 == 1)
        bits.append((current // 2) % 2 == 1)
        bits.append(current % 2 == 1)
    return bits


def build_align_tables():
    align_pow = [0] * ALIGN_P
    align_log = [0] * ALIGN_P
    current = 1
    for i in range(ALIGN_P - 1):
        align_pow[i] = current
        align_log[current] = i
        current = (current * ALIGN_ROOT) % ALIGN_P
    return align_pow, align_log


ALIGN_POW, ALIGN_LOG = build_align_tables()


def get_index(code):
    index = 0
    index += (5 * code[0] + 2 * code[3] + 6 * code[4] + code[5]) % 7
    index *= 7
    index += (2 * code[2] + 6 * code[3] + code[4]) % 7
    index *= 7
    index += (2 * code[1] + 6 * code[2] + code[3]) % 7
    index *= 7
    index += (2 * code[0] + 6 * code[1] + code[2]) % 7
    index *= 7
    index += (6 * code[0] + code[1]) % 7
    index *= 7
    index += code[0]
    return index


def sft(code):
    out = [0] * CODE_LENGTH
    for i in range(CODE_LENGTH):
        strobe = (ALIGN_K * i) % (ALIGN_P - 1)
        total = 0
        for j in range(CODE_LENGTH):
            total += ALIGN_POW[(strobe * j) % (ALIGN_P - 1)] * code[j]
            total %= ALIGN_P
        out[i] = total
    return out


def isft(code):
    out = [0] * CODE_LENGTH
    for i in range(CODE_LENGTH):
        strobe = (ALIGN_K * i) % (ALIGN_P - 1)
        total = 0
        for j in range(CODE_LENGTH):
            psn = (strobe * j) % (ALIGN_P - 1)
            total += ALIGN_POW[(ALIGN_LOGN1 + ALIGN_P - 2 - psn) % (ALIGN_P - 1)] * code[j]
            total %= ALIGN_P
        out[i] = total
    return out


def align_code(code):
    ft = sft(code)
    if ft[1] == 0:
        raise ValueError("periodic code")
    rotation = ALIGN_LOG[ft[1]] // ALIGN_K
    rot_idx = ALIGN_P - 1 - ALIGN_K * rotation
    for i in range(1, CODE_LENGTH):
        ft[i] = (ft[i] * ALIGN_POW[(rot_idx * i) % (ALIGN_P - 1)]) % ALIGN_P
    aligned = isft(ft)
    index = get_index(aligned)
    return aligned, index, rotation


def generate_codeword(seed_index):
    index = seed_index % NUM_WORDS
    code = [0] * CODE_LENGTH
    start = 0
    while index:
        value = index % 7
        for i in range(CODE_LENGTH):
            code[(start + i) % CODE_LENGTH] += (value * GENERATOR[i]) % 7
        index //= 7
        start += 1
    for i in range(CODE_LENGTH):
        code[i] %= 7
    aligned, aligned_index, rotation = align_code(code)
    return aligned, aligned_index, rotation


def generate_codes_file(output_path, target_count=17000):
    generated = {}
    seed = 0
    while len(generated) < target_count:
        try:
            code, idx, rotation = generate_codeword(seed)
        except ValueError:
            seed += 1
            continue
        if rotation == 0 and idx not in generated:
            generated[idx] = code
        seed += 1
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(generated)}\n")
        for idx in sorted(generated):
            code = generated[idx]
            handle.write(f"{idx} {len(code)} {' '.join(str(v) for v in code)}\n")


@dataclass
class TagCode:
    idx: int
    code: list
    bcode: list


@dataclass
class MarkPoint:
    x: float
    y: float
    angle: float
    size: float
    layer: int
    enabled: bool = False


def load_tags(path):
    tokens = Path(path).read_text().split()
    if not tokens:
        raise ValueError("empty tag file")
    pos = 0
    tag_count = int(tokens[pos])
    pos += 1
    tags = []
    for _ in range(tag_count):
        idx = int(tokens[pos])
        pos += 1
        code_len = int(tokens[pos])
        pos += 1
        code = [int(tokens[pos + i]) for i in range(code_len)]
        pos += code_len
        tags.append(TagCode(idx=idx, code=code, bcode=unpack_code(code)))
    return tags


def build_mark_points():
    alpha = ELLIPSE_SIZE * 2.0 * GAP_FACTOR
    slots_per_layer = int(math.floor((2.0 * math.pi) / alpha))
    alpha = (2.0 * math.pi) / slots_per_layer
    points = []
    for i in range(slots_per_layer):
        for layer in range(NUM_LAYERS):
            angle = alpha * i
            radius = (NUM_LAYERS + layer + 1) / (NUM_LAYERS * 2.0)
            points.append(
                MarkPoint(
                    x=radius * math.cos(angle),
                    y=radius * math.sin(angle),
                    angle=angle,
                    size=ELLIPSE_SIZE * radius,
                    layer=layer,
                )
            )
    return points, slots_per_layer


class RUNETagGenerator:
    def __init__(self, tags):
        self.tags = tags
        self.points, self.slots_per_layer = build_mark_points()
        self.num_slots = len(self.points)

    def build_enabled_points(self, tag_index):
        tag = self.tags[tag_index]
        if len(tag.bcode) != self.num_slots:
            raise ValueError(f"tag {tag.idx} has {len(tag.bcode)} bits, expected {self.num_slots}")
        points = [MarkPoint(**vars(point)) for point in self.points]
        for i, enabled in enumerate(tag.bcode):
            points[i].enabled = bool(enabled)
        return points

    def render(self, tag_index, marker_size_mm=200.0, pixels_per_mm=12.0, margin_mm=10.0):
        import cv2

        points = self.build_enabled_points(tag_index)
        outer_radius_mm = marker_size_mm / 2.0
        canvas_radius_mm = outer_radius_mm * (1.0 + ELLIPSE_SIZE) + margin_mm
        image_size = int(math.ceil(canvas_radius_mm * 2.0 * pixels_per_mm))
        image = np.full((image_size, image_size), 255, dtype=np.uint8)
        center = image_size / 2.0
        for point in points:
            if not point.enabled:
                continue
            radius_mm = point.size * outer_radius_mm
            cx = center + point.x * outer_radius_mm * pixels_per_mm
            cy = center - point.y * outer_radius_mm * pixels_per_mm
            rr = max(1, int(round(radius_mm * pixels_per_mm)))
            cv2.circle(image, (int(round(cx)), int(round(cy))), rr, 0, -1, lineType=cv2.LINE_AA)
        return image

    def export_descriptor(self, tag_index, output_path, marker_name, marker_size_mm=200.0):
        points = self.build_enabled_points(tag_index)
        tag = self.tags[tag_index]
        outer_radius_mm = marker_size_mm / 2.0
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("RUNE_direct\n")
            handle.write(f"{marker_name}\n")
            handle.write(f"{outer_radius_mm}\n")
            handle.write("mm\n")
            handle.write(f"{len(points)}\n")
            handle.write(f"{RADIUS_RATIO}\n")
            handle.write(f"{NUM_LAYERS}\n")
            handle.write(f"{GAP_FACTOR}\n")
            handle.write("-1\n")
            handle.write(f"{tag.idx}\n")
            for point in points:
                if not point.enabled:
                    handle.write("0\n")
                    continue
                radius = point.size * outer_radius_mm
                cx = point.x * outer_radius_mm
                cy = point.y * outer_radius_mm
                k = cx * cx + cy * cy - radius * radius
                handle.write(
                    "1 "
                    f"1.0 0.0 {-cx} "
                    f"0.0 1.0 {-cy} "
                    f"{-cx} {-cy} {k}\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codes", nargs="?")
    parser.add_argument("--tag-index", type=int, default=0)
    parser.add_argument("--marker-size-mm", type=float, default=200.0)
    parser.add_argument("--pixels-per-mm", type=float, default=12.0)
    parser.add_argument("--name", default="RUNETag")
    parser.add_argument("--png")
    parser.add_argument("--descriptor")
    parser.add_argument("--generate-codes")
    parser.add_argument("--generate-count", type=int, default=17000)
    args = parser.parse_args()

    if args.generate_codes:
        generate_codes_file(args.generate_codes, args.generate_count)
        return

    if not args.codes:
        raise ValueError("codes path is required unless --generate-codes is used")

    tags = load_tags(args.codes)
    generator = RUNETagGenerator(tags)
    if args.tag_index < 0 or args.tag_index >= len(tags):
        raise ValueError("tag index out of range")

    image = generator.render(
        tag_index=args.tag_index,
        marker_size_mm=args.marker_size_mm,
        pixels_per_mm=args.pixels_per_mm,
    )
    if args.png:
        import cv2

        cv2.imwrite(args.png, image)
    if args.descriptor:
        generator.export_descriptor(
            tag_index=args.tag_index,
            output_path=args.descriptor,
            marker_name=args.name,
            marker_size_mm=args.marker_size_mm,
        )


if __name__ == "__main__":
    main()
