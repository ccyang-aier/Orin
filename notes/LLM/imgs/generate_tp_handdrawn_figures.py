from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent

FONT_DIRS = [
    Path(r"C:\Windows\Fonts"),
    Path(r"C:\Users\17335\AppData\Local\Microsoft\Windows\Fonts"),
]


def find_font(*names: str) -> str:
    for name in names:
        for font_dir in FONT_DIRS:
            path = font_dir / name
            if path.exists():
                return str(path)
    return str(Path(r"C:\Windows\Fonts\msyh.ttc"))


TITLE_FONT = find_font("FZSTK.TTF", "LXGWBoldKai-HanOnly-Level2A.ttf", "simkai.ttf")
BODY_FONT = find_font("LXGWWenKai-Regular.ttf", "simkai.ttf", "msyh.ttc")
BODY_BOLD_FONT = find_font("LXGWWenKai-Regular.ttf", "msyhbd.ttc", "simhei.ttf")
MONO_FONT = find_font("consola.ttf", "msyh.ttc")


PAPER = (250, 249, 244)
INK = (20, 22, 22)
MUTED = (82, 83, 83)
BLUE = (24, 88, 211)
GREEN = (16, 132, 73)
ORANGE = (230, 91, 18)
PURPLE = (112, 66, 176)
GRAY = (126, 129, 130)
LIGHT = (255, 255, 251)


def font(size: int, kind: str = "body") -> ImageFont.FreeTypeFont:
    if kind == "title":
        return ImageFont.truetype(TITLE_FONT, size)
    if kind == "bold":
        return ImageFont.truetype(BODY_BOLD_FONT, size)
    if kind == "mono":
        return ImageFont.truetype(MONO_FONT, size)
    return ImageFont.truetype(BODY_FONT, size)


def new_canvas(w: int = 1600, h: int = 900) -> Image.Image:
    image = Image.new("RGB", (w, h), PAPER)
    draw = ImageDraw.Draw(image)
    rng = random.Random(13)
    for _ in range(int(w * h * 0.0012)):
        x = rng.randrange(w)
        y = rng.randrange(h)
        delta = rng.choice([-3, -2, 2, 3])
        base = PAPER[0] + delta
        draw.point((x, y), fill=(base, base, max(235, PAPER[2] + delta)))
    return image


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = INK,
    line_gap: int = 5,
) -> None:
    lines = text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total = sum(heights) + line_gap * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total) / 2
    for line, height in zip(lines, heights):
        width, _ = text_size(draw, line, fnt)
        draw.text((box[0] + (box[2] - box[0] - width) / 2, y), line, font=fnt, fill=fill)
        y += height + line_gap


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if current and text_size(draw, candidate, fnt)[0] > width:
            lines.append(current)
            current = ch
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def paragraph(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    outline: tuple[int, int, int] = INK,
    width: int = 3,
    radius: int = 18,
    fill: tuple[int, int, int] = LIGHT,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def section_title(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int], underline_w: int) -> None:
    draw.text((x, y), text, font=font(32, "bold"), fill=color)
    draw.line((x, y + 45, x + underline_w, y + 45), fill=color, width=3)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = INK,
    width: int = 3,
    head: int = 16,
) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
    right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill=color)


def matrix(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    rows: int,
    cols: int,
    cell: int = 34,
    outline: tuple[int, int, int] = INK,
    hatch: tuple[int, int, int] | None = None,
    width: int = 2,
    cell_hatches: dict[tuple[int, int], tuple[int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    cell_hatches = cell_hatches or {}
    for r in range(rows):
        for c in range(cols):
            cx = x + c * cell
            cy = y + r * cell
            color = cell_hatches.get((r, c), hatch)
            if color:
                # Hatching is deliberately clipped to each cell; this prevents spillover artifacts.
                draw.line((cx + 7, cy + cell - 7, cx + cell - 7, cy + 7), fill=color, width=1)
                if cell >= 30:
                    draw.line((cx + 14, cy + cell - 5, cx + cell - 5, cy + 14), fill=color, width=1)
    draw.rectangle((x, y, x + cols * cell, y + rows * cell), outline=outline, width=width)
    for c in range(1, cols):
        draw.line((x + c * cell, y, x + c * cell, y + rows * cell), fill=outline, width=1)
    for r in range(1, rows):
        draw.line((x, y + r * cell, x + cols * cell, y + r * cell), fill=outline, width=1)
    return x, y, x + cols * cell, y + rows * cell


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
    dash: int = 9,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    steps = int(dist / dash)
    for i in range(0, steps, 2):
        t1 = i / steps
        t2 = min((i + 1) / steps, 1)
        draw.line(
            (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1, x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2),
            fill=color,
            width=width,
        )


def note_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    fnt: ImageFont.FreeTypeFont | None = None,
) -> None:
    fnt = fnt or font(22)
    rounded(draw, box, outline=color, width=3, radius=12, fill=PAPER)
    lines = wrap_text(draw, text, fnt, box[2] - box[0] - 40)
    total = sum(text_size(draw, line, fnt)[1] for line in lines) + 6 * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total) / 2
    for line in lines:
        w, h = text_size(draw, line, fnt)
        draw.text((box[0] + (box[2] - box[0] - w) / 2, y), line, font=fnt, fill=color)
        y += h + 6


def title(draw: ImageDraw.ImageDraw, w: int, text: str, subtitle: str | None = None) -> None:
    tw, _ = text_size(draw, text, font(52, "title"))
    draw.text(((w - tw) / 2, 30), text, font=font(52, "title"), fill=INK)
    if subtitle:
        sw, _ = text_size(draw, subtitle, font(25))
        draw.text(((w - sw) / 2, 92), subtitle, font=font(25), fill=MUTED)


def save(image: Image.Image, name: str) -> None:
    image.save(OUT_DIR / name, optimize=True)


def draw_tp_mental_model() -> None:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    title(draw, 1600, "Tensor Parallelism：把一层拆给多张 GPU", "上一层 output -> 本层 input activations；同一层内部按 shard 并行")

    rounded(draw, (350, 145, 1250, 700), outline=INK, width=4, radius=26, fill=PAPER)
    center_text(draw, (350, 160, 1250, 215), "一个 Transformer layer 内的 TP group", font(30, "bold"), INK)

    xs = [410, 610, 810, 1010]
    colors = [BLUE, GREEN, ORANGE, PURPLE]
    for i, x in enumerate(xs):
        rounded(draw, (x, 270, x + 150, 555), outline=colors[i], width=4, radius=18, fill=LIGHT)
        center_text(draw, (x + 10, 292, x + 140, 335), f"GPU {i}", font(30, "bold"), colors[i])
        matrix(draw, x + 50, 365, 4, 2, 23, outline=colors[i], hatch=colors[i])
        center_text(draw, (x + 18, 475, x + 132, 510), "权重 shard", font(21), colors[i])
        center_text(draw, (x + 18, 518, x + 132, 548), "local GEMM", font(20), INK)
        arrow(draw, (x + 75, 610), (x + 75, 555), colors[i], width=3, head=13)
        arrow(draw, (x + 90, 555), (x + 90, 610), colors[i], width=3, head=13)

    rounded(draw, (440, 615, 1160, 670), outline=ORANGE, width=3, radius=12, fill=PAPER)
    center_text(draw, (455, 618, 1145, 667), "collective communication：all-reduce / all-gather，按布局发生", font(24), ORANGE)

    matrix(draw, 75, 410, 3, 4, 34, outline=INK)
    center_text(draw, (55, 530, 230, 605), "本层输入\nhidden states\n[S,H]", font(22), INK)
    arrow(draw, (235, 470), (345, 470), BLUE, width=4)
    matrix(draw, 1385, 410, 3, 4, 34, outline=INK)
    center_text(draw, (1360, 530, 1540, 610), "输出\nhidden states\n进入下一层", font(22), INK)
    arrow(draw, (1250, 470), (1375, 470), GREEN, width=4)

    note_box(
        draw,
        (150, 765, 1450, 840),
        "shard = 某个 rank 持有并计算的那一片权重、激活或输出；TP 主要切层内张量，不是简单切 batch。",
        PURPLE,
        font(24),
    )
    save(image, "tp-mental-model-cn.png")


def draw_column_row_parallel() -> None:
    image = new_canvas(1800, 1000)
    draw = ImageDraw.Draw(image)
    title(draw, 1800, "Column Parallel 与 Row Parallel：矩阵形状与合并语义", "统一例子：X ∈ R^(2×4)，W ∈ R^(4×4)，TP=2，最终 Y ∈ R^(2×4)")

    draw.line((900, 135, 900, 860), fill=INK, width=3)
    section_title(draw, 70, 145, "Column Parallel（按输出维切 W，沿列分片）", BLUE, 740)
    section_title(draw, 985, 145, "Row Parallel（按输入 / hidden 维切 X 与 W）", GREEN, 735)
    paragraph(draw, (75, 210), "完整输入 X 在每个 rank 可用；W 按列切分；各自计算输出的一段。", font(24), INK, 710)
    paragraph(draw, (985, 210), "X 与 W 沿 input / hidden 维对齐切分；各自计算同形状的部分贡献。", font(24), INK, 735)

    # Column parallel side.
    matrix(draw, 85, 315, 2, 4, 38, outline=INK)
    center_text(draw, (55, 405, 250, 465), "X (2×4)\n完整输入 replicated", font(22), BLUE)
    arrow(draw, (250, 355), (360, 355), INK, width=3)
    center_text(draw, (270, 305, 355, 348), "每个 GPU\n有完整 X", font(20), INK)

    for x, color, label in [(390, BLUE, "GPU 0"), (635, GREEN, "GPU 1")]:
        rounded(draw, (x, 250, x + 185, 475), outline=color, width=4, radius=18, fill=LIGHT)
        center_text(draw, (x + 20, 267, x + 165, 305), label, font(29, "bold"), color)
        center_text(draw, (x + 20, 310, x + 165, 338), "W0  (4×2)" if color == BLUE else "W1  (4×2)", font(21), INK)
        matrix(draw, x + 55, 345, 4, 2, 23, outline=color, hatch=color)
        center_text(draw, (x + 108, 370, x + 172, 430), "列分片", font(19), color)
        arrow(draw, (x + 92, 475), (x + 92, 555), color, width=3)
        center_text(draw, (x + 28, 585, x + 155, 622), "Y0 (2×2)" if color == BLUE else "Y1 (2×2)", font(22), color)
        matrix(draw, x + 65, 545, 2, 2, 27, outline=color)

    arrow(draw, (455, 617), (565, 705), ORANGE, width=3)
    arrow(draw, (725, 617), (615, 705), ORANGE, width=3)
    center_text(draw, (485, 635, 700, 700), "concat / all-gather\n沿列拼接", font(22), ORANGE)
    center_text(draw, (405, 715, 760, 750), "Y = [Y0 | Y1]  (2×4)", font(25), INK)
    cell_hatches = {(r, c): BLUE if c < 2 else GREEN for r in range(2) for c in range(4)}
    matrix(draw, 455, 755, 2, 4, 38, outline=INK, cell_hatches=cell_hatches)
    note_box(draw, (70, 865, 830, 935), "核心判断：Column Parallel 的局部结果是不同列，合并方式是拼接。", BLUE, font(23))

    # Row parallel side.
    matrix(draw, 1010, 320, 2, 4, 35, outline=INK)
    dashed_line(draw, (1080, 320), (1080, 390), ORANGE, width=2)
    center_text(draw, (1020, 397, 1160, 435), "X (2×4)", font(22), INK)
    center_text(draw, (935, 455, 1220, 505), "沿 hidden / input 维\n拆成 X0、X1", font(22), ORANGE)
    arrow(draw, (1075, 420), (1000, 540), ORANGE, width=3)
    arrow(draw, (1090, 420), (1170, 540), ORANGE, width=3)
    matrix(draw, 965, 540, 2, 2, 31, outline=ORANGE, hatch=ORANGE)
    center_text(draw, (935, 610, 1060, 650), "X0 (2×2)", font(20), ORANGE)
    matrix(draw, 1145, 540, 2, 2, 31, outline=ORANGE, hatch=ORANGE)
    center_text(draw, (1115, 610, 1240, 650), "X1 (2×2)", font(20), ORANGE)

    for x, color, label, matrix_label, x_label, z_label in [
        (1260, BLUE, "GPU 0", "W0 (2×4)", "X0 (2×2)", "Z0 (2×4)"),
        (1515, GREEN, "GPU 1", "W1 (2×4)", "X1 (2×2)", "Z1 (2×4)"),
    ]:
        rounded(draw, (x, 245, x + 195, 510), outline=color, width=4, radius=18, fill=LIGHT)
        center_text(draw, (x + 20, 265, x + 175, 302), label, font(29, "bold"), color)
        center_text(draw, (x + 20, 315, x + 175, 344), matrix_label, font(21), INK)
        matrix(draw, x + 44, 355, 2, 4, 24, outline=color, hatch=color)
        center_text(draw, (x + 35, 415, x + 165, 450), "行分片", font(19), color)
        center_text(draw, (x + 35, 450, x + 165, 477), x_label, font(19), INK)
        arrow(draw, (x + 98, 510), (x + 98, 600), color, width=3)
        center_text(draw, (x + 20, 652, x + 175, 682), z_label, font(21), color)
        matrix(draw, x + 36, 600, 2, 4, 25, outline=color, hatch=color)

    plus = (1480, 738)
    arrow(draw, (1358, 682), (plus[0] - 25, plus[1]), ORANGE, width=3)
    arrow(draw, (1605, 682), (plus[0] + 25, plus[1]), ORANGE, width=3)
    draw.ellipse((plus[0] - 18, plus[1] - 18, plus[0] + 18, plus[1] + 18), outline=ORANGE, width=3)
    center_text(draw, (plus[0] - 18, plus[1] - 18, plus[0] + 18, plus[1] + 18), "+", font(23, "bold"), ORANGE)
    center_text(draw, (1515, 710, 1730, 765), "all-reduce\n逐元素求和", font(21), ORANGE)
    arrow(draw, (plus[0], plus[1] + 18), (plus[0], 790), ORANGE, width=3)
    center_text(draw, (1320, 775, 1625, 812), "Y = Z0 + Z1  (2×4)", font(25), INK)
    matrix(draw, 1395, 825, 2, 4, 35, outline=INK)
    note_box(draw, (955, 865, 1735, 935), "核心判断：Row Parallel 的局部结果是同形状贡献，合并方式是逐元素求和。", GREEN, font(23))

    note_box(
        draw,
        (70, 945, 1735, 985),
        "心智模型：Column Parallel = 拼接（concat）；Row Parallel = 求和（all-reduce）。",
        PURPLE,
        font(23),
    )
    save(image, "tp-column-row-parallel-cn.png")


def draw_megatron_mlp_flow() -> None:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    title(draw, 1600, "Megatron-style MLP：先列并行，后行并行", "中间分片留在本地，逐元素激活在 shard 上完成，最后再 all-reduce")

    matrix(draw, 80, 405, 3, 4, 34, outline=INK)
    center_text(draw, (75, 530, 230, 595), "输入 X\nreplicated", font(22), INK)
    arrow(draw, (235, 465), (330, 465), BLUE, width=4)

    lanes = [(330, 200, BLUE, "GPU 0"), (330, 525, GREEN, "GPU 1")]
    for x, y, color, label in lanes:
        rounded(draw, (x, y, x + 265, y + 235), outline=color, width=4, radius=18, fill=LIGHT)
        draw.text((x + 22, y + 22), label, font=font(29, "bold"), fill=color)
        matrix(draw, x + 38, y + 78, 4, 2, 22, outline=color, hatch=color)
        arrow(draw, (x + 96, y + 122), (x + 168, y + 122), color, width=2)
        matrix(draw, x + 170, y + 98, 2, 2, 25, outline=color)
        center_text(draw, (x + 25, y + 166, x + 245, y + 222), "Column Parallel\nWup/Wgate -> H_i", font(19), color)

        rounded(draw, (710, y + 60, 945, y + 167), outline=color, width=3, radius=18, fill=LIGHT)
        center_text(draw, (725, y + 72, 930, y + 157), "GeLU / SwiGLU\n本地逐元素", font(25), color)
        arrow(draw, (595, y + 122), (710, y + 122), color, width=3)

        rounded(draw, (1030, y + 60, 1260, y + 167), outline=color, width=3, radius=18, fill=LIGHT)
        center_text(draw, (1045, y + 72, 1245, y + 132), "Down Projection\nRow Parallel", font(23), color)
        matrix(draw, x=1088, y=y + 134, rows=1, cols=4, cell=22, outline=color, hatch=color)
        center_text(draw, (1055, y + 167, 1235, y + 199), "partial [S,H]", font(19), color)
        arrow(draw, (945, y + 122), (1030, y + 122), color, width=3)

    rounded(draw, (1320, 392, 1460, 500), outline=ORANGE, width=4, radius=16, fill=LIGHT)
    center_text(draw, (1335, 402, 1445, 490), "All-Reduce\n求和", font(24), ORANGE)
    arrow(draw, (1260, 322), (1320, 445), BLUE, width=3)
    arrow(draw, (1260, 647), (1320, 455), GREEN, width=3)
    arrow(draw, (1460, 435), (1502, 435), ORANGE, width=3)
    matrix(draw, 1502, 395, 3, 3, 25, outline=INK)
    center_text(draw, (1485, 485, 1590, 530), "输出 Y", font(20), INK)

    note_box(
        draw,
        (220, 765, 1380, 835),
        "关键点：第一层扩大 hidden 后不立刻聚合；通信只在 Row Parallel 边界集中发生。",
        PURPLE,
        font(24),
    )
    save(image, "tp-megatron-mlp-flow-cn.png")


def draw_attention_head_sharding() -> None:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    title(draw, 1600, "Attention TP：QKV 按 head shard 留在本地", "例子：16 heads，TP=4；每个 rank 负责 4 个 heads")

    matrix(draw, 75, 390, 3, 4, 33, outline=INK)
    center_text(draw, (60, 510, 220, 570), "输入 hidden states\n[S,H]", font(22), INK)
    arrow(draw, (225, 445), (315, 445), BLUE, width=4)
    rounded(draw, (315, 310, 520, 575), outline=BLUE, width=4, radius=18, fill=LIGHT)
    center_text(draw, (330, 325, 505, 380), "QKV Projection\nColumn Parallel", font(23), BLUE)
    matrix(draw, 385, 405, 3, 3, 29, outline=BLUE, hatch=BLUE)
    center_text(draw, (335, 505, 500, 555), "本地产生\nQ/K/V head shard", font(20), BLUE)

    gpu_xs = [620, 795, 970, 1145]
    cols = [BLUE, GREEN, ORANGE, PURPLE]
    for i, (x, c) in enumerate(zip(gpu_xs, cols)):
        rounded(draw, (x, 250, x + 145, 610), outline=c, width=3, radius=18, fill=LIGHT)
        center_text(draw, (x + 10, 268, x + 135, 310), f"rank {i}", font(24, "bold"), c)
        center_text(draw, (x + 10, 320, x + 135, 350), f"heads {4*i}-{4*i+3}", font(19), INK)
        for j, y in enumerate([375, 443, 511]):
            matrix(draw, x + 28, y, 2, 4, 15, outline=c, hatch=c)
            center_text(draw, (x + 95, y - 1, x + 138, y + 32), ["Q", "K", "V"][j], font(18), c)
        center_text(draw, (x + 10, 570, x + 135, 600), "local attention", font(18), c)

    arrow(draw, (520, 445), (620, 445), BLUE, width=3)
    arrow(draw, (1290, 445), (1370, 445), GREEN, width=3)
    rounded(draw, (1370, 330, 1520, 560), outline=ORANGE, width=4, radius=18, fill=LIGHT)
    center_text(draw, (1385, 350, 1505, 418), "O Projection\nRow Parallel", font(23), ORANGE)
    matrix(draw, 1415, 445, 2, 4, 23, outline=ORANGE, hatch=ORANGE)
    center_text(draw, (1385, 505, 1505, 548), "all-reduce 后\n得到完整输出", font(19), ORANGE)

    note_box(
        draw,
        (165, 735, 1435, 820),
        "不要理解成先算完整 QKV 再分发；权重按 head/column 分片后，每个 rank 直接产生自己负责的 heads。",
        PURPLE,
        font(24),
    )
    save(image, "tp-attention-head-sharding-cn.png")


def draw_sharding_dimensions() -> None:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    title(draw, 1600, "TP 常见切分维度：切哪里，决定怎么合并", "同一套思想会落到 output、input、heads、vocab 等不同语义维度上")

    panels = [
        (70, 210, 390, 610, BLUE, "Column Parallel", "切 W 的输出列", "局部结果：不同列\n合并：concat"),
        (450, 210, 770, 610, GREEN, "Row Parallel", "切 X 与 W 的输入维", "局部结果：同形贡献\n合并：all-reduce"),
        (830, 210, 1150, 610, ORANGE, "Head Parallel", "切 attention heads", "每个 rank 负责一组 heads\nO projection 后合并"),
        (1210, 210, 1530, 610, PURPLE, "Vocab Parallel", "切 vocab range", "embedding / LM Head\n按 vocab range 分片"),
    ]
    for x1, y1, x2, y2, color, head, sub, foot in panels:
        rounded(draw, (x1, y1, x2, y2), outline=color, width=4, radius=18, fill=LIGHT)
        center_text(draw, (x1 + 20, y1 + 22, x2 - 20, y1 + 74), head, font(27, "bold"), color)
        center_text(draw, (x1 + 20, y1 + 78, x2 - 20, y1 + 118), sub, font(22), INK)
        if head == "Column Parallel":
            matrix(draw, x1 + 65, y1 + 155, 4, 4, 30, outline=INK)
            dashed_line(draw, (x1 + 125, y1 + 155), (x1 + 125, y1 + 275), color, width=3)
        elif head == "Row Parallel":
            matrix(draw, x1 + 75, y1 + 150, 4, 4, 30, outline=INK)
            dashed_line(draw, (x1 + 75, y1 + 210), (x1 + 195, y1 + 210), color, width=3)
        elif head == "Head Parallel":
            for i in range(4):
                hx = x1 + 55 + i * 52
                matrix(draw, hx, y1 + 170, 2, 2, 22, outline=color, hatch=color)
                center_text(draw, (hx, y1 + 222, hx + 45, y1 + 252), f"h{i}", font(16), color)
            draw.line((x1 + 60, y1 + 302, x2 - 60, y1 + 302), fill=color, width=3)
        else:
            for i in range(5):
                y = y1 + 155 + i * 28
                draw.rectangle((x1 + 85, y, x2 - 85, y + 22), outline=color, width=2)
                if i in [1, 2]:
                    draw.line((x1 + 95, y + 18, x2 - 95, y + 4), fill=color, width=1)
        center_text(draw, (x1 + 24, y2 - 88, x2 - 24, y2 - 24), foot, font(19), color)

    note_box(
        draw,
        (135, 730, 1465, 815),
        "判断规则：先问局部结果是不是完整输出的不同片段；如果是，拼接；如果是同一位置上的不同贡献，求和。",
        PURPLE,
        font(24),
    )
    save(image, "tp-sharding-dimensions-semantics-cn.png")


def draw_communication_cost() -> None:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    title(draw, 1600, "TP Communication Hotspots in One Transformer Block", "Megatron-style dense block：两处主要 all-reduce 边界，payload 通常按 [S,H] 估算")

    stages = [
        (65, 220, 245, 360, BLUE, "QKV\nColumn", "无立即通信"),
        (300, 220, 480, 360, GREEN, "Local\nAttention", "本地计算"),
        (535, 220, 715, 360, ORANGE, "O Projection\nRow Parallel", "All-Reduce\n[S,H]"),
        (770, 220, 950, 360, BLUE, "Gate/Up\nColumn", "无立即通信"),
        (1005, 220, 1185, 360, GREEN, "Activation\nGate", "本地逐元素"),
        (1240, 220, 1420, 360, ORANGE, "Down\nRow Parallel", "All-Reduce\n[S,H]"),
    ]
    for i, (x1, y1, x2, y2, color, head, sub) in enumerate(stages):
        rounded(draw, (x1, y1, x2, y2), outline=color, width=4, radius=18, fill=LIGHT)
        center_text(draw, (x1 + 10, y1 + 20, x2 - 10, y1 + 78), head, font(23, "bold"), color)
        center_text(draw, (x1 + 12, y1 + 90, x2 - 12, y2 - 16), sub, font(20), color)
        if i < len(stages) - 1:
            arrow(draw, (x2 + 12, 290), (stages[i + 1][0] - 12, 290), INK, width=3, head=15)
        if color == ORANGE:
            draw.ellipse((x1 + 72, y1 - 50, x1 + 112, y1 - 10), outline=ORANGE, width=3)
            center_text(draw, (x1 + 72, y1 - 50, x1 + 112, y1 - 10), "!", font(22, "bold"), ORANGE)
            arrow(draw, (x1 + 92, y1 - 10), (x1 + 92, y1), ORANGE, width=2, head=10)

    xs = [155, 390, 625, 860, 1095, 1330]
    labels = [("No Comm.", BLUE), ("Local Compute", GREEN), ("All-Reduce", ORANGE), ("No Comm.", BLUE), ("Local Compute", GREEN), ("All-Reduce", ORANGE)]
    draw.line((85, 470, 1395, 470), fill=INK, width=2)
    for x, (label, color) in zip(xs, labels):
        draw.ellipse((x - 11, 459, x + 11, 481), fill=color)
        center_text(draw, (x - 95, 500, x + 95, 545), label, font(23), color)

    rounded(draw, (100, 600, 660, 815), outline=BLUE, width=4, radius=18, fill=LIGHT)
    draw.text((135, 635), "Payload size", font=font(24, "bold"), fill=BLUE)
    center_text(draw, (150, 675, 610, 728), "M = S × H × b", font(39, "title"), INK)
    draw.text((145, 748), "S = batch × sequence tokens", font=font(19), fill=INK)
    draw.text((145, 778), "H = hidden size；b = bytes / element", font=font(19), fill=INK)

    rounded(draw, (760, 595, 1490, 805), outline=INK, width=4, radius=18, fill=LIGHT)
    draw.text((805, 630), "Approx. per-rank traffic", font=font(28, "bold"), fill=INK)
    rows = [
        ("all-reduce", "2(p−1)/p × M", "部分贡献 -> replicated 输出", ORANGE),
        ("reduce-scatter", "(p−1)/p × M", "规约后仍是 shard", BLUE),
        ("all-gather", "(p−1)/p × M", "shard -> 完整视图", GREEN),
    ]
    for idx, (name, formula, desc, color) in enumerate(rows):
        y = 675 + idx * 48
        draw.text((820, y), name, font=font(24, "bold"), fill=color)
        draw.text((1120, y), formula, font=font(23), fill=INK)
        draw.text((1300, y), desc, font=font(20), fill=INK)

    note_box(draw, (135, 835, 1465, 885), "拓扑判断：NVLink / NVSwitch 最适合 TP；PCIe 需要实测；跨节点 TP 通常要非常谨慎。", PURPLE, font(23))
    save(image, "tp-communication-cost-cn.png")


def draw_dp_tp_pp_decision_map() -> None:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    title(draw, 1600, "DP、TP、PP：并行策略的第一层选型地图", "先判断瓶颈，再选择 replica、shard 或 stage")

    cols = [
        (90, 230, 455, 660, GREEN, "DP", "replica", "模型能放进单卡\n请求或 batch 想扩吞吐", "复制完整模型\n不同副本处理不同数据"),
        (615, 230, 980, 660, BLUE, "TP", "shard", "单层权重或激活太大\n需要多卡合算同一层", "切层内张量\n依赖高速互联"),
        (1140, 230, 1505, 660, ORANGE, "PP", "stage", "层数很多或跨节点部署\n可接受流水气泡", "按层段切模型\n传递 activation"),
    ]
    for x1, y1, x2, y2, color, name, concept, when, how in cols:
        rounded(draw, (x1, y1, x2, y2), outline=color, width=4, radius=22, fill=LIGHT)
        center_text(draw, (x1 + 20, y1 + 25, x2 - 20, y1 + 85), name, font(42, "bold"), color)
        center_text(draw, (x1 + 20, y1 + 95, x2 - 20, y1 + 135), concept, font(28, "title"), INK)
        matrix(draw, x1 + 125, y1 + 165, 3, 4, 29, outline=color, hatch=color)
        draw.line((x1 + 55, y1 + 290, x2 - 55, y1 + 290), fill=color, width=2)
        center_text(draw, (x1 + 35, y1 + 315, x2 - 35, y1 + 405), when, font(22), INK)
        center_text(draw, (x1 + 35, y1 + 470, x2 - 35, y1 + 575), how, font(23), color)

    arrow(draw, (455, 445), (615, 445), GRAY, width=3)
    center_text(draw, (475, 392, 595, 430), "放不下单卡？", font(18), GRAY)
    arrow(draw, (980, 445), (1140, 445), GRAY, width=3)
    center_text(draw, (1005, 392, 1120, 430), "层段可切？", font(18), GRAY)

    note_box(
        draw,
        (155, 755, 1445, 835),
        "经验顺序：能复制就先用 DP 扩吞吐；单副本放不下或单层太大时考虑 TP；模型层段和集群边界明显时再组合 PP。",
        PURPLE,
        font(24),
    )
    save(image, "tp-dp-tp-pp-decision-map-cn.png")


def main() -> None:
    draw_tp_mental_model()
    draw_column_row_parallel()
    draw_megatron_mlp_flow()
    draw_attention_head_sharding()
    draw_sharding_dimensions()
    draw_communication_cost()
    draw_dp_tp_pp_decision_map()


if __name__ == "__main__":
    main()
