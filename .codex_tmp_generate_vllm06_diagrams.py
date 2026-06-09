from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


series_dir = next((Path("notes") / "vLLM").glob("vLLM*"))
out_dir = series_dir / "imgs"
out_dir.mkdir(parents=True, exist_ok=True)

W, H = 1600, 900
S = 3
BG = (250, 250, 247)
INK = (24, 33, 48)
GREEN = (18, 112, 62)
BLUE = (18, 83, 170)
RED = (196, 42, 42)
ORANGE = (220, 118, 28)
PURPLE = (104, 54, 170)
GRAY = (105, 112, 120)
LIGHT_GREEN = (234, 246, 237)
LIGHT_BLUE = (234, 242, 255)
LIGHT_RED = (255, 238, 238)
LIGHT_ORANGE = (255, 246, 230)
LIGHT_PURPLE = (246, 240, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"]
        if bold
        else [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    )
    candidates.extend([r"C:\Windows\Fonts\arial.ttf"])
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size * S)
    return ImageFont.load_default()


F_TITLE = font(38, True)
F_SUB = font(24)
F_LABEL = font(25, True)
F_BODY = font(22)
F_SMALL = font(18)
F_TINY = font(16)


def sc(value: float) -> int:
    return int(round(value * S))


def make_canvas() -> Image.Image:
    return Image.new("RGB", (W * S, H * S), BG)


def draw_text(draw, xy, text, fill=INK, f=F_BODY, anchor=None, align="center", spacing=6):
    draw.multiline_text(
        (sc(xy[0]), sc(xy[1])),
        text,
        fill=fill,
        font=f,
        anchor=anchor,
        align=align,
        spacing=sc(spacing),
    )


def rounded(draw, box, outline=INK, fill=None, width=4, radius=18, dash=False):
    if dash:
        x1, y1, x2, y2 = box
        r = radius
        seg = 18
        gap = 10

        def dashed_line(p1, p2, color, line_width):
            sx, sy = p1
            ex, ey = p2
            length = math.hypot(ex - sx, ey - sy)
            if length == 0:
                return
            dx = (ex - sx) / length
            dy = (ey - sy) / length
            t = 0
            while t < length:
                t2 = min(t + seg, length)
                draw.line(
                    (
                        sc(sx + dx * t),
                        sc(sy + dy * t),
                        sc(sx + dx * t2),
                        sc(sy + dy * t2),
                    ),
                    fill=color,
                    width=sc(line_width),
                )
                t += seg + gap

        dashed_line((x1 + r, y1), (x2 - r, y1), outline, width)
        dashed_line((x2, y1 + r), (x2, y2 - r), outline, width)
        dashed_line((x2 - r, y2), (x1 + r, y2), outline, width)
        dashed_line((x1, y2 - r), (x1, y1 + r), outline, width)
        return
    draw.rounded_rectangle(
        tuple(sc(x) for x in box), radius=sc(radius), fill=fill, outline=outline, width=sc(width)
    )


def arrow(draw, start, end, fill=INK, width=5, head=16):
    x1, y1 = start
    x2, y2 = end
    draw.line((sc(x1), sc(y1), sc(x2), sc(y2)), fill=fill, width=sc(width))
    ang = math.atan2(y2 - y1, x2 - x1)
    for a in (ang + math.pi * 0.82, ang - math.pi * 0.82):
        x = x2 + math.cos(a) * head
        y = y2 + math.sin(a) * head
        draw.line((sc(x2), sc(y2), sc(x), sc(y)), fill=fill, width=sc(width))


def resize_save(img: Image.Image, name: str):
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    img.save(out_dir / name, optimize=True)


img = make_canvas()
d = ImageDraw.Draw(img)
draw_text(
    d,
    (70, 48),
    "Iteration-level scheduling：每个 token iteration 重新组批",
    GREEN,
    F_TITLE,
    anchor="la",
    align="left",
)
draw_text(
    d,
    (72, 105),
    "请求不再等到整段序列结束才释放 batch 位置，完成、继续、加入都发生在相邻 iteration 边界。",
    GRAY,
    F_SUB,
    anchor="la",
    align="left",
)

requests = [
    (
        "Req A",
        "prompt 已完成，decode 继续",
        BLUE,
        130,
        [
            (180, 210, 330, 260, "decode"),
            (410, 210, 560, 260, "decode"),
            (640, 210, 790, 260, "done"),
        ],
    ),
    (
        "Req B",
        "长 prompt 分块进入",
        GREEN,
        255,
        [
            (180, 335, 330, 385, "prefill"),
            (410, 335, 560, 385, "decode"),
            (640, 335, 790, 385, "decode"),
            (870, 335, 1020, 385, "decode"),
        ],
    ),
    (
        "Req C",
        "中途到达，下一拍加入",
        ORANGE,
        380,
        [
            (410, 460, 560, 510, "waiting"),
            (640, 460, 790, 510, "prefill"),
            (870, 460, 1020, 510, "decode"),
        ],
    ),
]
for name, note, color, y, blocks in requests:
    rounded(d, (70, y + 66, 168, y + 130), outline=color, fill=(255, 255, 255), width=4, radius=14)
    draw_text(d, (119, y + 86), name, color, F_LABEL, anchor="ma")
    draw_text(d, (74, y + 137), note, GRAY, F_TINY, anchor="la", align="left")
    d.line((sc(175), sc(y + 98), sc(1100), sc(y + 98)), fill=(205, 210, 215), width=sc(3))
    for x1, y1, x2, y2, label in blocks:
        fill = LIGHT_BLUE if color == BLUE else LIGHT_GREEN if color == GREEN else LIGHT_ORANGE
        if label == "waiting":
            fill = (246, 246, 246)
        rounded(
            d,
            (x1, y1, x2, y2),
            outline=color if label != "waiting" else GRAY,
            fill=fill,
            width=4,
            radius=14,
            dash=(label == "waiting"),
        )
        draw_text(d, ((x1 + x2) / 2, (y1 + y2) / 2), label, color if label != "waiting" else GRAY, F_SMALL, anchor="mm")

xs = [255, 485, 715, 945]
for i, x in enumerate(xs, 1):
    d.line((sc(x), sc(170), sc(x), sc(590)), fill=(170, 170, 170), width=sc(2))
    draw_text(d, (x, 160), f"iteration {i}", INK, F_BODY, anchor="mm")

for i, (x, members) in enumerate([(255, "A + B"), (485, "A + B + C"), (715, "B + C"), (945, "B + C")], 1):
    rounded(d, (x - 95, 625, x + 95, 700), outline=PURPLE, fill=LIGHT_PURPLE, width=4, radius=18)
    draw_text(d, (x, 650), f"Batch {i}", PURPLE, F_LABEL, anchor="mm")
    draw_text(d, (x, 681), members, INK, F_SMALL, anchor="mm")
    if i < 4:
        arrow(d, (x + 105, 663), (x + 210, 663), fill=PURPLE, width=4, head=13)

draw_text(d, (1195, 230), "核心变化", GREEN, F_LABEL, anchor="la", align="left")
notes = [
    "按 iteration 决策\n不是按完整 request 固定 batch",
    "完成请求立刻离开\n等待请求可在下一拍加入",
    "decode、小块 prefill、prefix 命中\n进入同一调度视野",
]
for j, note in enumerate(notes):
    yy = 285 + j * 90
    rounded(d, (1160, yy - 27, 1535, yy + 58), outline=GREEN, fill=(255, 255, 255), width=3, radius=14)
    draw_text(d, (1180, yy + 1), note, INK, F_TINY, anchor="la", align="left")

draw_text(
    d,
    (70, 815),
    "读图要点：Orca 的贡献不是某个固定队列，而是把在线生成服务的 batch 边界推进到每个 decode iteration。",
    GRAY,
    F_BODY,
    anchor="la",
    align="left",
)
resize_save(img, "06_iteration_level_scheduling.png")

img = make_canvas()
d = ImageDraw.Draw(img)
draw_text(
    d,
    (70, 48),
    "Operation-level pipeline：把一次生成拆成可重叠的资源流",
    GREEN,
    F_TITLE,
    anchor="la",
    align="left",
)
draw_text(
    d,
    (72, 105),
    "NanoFlow 的思想是让 compute、memory、network 与 CPU scheduling 不再完全串行，而是拆成更小 nano-batch 后错开执行。",
    GRAY,
    F_SUB,
    anchor="la",
    align="left",
)

rounded(d, (70, 165, 735, 385), outline=RED, fill=(255, 255, 255), width=4, radius=20)
draw_text(d, (95, 195), "传统串行执行", RED, F_LABEL, anchor="la", align="left")
serial = [
    ("schedule", GREEN),
    ("prep", BLUE),
    ("compute", RED),
    ("memory", ORANGE),
    ("copy/net", PURPLE),
]
x = 105
for idx, (label, color) in enumerate(serial):
    rounded(d, (x, 260, x + 98, 320), outline=color, fill=(255, 255, 255), width=3, radius=12)
    draw_text(d, (x + 49, 290), label, color, F_TINY, anchor="mm")
    if idx != len(serial) - 1:
        arrow(d, (x + 105, 290), (x + 135, 290), fill=GRAY, width=3, head=10)
    x += 137
draw_text(d, (105, 345), "资源一个接一个等待，任一环节偏慢都会把空泡传给下一环节。", GRAY, F_SMALL, anchor="la", align="left")

rounded(d, (70, 430, 1530, 760), outline=GREEN, fill=(255, 255, 255), width=4, radius=20)
draw_text(d, (95, 460), "重叠后的 operation-level 流水", GREEN, F_LABEL, anchor="la", align="left")
lanes = [("nano-batch 0", 520), ("nano-batch 1", 600), ("nano-batch 2", 680)]
ops = [
    ("schedule", GREEN, 0, 120),
    ("prep", BLUE, 100, 140),
    ("compute", RED, 220, 260),
    ("memory", ORANGE, 430, 150),
    ("copy/net", PURPLE, 560, 140),
]
for label, y in lanes:
    draw_text(d, (105, y + 24), label, INK, F_SMALL, anchor="la")
    d.line((sc(230), sc(y + 24), sc(1450), sc(y + 24)), fill=(210, 215, 220), width=sc(2))
    offset = {"nano-batch 0": 0, "nano-batch 1": 85, "nano-batch 2": 170}[label]
    for op, color, start, w in ops:
        fill = LIGHT_GREEN if color == GREEN else LIGHT_BLUE if color == BLUE else LIGHT_RED if color == RED else LIGHT_ORANGE if color == ORANGE else LIGHT_PURPLE
        x1 = 250 + start + offset
        x2 = x1 + w
        rounded(d, (x1, y, x2, y + 48), outline=color, fill=fill, width=3, radius=12)
        draw_text(d, ((x1 + x2) / 2, y + 24), op, color, F_TINY, anchor="mm")

for x, label, color in [
    (600, "prep overlap", BLUE),
    (840, "memory overlap", ORANGE),
    (1035, "copy overlap", PURPLE),
]:
    d.line((sc(x), sc(500), sc(x), sc(725)), fill=color, width=sc(3))
    draw_text(d, (x + 12, 733), label, color, F_TINY, anchor="la")

rounded(d, (795, 165, 1530, 385), outline=BLUE, fill=LIGHT_BLUE, width=4, radius=20)
draw_text(d, (825, 195), "vLLM async scheduling 借走的思想", BLUE, F_LABEL, anchor="la", align="left")
bridge = [
    "目标相同：减少 CPU/GPU 串行边界上的等待",
    "层级不同：vLLM 主要重叠 schedule/input prep 与 GPU forward",
    "账本不同：vLLM 依赖 output placeholder 与真实 token 回来后的校正",
]
for j, item in enumerate(bridge):
    yy = 250 + j * 45
    d.ellipse((sc(830), sc(yy - 7), sc(844), sc(yy + 7)), fill=BLUE)
    draw_text(d, (860, yy), item, INK, F_SMALL, anchor="la", align="left")

draw_text(
    d,
    (70, 820),
    "读图要点：NanoFlow 是 operation-level pipeline，不等同于 vLLM 的 AsyncScheduler；它提供的是资源重叠的设计参照。",
    GRAY,
    F_BODY,
    anchor="la",
    align="left",
)
resize_save(img, "06_operation_level_pipeline.png")
