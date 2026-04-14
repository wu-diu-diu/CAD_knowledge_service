from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


ARTICLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ARTICLE_DIR / "results"

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
)


@dataclass(frozen=True)
class LegendItem:
    kind: str
    color: str
    label: str | None = None


@dataclass(frozen=True)
class LegendJob:
    image_path: Path
    output_path: Path
    legend_items: Sequence[LegendItem]
    reserved_chars_per_gap: int
    title: str | None = None


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _draw_color_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[tuple[int, int], tuple[int, int]],
    fill: str,
    outline: str | tuple[int, int, int],
) -> None:
    draw.rectangle(xy, fill=fill, outline=outline)


def _draw_line_sample(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str | tuple[int, int, int],
    width: int,
) -> None:
    draw.line((start, end), fill=fill, width=width)


def add_color_legend(
    image_path: str | Path,
    output_path: str | Path,
    legend_items: Sequence[LegendItem],
    title: str | None = "图例",
    reserved_chars_per_gap: int = 0,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)

    src_image = Image.open(image_path).convert("RGBA")
    white_bg = Image.new("RGBA", src_image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white_bg, src_image).convert("RGB")
    font = _load_font(size=max(26, image.width // 70))
    title_font = _load_font(size=max(30, image.width // 56))

    tmp_draw = ImageDraw.Draw(image)
    box_size = max(34, image.width // 42)
    line_length = max(box_size * 2, image.width // 24)
    line_width = max(6, box_size // 5)
    section_gap = max(60, image.width // 35)
    legend_padding_x = max(48, image.width // 40)
    legend_padding_y = max(28, image.height // 45)

    title_w = 0
    title_h = 0
    if title:
        title_w, title_h = _text_size(tmp_draw, title, title_font)

    item_widths: list[int] = []
    item_height = box_size
    text_gap = max(16, box_size // 3)
    reserved_gap_width = 0
    if reserved_chars_per_gap > 0:
        reserved_gap_width = _text_size(tmp_draw, "汉" * reserved_chars_per_gap, font)[0] + text_gap * 2
        section_gap = max(section_gap, reserved_gap_width)

    for item in legend_items:
        symbol_width = line_length if item.kind == "line" else box_size
        symbol_height = line_width if item.kind == "line" else box_size
        if item.label:
            text_w, text_h = _text_size(tmp_draw, item.label, font)
            item_widths.append(symbol_width + text_gap + text_w)
            item_height = max(item_height, symbol_height, text_h)
        else:
            item_widths.append(symbol_width)
            item_height = max(item_height, symbol_height)

    legend_row_width = sum(item_widths) + section_gap * max(0, len(item_widths) - 1)
    legend_height = legend_padding_y * 2 + max(box_size, item_height)
    if title:
        legend_height += legend_padding_y + title_h

    canvas = Image.new("RGB", (image.width, image.height + legend_height), "white")
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line(
        [(0, image.height), (image.width, image.height)],
        fill=(215, 215, 215),
        width=max(2, image.width // 900),
    )

    row_y = image.height + legend_padding_y
    if title:
        title_x = (image.width - title_w) // 2
        title_y = image.height + legend_padding_y
        draw.text((title_x, title_y), title, fill="black", font=title_font)
        row_y = image.height + legend_padding_y * 2 + title_h

    current_x = max(legend_padding_x, (image.width - legend_row_width) // 2)
    for idx, item in enumerate(legend_items):
        if item.kind == "line":
            line_y = row_y + item_height // 2
            _draw_line_sample(
                draw,
                (current_x, line_y),
                (current_x + line_length, line_y),
                fill=item.color,
                width=line_width,
            )
            symbol_width = line_length
        else:
            top = row_y + max(0, (item_height - box_size) // 2)
            bottom = top + box_size
            _draw_color_box(
                draw,
                ((current_x, top), (current_x + box_size, bottom)),
                fill=item.color,
                outline=item.color,
            )
            symbol_width = box_size

        if item.label:
            text_x = current_x + symbol_width + text_gap
            text_h = _text_size(draw, item.label, font)[1]
            text_y = row_y + max(0, (item_height - text_h) // 2)
            draw.text((text_x, text_y), item.label, fill="black", font=font)
        current_x += item_widths[idx] + section_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    base_legend_items = [
        LegendItem(kind="box", color="#0FE608"),
        LegendItem(kind="box", color="#FF1010"),
        LegendItem(kind="box", color="#1010FF"),
        LegendItem(kind="box", color="#FFF200"),
    ]
    jobs = [
        LegendJob(
            image_path=ARTICLE_DIR / "aaa_房间离散化示意.png",
            output_path=RESULTS_DIR / "aaa_房间离散化示意_legend.png",
            legend_items=base_legend_items,
            title=None,
            reserved_chars_per_gap=15,
        ),
        LegendJob(
            image_path=ARTICLE_DIR / "aaa_不规则房间.png",
            output_path=RESULTS_DIR / "aaa_不规则房间_legend.png",
            legend_items=base_legend_items,
            title=None,
            reserved_chars_per_gap=15,
        ),
        LegendJob(
            image_path=ARTICLE_DIR / "aaa_布线结果对比.png",
            output_path=RESULTS_DIR / "aaa_布线结果对比_legend.png",
            legend_items=[
                *base_legend_items,
                LegendItem(kind="line", color="#FFF200"),
            ],
            title=None,
            reserved_chars_per_gap=11,
        ),
        LegendJob(
            image_path=ARTICLE_DIR / "aaa_布局结果对比.png",
            output_path=RESULTS_DIR / "aaa_布局结果对比_legend.png",
            legend_items=[
                *base_legend_items,
                LegendItem(kind="box", color="#050505"),
            ],
            title=None,
            reserved_chars_per_gap=11,
        ),
    ]

    for job in jobs:
        output_path = add_color_legend(
            image_path=job.image_path,
            output_path=job.output_path,
            legend_items=job.legend_items,
            title=job.title,
            reserved_chars_per_gap=job.reserved_chars_per_gap,
        )
        print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
