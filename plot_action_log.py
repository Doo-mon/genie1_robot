"""从 ``actions.jsonl`` 离线生成逐维模型/实际动作曲线。

该脚本只读取已经落盘的日志，不依赖推理进程仍然存活，因此 Shell 可在正常
退出、异常或 Ctrl-C 后统一调用。每个动作维度生成一张独立 SVG，同一张图中
灰色点线表示服务端原始输出，红色虚线表示执行队列选中的有效模型目标，蓝色
实线表示限幅、滞回/二值化后的实际下发命令；额外生成 ``plots/index.html``，
以及把所有维度放进同一画布的 ``plots/overview.svg``。总览中的每个子图都有
独立的纵轴范围，避免位置、旋转和夹爪等不同量纲互相压缩。

实现仅使用 Python 标准库，不要求 Matplotlib。日志中没有任何成功下发动作时
会正常退出并明确提示跳过，不生成误导性的空曲线；Ctrl-C 留下的最后一行若不
完整也会被安全忽略，其余已完成记录仍可绘制。
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SVG_WIDTH = 1000
SVG_HEIGHT = 520
LEFT = 82
RIGHT = 28
TOP = 54
BOTTOM = 66

# 总览图采用两列子图，既能一次看到所有维度，又不会让 14/16 维动作把单个
# 子图压得过矮。总览仍然是纯 SVG，复制到没有 Python 环境的机器也能打开。
OVERVIEW_WIDTH = 1600
OVERVIEW_COLUMNS = 2
OVERVIEW_OUTER_X = 24
OVERVIEW_TOP = 78
OVERVIEW_PANEL_HEIGHT = 270
OVERVIEW_COLUMN_GAP = 22
OVERVIEW_ROW_GAP = 20
OVERVIEW_BOTTOM = 24


def _safe_file_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return cleaned or "dimension"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_action_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    ignored = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records, ignored
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            model = [float(value) for value in record["model_action"]]
            actual = [float(value) for value in record["actual_action"]]
            raw_model = [
                float(value)
                for value in record.get("raw_model_action", record["model_action"])
            ]
            elapsed = float(record["elapsed_s"])
            if (
                len(raw_model) != len(model)
                or len(model) != len(actual)
                or not model
                or not math.isfinite(elapsed)
                or not all(
                    math.isfinite(value) for value in raw_model + model + actual
                )
            ):
                raise ValueError("invalid action row")
            record["model_action"] = model
            record["actual_action"] = actual
            record["raw_model_action"] = raw_model
            record["elapsed_s"] = elapsed
            records.append(record)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            ignored += 1
    return records, ignored


def _scaled_range(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        padding = max(abs(low) * 0.05, 0.05)
    else:
        padding = (high - low) * 0.08
    return low - padding, high + padding


def _points(
    xs: list[float],
    ys: list[float],
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> str:
    plot_width = SVG_WIDTH - LEFT - RIGHT
    plot_height = SVG_HEIGHT - TOP - BOTTOM
    result = []
    # ``xs`` and ``ys`` are validated to the same record count by the caller.
    # Avoid ``zip(strict=True)`` here so the standalone exit-hook plotter also
    # works with the system Python shipped on older robot images.
    for x_value, y_value in zip(xs, ys):
        x = LEFT + (x_value - x_low) / (x_high - x_low) * plot_width
        y = TOP + (y_high - y_value) / (y_high - y_low) * plot_height
        result.append(f"{x:.2f},{y:.2f}")
    return " ".join(result)


def _points_in_rect(
    xs: list[float],
    ys: list[float],
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    """把一组数据坐标映射到指定子图，供总览图的独立坐标轴复用。"""

    result = []
    for x_value, y_value in zip(xs, ys):
        x = left + (x_value - x_low) / (x_high - x_low) * width
        y = top + (y_high - y_value) / (y_high - y_low) * height
        result.append(f"{x:.2f},{y:.2f}")
    return " ".join(result)


def _render_svg(
    dimension_name: str,
    xs: list[float],
    model: list[float],
    actual: list[float],
    raw_model: list[float],
) -> str:
    x_low, x_high = _scaled_range(xs)
    y_low, y_high = _scaled_range(raw_model + model + actual)
    raw_model_points = _points(
        xs, raw_model, x_low=x_low, x_high=x_high, y_low=y_low, y_high=y_high
    )
    model_points = _points(
        xs,
        model,
        x_low=x_low,
        x_high=x_high,
        y_low=y_low,
        y_high=y_high,
    )
    actual_points = _points(
        xs,
        actual,
        x_low=x_low,
        x_high=x_high,
        y_low=y_low,
        y_high=y_high,
    )
    title = html.escape(dimension_name)
    plot_width = SVG_WIDTH - LEFT - RIGHT
    plot_height = SVG_HEIGHT - TOP - BOTTOM

    grid_lines: list[str] = []
    labels: list[str] = []
    for index in range(6):
        fraction = index / 5.0
        x = LEFT + fraction * plot_width
        y = TOP + fraction * plot_height
        x_value = x_low + fraction * (x_high - x_low)
        y_value = y_high - fraction * (y_high - y_low)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" '
            f'y2="{TOP + plot_height}" class="grid" />'
        )
        grid_lines.append(
            f'<line x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + plot_width}" '
            f'y2="{y:.2f}" class="grid" />'
        )
        labels.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - 34}" '
            f'class="tick" text-anchor="middle">{x_value:.2f}</text>'
        )
        labels.append(
            f'<text x="{LEFT - 10}" y="{y + 4:.2f}" '
            f'class="tick" text-anchor="end">{y_value:.4g}</text>'
        )

    circles = ""
    if len(xs) == 1:
        raw_xy = raw_model_points.split()[0].split(",")
        model_xy = model_points.split()[0].split(",")
        actual_xy = actual_points.split()[0].split(",")
        circles = (
            f'<circle cx="{raw_xy[0]}" cy="{raw_xy[1]}" r="4" class="raw" />'
            f'<circle cx="{model_xy[0]}" cy="{model_xy[1]}" r="4" class="model" />'
            f'<circle cx="{actual_xy[0]}" cy="{actual_xy[1]}" r="4" class="actual" />'
        )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <style>
    .background {{ fill: #ffffff; }}
    .grid {{ stroke: #d9dee8; stroke-width: 1; }}
    .axis {{ stroke: #303744; stroke-width: 1.4; }}
    .model {{ fill: none; stroke: #d94841; stroke-width: 2; stroke-dasharray: 7 4; }}
    .raw {{ fill: none; stroke: #777f8c; stroke-width: 1.8; stroke-dasharray: 2 4; }}
    .actual {{ fill: none; stroke: #2468d8; stroke-width: 2.2; }}
    .title {{ font: 600 19px sans-serif; fill: #202632; }}
    .label {{ font: 14px sans-serif; fill: #303744; }}
    .tick {{ font: 12px sans-serif; fill: #596273; }}
    .legend {{ font: 13px sans-serif; fill: #303744; }}
  </style>
  <rect width="100%" height="100%" class="background" />
  <text x="{LEFT}" y="29" class="title">{title}: raw vs effective vs actual</text>
  {''.join(grid_lines)}
  <line x1="{LEFT}" y1="{TOP + plot_height}" x2="{LEFT + plot_width}" y2="{TOP + plot_height}" class="axis" />
  <line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_height}" class="axis" />
  {''.join(labels)}
  <polyline points="{raw_model_points}" class="raw" />
  <polyline points="{model_points}" class="model" />
  <polyline points="{actual_points}" class="actual" />
  {circles}
  <line x1="{SVG_WIDTH - 445}" y1="28" x2="{SVG_WIDTH - 410}" y2="28" class="raw" />
  <text x="{SVG_WIDTH - 402}" y="33" class="legend">raw server</text>
  <line x1="{SVG_WIDTH - 300}" y1="28" x2="{SVG_WIDTH - 265}" y2="28" class="model" />
  <text x="{SVG_WIDTH - 257}" y="33" class="legend">effective target</text>
  <line x1="{SVG_WIDTH - 120}" y1="28" x2="{SVG_WIDTH - 85}" y2="28" class="actual" />
  <text x="{SVG_WIDTH - 77}" y="33" class="legend">actual</text>
  <text x="{LEFT + plot_width / 2:.2f}" y="{SVG_HEIGHT - 9}" class="label" text-anchor="middle">elapsed time (s)</text>
  <text x="18" y="{TOP + plot_height / 2:.2f}" class="label" text-anchor="middle" transform="rotate(-90 18 {TOP + plot_height / 2:.2f})">action value</text>
</svg>
'''


def _render_overview_svg(
    names: list[str],
    xs: list[float],
    raw_models: list[list[float]],
    models: list[list[float]],
    actuals: list[list[float]],
) -> str:
    """生成所有维度共用一个画布、但各自独立缩放的总览 SVG。"""

    row_count = (len(names) + OVERVIEW_COLUMNS - 1) // OVERVIEW_COLUMNS
    total_height = (
        OVERVIEW_TOP
        + row_count * OVERVIEW_PANEL_HEIGHT
        + max(0, row_count - 1) * OVERVIEW_ROW_GAP
        + OVERVIEW_BOTTOM
    )
    panel_width = (
        OVERVIEW_WIDTH
        - 2 * OVERVIEW_OUTER_X
        - (OVERVIEW_COLUMNS - 1) * OVERVIEW_COLUMN_GAP
    ) / OVERVIEW_COLUMNS
    plot_left_in_panel = 68.0
    plot_right_in_panel = 16.0
    plot_top_in_panel = 38.0
    plot_bottom_in_panel = 43.0
    plot_width = panel_width - plot_left_in_panel - plot_right_in_panel
    plot_height = OVERVIEW_PANEL_HEIGHT - plot_top_in_panel - plot_bottom_in_panel
    x_low, x_high = _scaled_range(xs)

    definitions: list[str] = []
    panels: list[str] = []
    for index, name in enumerate(names):
        row = index // OVERVIEW_COLUMNS
        column = index % OVERVIEW_COLUMNS
        panel_x = OVERVIEW_OUTER_X + column * (panel_width + OVERVIEW_COLUMN_GAP)
        panel_y = OVERVIEW_TOP + row * (
            OVERVIEW_PANEL_HEIGHT + OVERVIEW_ROW_GAP
        )
        plot_left = panel_x + plot_left_in_panel
        plot_top = panel_y + plot_top_in_panel
        y_low, y_high = _scaled_range(
            raw_models[index] + models[index] + actuals[index]
        )
        definitions.append(
            f'<clipPath id="overview-clip-{index}">'
            f'<rect x="{plot_left:.2f}" y="{plot_top:.2f}" '
            f'width="{plot_width:.2f}" height="{plot_height:.2f}" />'
            f'</clipPath>'
        )

        grid: list[str] = []
        labels: list[str] = []
        # 少量网格线让总览保持清楚；每个面板的纵轴数值都单独计算。
        for tick_index in range(5):
            fraction = tick_index / 4.0
            x = plot_left + fraction * plot_width
            y = plot_top + fraction * plot_height
            x_value = x_low + fraction * (x_high - x_low)
            y_value = y_high - fraction * (y_high - y_low)
            grid.append(
                f'<line x1="{x:.2f}" y1="{plot_top:.2f}" '
                f'x2="{x:.2f}" y2="{plot_top + plot_height:.2f}" '
                f'class="overview-grid" />'
            )
            grid.append(
                f'<line x1="{plot_left:.2f}" y1="{y:.2f}" '
                f'x2="{plot_left + plot_width:.2f}" y2="{y:.2f}" '
                f'class="overview-grid" />'
            )
            labels.append(
                f'<text x="{x:.2f}" y="{plot_top + plot_height + 18:.2f}" '
                f'class="overview-tick" text-anchor="middle">'
                f'{x_value:.2f}</text>'
            )
            labels.append(
                f'<text x="{plot_left - 7:.2f}" y="{y + 4:.2f}" '
                f'class="overview-tick" text-anchor="end">'
                f'{y_value:.4g}</text>'
            )

        raw_points = _points_in_rect(
            xs,
            raw_models[index],
            x_low=x_low,
            x_high=x_high,
            y_low=y_low,
            y_high=y_high,
            left=plot_left,
            top=plot_top,
            width=plot_width,
            height=plot_height,
        )
        model_points = _points_in_rect(
            xs,
            models[index],
            x_low=x_low,
            x_high=x_high,
            y_low=y_low,
            y_high=y_high,
            left=plot_left,
            top=plot_top,
            width=plot_width,
            height=plot_height,
        )
        actual_points = _points_in_rect(
            xs,
            actuals[index],
            x_low=x_low,
            x_high=x_high,
            y_low=y_low,
            y_high=y_high,
            left=plot_left,
            top=plot_top,
            width=plot_width,
            height=plot_height,
        )
        point_markers = ""
        if len(xs) == 1:
            marker_parts: list[str] = []
            for points, css_class in (
                (raw_points, "overview-raw-point"),
                (model_points, "overview-model-point"),
                (actual_points, "overview-actual-point"),
            ):
                x_text, y_text = points.split(",")
                marker_parts.append(
                    f'<circle cx="{x_text}" cy="{y_text}" r="3" '
                    f'class="{css_class}" />'
                )
            point_markers = "".join(marker_parts)

        panels.append(
            f'<g class="overview-panel">'
            f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" '
            f'width="{panel_width:.2f}" height="{OVERVIEW_PANEL_HEIGHT}" '
            f'rx="7" class="overview-panel-background" />'
            f'<text x="{plot_left:.2f}" y="{panel_y + 25:.2f}" '
            f'class="overview-panel-title">{html.escape(name)}</text>'
            f'{"".join(grid)}'
            f'<line x1="{plot_left:.2f}" y1="{plot_top + plot_height:.2f}" '
            f'x2="{plot_left + plot_width:.2f}" '
            f'y2="{plot_top + plot_height:.2f}" class="overview-axis" />'
            f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" '
            f'x2="{plot_left:.2f}" y2="{plot_top + plot_height:.2f}" '
            f'class="overview-axis" />'
            f'{"".join(labels)}'
            f'<polyline points="{raw_points}" class="overview-raw" '
            f'clip-path="url(#overview-clip-{index})" />'
            f'<polyline points="{model_points}" class="overview-model" '
            f'clip-path="url(#overview-clip-{index})" />'
            f'<polyline points="{actual_points}" class="overview-actual" '
            f'clip-path="url(#overview-clip-{index})" />'
            f'{point_markers}'
            f'<text x="{plot_left + plot_width / 2:.2f}" '
            f'y="{panel_y + OVERVIEW_PANEL_HEIGHT - 8:.2f}" '
            f'class="overview-axis-label" text-anchor="middle">elapsed time (s)</text>'
            f'</g>'
        )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{OVERVIEW_WIDTH}" height="{total_height}" viewBox="0 0 {OVERVIEW_WIDTH} {total_height}">
  <defs>{''.join(definitions)}</defs>
  <style>
    .overview-background {{ fill: #f5f7fa; }}
    .overview-panel-background {{ fill: #ffffff; stroke: #d5dae3; stroke-width: 1; }}
    .overview-grid {{ stroke: #e0e4eb; stroke-width: 1; }}
    .overview-axis {{ stroke: #303744; stroke-width: 1.15; }}
    .overview-raw {{ fill: none; stroke: #777f8c; stroke-width: 1.35; stroke-dasharray: 2 4; }}
    .overview-model {{ fill: none; stroke: #d94841; stroke-width: 1.55; stroke-dasharray: 7 4; }}
    .overview-actual {{ fill: none; stroke: #2468d8; stroke-width: 1.8; }}
    .overview-raw-point {{ fill: #777f8c; }}
    .overview-model-point {{ fill: #d94841; }}
    .overview-actual-point {{ fill: #2468d8; }}
    .overview-title {{ font: 600 22px sans-serif; fill: #202632; }}
    .overview-panel-title {{ font: 600 15px sans-serif; fill: #202632; }}
    .overview-tick {{ font: 10px sans-serif; fill: #596273; }}
    .overview-axis-label {{ font: 11px sans-serif; fill: #596273; }}
    .overview-legend {{ font: 13px sans-serif; fill: #303744; }}
  </style>
  <rect width="100%" height="100%" class="overview-background" />
  <text x="{OVERVIEW_OUTER_X}" y="31" class="overview-title">All action dimensions: raw vs effective vs actual</text>
  <line x1="{OVERVIEW_WIDTH - 474}" y1="27" x2="{OVERVIEW_WIDTH - 439}" y2="27" class="overview-raw" />
  <text x="{OVERVIEW_WIDTH - 431}" y="32" class="overview-legend">raw server</text>
  <line x1="{OVERVIEW_WIDTH - 324}" y1="27" x2="{OVERVIEW_WIDTH - 289}" y2="27" class="overview-model" />
  <text x="{OVERVIEW_WIDTH - 281}" y="32" class="overview-legend">effective target</text>
  <line x1="{OVERVIEW_WIDTH - 137}" y1="27" x2="{OVERVIEW_WIDTH - 102}" y2="27" class="overview-actual" />
  <text x="{OVERVIEW_WIDTH - 94}" y="32" class="overview-legend">actual</text>
  <text x="{OVERVIEW_OUTER_X}" y="54" class="overview-axis-label">Each subplot uses its own Y-axis range.</text>
  {''.join(panels)}
</svg>
'''


def plot_log_directory(log_directory: Path) -> int:
    log_directory = log_directory.expanduser().resolve()
    records, ignored = _read_action_records(log_directory / "actions.jsonl")
    if not records:
        print(
            f"[action-plot] no successfully applied actions in {log_directory}; skip"
        )
        return 0

    dimension = len(records[0]["model_action"])
    consistent_records = [
        record
        for record in records
        if len(record["model_action"]) == dimension
        and len(record["raw_model_action"]) == dimension
        and len(record["actual_action"]) == dimension
    ]
    ignored += len(records) - len(consistent_records)
    records = consistent_records
    if not records:
        print(f"[action-plot] no consistent action rows in {log_directory}; skip")
        return 0

    metadata = _read_json(log_directory / "run.json")
    names = metadata.get("dimension_names")
    if not isinstance(names, list) or len(names) != dimension:
        names = [f"dim_{index:02d}" for index in range(dimension)]
    names = [str(name) for name in names]

    output = log_directory / "plots"
    output.mkdir(parents=True, exist_ok=True)
    xs = [float(record["elapsed_s"]) for record in records]
    generated: list[tuple[str, str]] = []
    all_models: list[list[float]] = []
    all_actuals: list[list[float]] = []
    all_raw_models: list[list[float]] = []
    for index, name in enumerate(names):
        model = [float(record["model_action"][index]) for record in records]
        actual = [float(record["actual_action"][index]) for record in records]
        raw_model = [
            float(record["raw_model_action"][index]) for record in records
        ]
        all_models.append(model)
        all_actuals.append(actual)
        all_raw_models.append(raw_model)
        file_name = f"{index:02d}_{_safe_file_name(name)}.svg"
        (output / file_name).write_text(
            _render_svg(name, xs, model, actual, raw_model),
            encoding="utf-8",
        )
        generated.append((name, file_name))

    overview_file_name = "overview.svg"
    (output / overview_file_name).write_text(
        _render_overview_svg(
            names,
            xs,
            all_raw_models,
            all_models,
            all_actuals,
        ),
        encoding="utf-8",
    )
    (output / "overview.html").write_text(
        f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>动作曲线总览</title>
<style>body{{font-family:sans-serif;margin:24px;background:#f5f7fa;color:#202632}}main{{max-width:{OVERVIEW_WIDTH}px;margin:auto}}img{{display:block;width:100%;height:auto;background:white;box-shadow:0 1px 6px #ccd2dc}}a{{color:#2468d8}}</style></head>
<body><main><h1>模型动作与实际动作总览</h1><p>每个维度使用独立纵轴；灰色为服务端原始输出，红色为有效模型目标，蓝色为实际下发动作。<a href="index.html">查看逐维大图</a></p><img src="{overview_file_name}" alt="全部动作维度总览"></main></body></html>
''',
        encoding="utf-8",
    )

    cards = "\n".join(
        f'<section><h2>{html.escape(name)}</h2><img src="{html.escape(file_name)}" '
        f'alt="{html.escape(name)} action curve"></section>'
        for name, file_name in generated
    )
    (output / "index.html").write_text(
        f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Action curves</title>
<style>body{{font-family:sans-serif;margin:24px;background:#f5f7fa;color:#202632}}section{{background:white;padding:12px 18px;margin:0 0 20px;border-radius:8px;box-shadow:0 1px 5px #ccd2dc}}img{{width:100%;height:auto}}h1,h2{{margin:8px 0}}a{{color:#2468d8}}.overview{{border:2px solid #b9c9e9}}</style></head>
<body><h1>模型动作与实际动作逐维对比</h1><p>成功动作数：{len(records)}；忽略的损坏/不兼容行：{ignored}；<a href="overview.html">单独打开全部维度总览</a></p><section class="overview"><h2>全部维度总览（每个子图独立纵轴）</h2><a href="overview.html"><img src="{overview_file_name}" alt="全部动作维度总览"></a></section>{cards}</body></html>
''',
        encoding="utf-8",
    )
    (output / "plot_summary.json").write_text(
        json.dumps(
            {
                "source": str(log_directory / "actions.jsonl"),
                "records": len(records),
                "ignored_rows": ignored,
                "dimensions": names,
                "format": "svg",
                "overview": overview_file_name,
                "overview_html": "overview.html",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[action-plot] generated {dimension} dimension plots and overview: "
        f"{output / 'overview.html'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot model targets and actual commands from one inference log"
    )
    parser.add_argument("log_directory", type=Path)
    args = parser.parse_args()
    return plot_log_directory(args.log_directory)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[action-plot] interrupted", file=sys.stderr)
        raise SystemExit(130)
