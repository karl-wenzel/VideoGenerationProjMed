"""Shared API-call timing and timeline rendering for the video pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TIMELINE_JSON_PATH = PROJECT_ROOT / "tmp" / "api_timeline.json"
DEFAULT_TIMELINE_SVG_PATH = PROJECT_ROOT / "tmp" / "api_timeline.svg"


@dataclass
class TimelineEvent:
    """One external API call observed by the pipeline."""

    event_id: str
    category: str
    name: str
    start_perf: float
    end_perf: float | None = None
    start_time: str = ""
    end_time: str = ""
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        end_perf = self.end_perf if self.end_perf is not None else time.perf_counter()
        return max(0.0, end_perf - self.start_perf)


class ApiTimeline:
    """Thread-safe store for API-call timing events."""

    def __init__(self) -> None:
        self._events: list[TimelineEvent] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._events = []

    def start(
        self,
        category: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event = TimelineEvent(
            event_id=uuid4().hex,
            category=category,
            name=name,
            start_perf=time.perf_counter(),
            start_time=_utc_timestamp(),
            metadata=metadata or {},
        )
        with self._lock:
            self._events.append(event)
        return event.event_id

    def end(
        self,
        event_id: str,
        *,
        status: str = "success",
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            event = next(
                item for item in self._events if item.event_id == event_id
            )
            event.end_perf = time.perf_counter()
            event.end_time = _utc_timestamp()
            event.status = status
            event.error = str(error) if error is not None else None

    def snapshot(self) -> list[TimelineEvent]:
        with self._lock:
            return [
                TimelineEvent(
                    event_id=event.event_id,
                    category=event.category,
                    name=event.name,
                    start_perf=event.start_perf,
                    end_perf=event.end_perf,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    status=event.status,
                    metadata=dict(event.metadata),
                    error=event.error,
                )
                for event in self._events
            ]

    @contextmanager
    def track(
        self,
        category: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        event_id = self.start(category, name, metadata)
        try:
            yield
        except Exception as error:
            self.end(event_id, status="error", error=error)
            raise
        else:
            self.end(event_id)

    @asynccontextmanager
    async def track_async(
        self,
        category: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ):
        event_id = self.start(category, name, metadata)
        try:
            yield
        except Exception as error:
            self.end(event_id, status="error", error=error)
            raise
        else:
            self.end(event_id)


api_timeline = ApiTimeline()


def track_api_call(
    category: str,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Track a synchronous external API call."""
    return api_timeline.track(category, name, metadata)


def track_api_call_async(
    category: str,
    name: str,
    metadata: dict[str, Any] | None = None,
):
    """Track an async external API call."""
    return api_timeline.track_async(category, name, metadata)


def reset_api_timeline() -> None:
    """Clear any previously recorded events."""
    api_timeline.reset()


def save_api_timeline(
    json_path: str | Path = DEFAULT_TIMELINE_JSON_PATH,
    svg_path: str | Path = DEFAULT_TIMELINE_SVG_PATH,
) -> tuple[Path, Path]:
    """Persist the current timeline as JSON and SVG."""
    events = api_timeline.snapshot()
    return save_timeline_events(events, json_path, svg_path)


def save_timeline_events(
    events: list[TimelineEvent],
    json_path: str | Path,
    svg_path: str | Path,
) -> tuple[Path, Path]:
    """Persist provided timing events as JSON and a flow diagram SVG."""
    resolved_json_path = Path(json_path)
    resolved_svg_path = Path(svg_path)
    resolved_json_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_svg_path.parent.mkdir(parents=True, exist_ok=True)

    event_rows = _serialize_events(events)
    resolved_json_path.write_text(
        json.dumps(event_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    resolved_svg_path.write_text(
        render_timeline_svg(events),
        encoding="utf-8",
    )
    return resolved_json_path, resolved_svg_path


def render_timeline_svg(events: list[TimelineEvent]) -> str:
    """Render a compact SVG time-flow diagram showing overlap and duration."""
    completed_events = [
        event for event in events if event.end_perf is not None
    ]
    if not completed_events:
        return _empty_svg("No API calls were recorded.")

    start_zero = min(event.start_perf for event in completed_events)
    end_time = max(event.end_perf or event.start_perf for event in completed_events)
    total_duration = max(0.001, end_time - start_zero)
    rows = sorted(completed_events, key=lambda item: (item.start_perf, item.end_perf or 0))

    left_margin = 310
    right_margin = 40
    top_margin = 98
    row_height = 34
    timeline_width = 900
    width = left_margin + timeline_width + right_margin
    height = top_margin + len(rows) * row_height + 70

    colors = {
        "text": "#6d5dfc",
        "image": "#0ea5a4",
        "audio": "#f59e0b",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,sans-serif;fill:#172033}",
        ".small{font-size:12px;fill:#475569}",
        ".label{font-size:13px;font-weight:600}",
        ".title{font-size:20px;font-weight:700}",
        ".axis{stroke:#cbd5e1;stroke-width:1}",
        ".grid{stroke:#e2e8f0;stroke-width:1}",
        ".bar{rx:5;ry:5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" class="title">API Call Time Flow</text>',
        f'<text x="24" y="56" class="small">Total observed API time window: {total_duration:.2f}s</text>',
    ]

    tick_count = 5
    axis_y = top_margin - 28
    for tick in range(tick_count + 1):
        ratio = tick / tick_count
        x = left_margin + ratio * timeline_width
        seconds = ratio * total_duration
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_margin - 6}" x2="{x:.1f}" y2="{height - 44}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x - 12:.1f}" y="{axis_y}" class="small">{seconds:.1f}s</text>'
        )
    parts.append(
        f'<line x1="{left_margin}" y1="{top_margin - 6}" x2="{left_margin + timeline_width}" y2="{top_margin - 6}" class="axis"/>'
    )

    header_y = top_margin - 10
    parts.append(f'<text x="24" y="{header_y}" class="small">API call</text>')
    parts.append(f'<text x="{left_margin}" y="{header_y}" class="small">Elapsed time</text>')

    for row_index, event in enumerate(rows):
        y = top_margin + row_index * row_height
        label = _row_label(event)
        start_offset = event.start_perf - start_zero
        x = left_margin + (start_offset / total_duration) * timeline_width
        bar_width = max(3.0, (event.duration_seconds / total_duration) * timeline_width)
        color = colors.get(event.category, "#64748b")
        opacity = "0.45" if event.status != "success" else "0.88"
        title = _event_title(event)

        parts.append(
            f'<text x="24" y="{y + 15}" class="label">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text x="24" y="{y + 29}" class="small">start {start_offset:.1f}s, duration {event.duration_seconds:.1f}s</text>'
        )
        parts.append(
            f'<line x1="{left_margin}" y1="{y + 9}" x2="{left_margin + timeline_width}" y2="{y + 9}" class="axis" opacity="0.25"/>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="20" class="bar" fill="{color}" opacity="{opacity}">'
            f"<title>{html.escape(title)}</title>"
            "</rect>"
        )
        if event.status != "success":
            marker_width = max(10.0, bar_width)
            parts.append(
                f'<rect x="{x:.1f}" y="{y - 5}" width="{marker_width:.1f}" height="4" fill="#dc2626" rx="2">'
                f"<title>{html.escape(title)}</title>"
                "</rect>"
            )
            parts.append(
                f'<text x="{x + marker_width + 6:.1f}" y="{y + 7}" class="small" fill="#dc2626">failed</text>'
            )

    legend_y = height - 24
    legend_x = 24
    for category, color in colors.items():
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 11}" width="14" height="14" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 20}" y="{legend_y}" class="small">{category}</text>'
        )
        legend_x += 90
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y - 8}" width="20" height="4" fill="#dc2626" rx="2"/>'
    )
    parts.append(
        f'<text x="{legend_x + 26}" y="{legend_y}" class="small">failed call</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def create_mock_timeline_diagram(
    json_path: str | Path = PROJECT_ROOT / "tmp" / "mock_api_timeline.json",
    svg_path: str | Path = PROJECT_ROOT / "tmp" / "mock_api_timeline.svg",
) -> tuple[Path, Path]:
    """Create a sample timeline diagram without making external API calls."""
    base = time.perf_counter()
    mock_events = [
        _mock_event(base, 0.0, 2.4, "text", "story_json_generation"),
        _mock_event(base, 2.7, 7.6, "image", "character_reference_1"),
        _mock_event(base, 2.8, 8.9, "image", "character_reference_2"),
        _mock_event(base, 9.2, 11.1, "text", "scene_1_layout"),
        _mock_event(base, 9.2, 11.6, "text", "scene_2_layout"),
        _mock_event(base, 11.9, 18.8, "image", "scene_1_pose"),
        _mock_event(base, 12.0, 19.6, "image", "scene_2_pose"),
        _mock_event(
            base,
            20.2,
            24.8,
            "image",
            "scene_1_final_attempt_failed",
            status="error",
            error="500 Server Error: Internal Server Error",
        ),
        _mock_event(base, 26.1, 34.0, "image", "scene_1_final_retry"),
        _mock_event(base, 20.3, 33.4, "image", "scene_2_final"),
        _mock_event(base, 34.2, 39.8, "audio", "scene_1_tts"),
        _mock_event(base, 39.9, 45.1, "audio", "scene_2_tts"),
    ]
    return save_timeline_events(mock_events, json_path, svg_path)


def _serialize_events(events: list[TimelineEvent]) -> list[dict[str, Any]]:
    if not events:
        return []

    start_zero = min(event.start_perf for event in events)
    return [
        {
            "id": event.event_id,
            "category": event.category,
            "name": event.name,
            "status": event.status,
            "start_offset_seconds": round(event.start_perf - start_zero, 4),
            "duration_seconds": round(event.duration_seconds, 4),
            "start_time_utc": event.start_time,
            "end_time_utc": event.end_time,
            "metadata": event.metadata,
            "error": event.error,
        }
        for event in sorted(events, key=lambda item: item.start_perf)
    ]


def _event_title(event: TimelineEvent) -> str:
    details = [
        f"{event.category}: {event.name}",
        f"duration: {event.duration_seconds:.2f}s",
        f"status: {event.status}",
    ]
    for key, value in event.metadata.items():
        details.append(f"{key}: {value}")
    if event.error:
        details.append(f"error: {event.error}")
    return "\n".join(details)


def _row_label(event: TimelineEvent) -> str:
    label = f"{event.category}: {event.name}"
    max_length = 43
    if len(label) <= max_length:
        return label
    return f"{label[: max_length - 1]}..."


def _mock_event(
    base: float,
    start_offset: float,
    end_offset: float,
    category: str,
    name: str,
    *,
    status: str = "success",
    error: str | None = None,
) -> TimelineEvent:
    now = _utc_timestamp()
    return TimelineEvent(
        event_id=uuid4().hex,
        category=category,
        name=name,
        start_perf=base + start_offset,
        end_perf=base + end_offset,
        start_time=now,
        end_time=now,
        status=status,
        error=error,
    )


def _empty_svg(message: str) -> str:
    escaped_message = html.escape(message)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="160">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<text x="24" y="48" font-family="Arial,sans-serif" '
        f'font-size="18" fill="#172033">{escaped_message}</text></svg>'
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


if __name__ == "__main__":
    json_path, svg_path = create_mock_timeline_diagram()
    print(f"Mock timeline JSON saved to: {json_path}")
    print(f"Mock timeline SVG saved to: {svg_path}")
