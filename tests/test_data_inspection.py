"""Unit tests for dataset inspection, video metadata analysis, and report generation."""

from pathlib import Path

from sanket.data.inspection import (
    ClassDistributionStats,
    DatasetInspectionSummary,
    VideoMetadata,
    compute_inspection_summary,
    generate_markdown_report,
    inspect_dataset_directory,
    inspect_video_file,
    parse_signer_from_filename,
    save_inspection_report,
)


def test_parse_signer_from_filename() -> None:
    """Test extracting signer identifiers from common file naming patterns."""
    assert parse_signer_from_filename("Signer01_HELLO_01.mp4") == "Signer01"
    assert parse_signer_from_filename("HELLO_S2_03.mp4") == "S2"
    assert parse_signer_from_filename("include_clip_s05.mp4") == "s05"
    assert parse_signer_from_filename("random_clip.mp4") is None


def test_inspect_video_file_nonexistent(tmp_path: Path) -> None:
    """Test that inspecting a non-existent file marks it invalid."""
    fake_path = tmp_path / "does_not_exist.mp4"
    meta = inspect_video_file(fake_path)
    assert meta.is_valid is False
    assert "does not exist" in (meta.error_message or "").lower()


def test_inspect_video_file_too_small(tmp_path: Path) -> None:
    """Test that inspecting a truncated zero/small file marks it invalid."""
    small_file = tmp_path / "truncated.mp4"
    small_file.write_bytes(b"short")

    meta = inspect_video_file(small_file, min_bytes=512)
    assert meta.is_valid is False
    assert "too small" in (meta.error_message or "").lower()


def test_compute_inspection_summary_aggregations(tmp_path: Path) -> None:
    """Test calculating aggregate summary statistics from a list of video records."""
    records = [
        VideoMetadata(
            file_path=tmp_path / "HELLO" / "Signer01_HELLO_01.mp4",
            class_name="HELLO",
            signer_id="Signer01",
            file_size_bytes=500000,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=60,
            duration_seconds=2.0,
            is_valid=True,
        ),
        VideoMetadata(
            file_path=tmp_path / "HELLO" / "Signer02_HELLO_02.mp4",
            class_name="HELLO",
            signer_id="Signer02",
            file_size_bytes=750000,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=90,
            duration_seconds=3.0,
            is_valid=True,
        ),
        VideoMetadata(
            file_path=tmp_path / "WATER" / "Signer01_WATER_01.mp4",
            class_name="WATER",
            signer_id="Signer01",
            file_size_bytes=600000,
            width=1280,
            height=720,
            fps=25.0,
            frame_count=50,
            duration_seconds=2.0,
            is_valid=True,
        ),
        VideoMetadata(
            file_path=tmp_path / "WATER" / "corrupted_01.mp4",
            class_name="WATER",
            file_size_bytes=100,
            is_valid=False,
            error_message="File too small",
        ),
    ]

    summary = compute_inspection_summary(records, dataset_name="TEST-DATASET")
    assert summary.total_videos == 4
    assert summary.valid_videos == 3
    assert summary.corrupted_videos == 1
    assert summary.unique_classes == 2
    assert summary.unique_signers == 2
    assert summary.total_duration_seconds == 7.0
    assert summary.mean_duration_seconds == round(7.0 / 3.0, 2)
    assert summary.resolution_distribution["1920x1080"] == 2
    assert summary.resolution_distribution["1280x720"] == 1
    assert "HELLO" in summary.class_stats
    assert summary.class_stats["HELLO"].sample_count == 2
    assert summary.class_stats["HELLO"].mean_duration_seconds == 2.5


def test_generate_markdown_report_structure() -> None:
    """Test generating a markdown inspection report from summary metrics."""
    summary = DatasetInspectionSummary(
        dataset_name="INCLUDE-16",
        total_videos=10,
        valid_videos=9,
        corrupted_videos=1,
        unique_classes=2,
        unique_signers=2,
        total_duration_seconds=20.0,
        mean_duration_seconds=2.22,
        median_duration_seconds=2.0,
        resolution_distribution={"1920x1080": 9},
        fps_distribution={"30fps": 9},
        class_stats={
            "HELLO": ClassDistributionStats(
                class_name="HELLO",
                sample_count=5,
                total_duration_seconds=10.0,
                mean_duration_seconds=2.0,
                min_duration_seconds=1.5,
                max_duration_seconds=2.5,
                signers=["S1", "S2"],
            ),
        },
        corrupted_files=[
            {"filename": "bad.mp4", "class": "HELLO", "bytes": 10, "reason": "Corrupted header"},
        ],
    )

    md = generate_markdown_report(summary)
    assert "# Dataset Inspection & Label Balance Report — INCLUDE-16" in md
    assert "Total Video Files Evaluated" in md
    assert "1920x1080" in md
    assert "| `HELLO` | 5 |" in md
    assert "bad.mp4" in md


def test_inspect_dataset_directory_and_save_report(tmp_path: Path) -> None:
    """Test directory inspection scanner and disk saving."""
    dataset_dir = tmp_path / "raw_videos"
    hello_dir = dataset_dir / "HELLO"
    hello_dir.mkdir(parents=True, exist_ok=True)

    dummy_clip = hello_dir / "Signer01_HELLO_01.mp4"
    try:
        import cv2
        import numpy as np

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(dummy_clip), fourcc, 10.0, (100, 100))
        for _ in range(5):
            out.write(np.zeros((100, 100, 3), dtype=np.uint8))
        out.release()
    except ImportError:
        dummy_clip.write_bytes(b"A" * 2048)

    summary = inspect_dataset_directory(dataset_dir=dataset_dir, dataset_name="MOCK-INCLUDE")
    assert summary.total_videos == 1
    assert summary.valid_videos == 1
    assert summary.unique_classes == 1

    report_md = tmp_path / "reports" / "report.md"
    report_json = tmp_path / "reports" / "report.json"
    save_inspection_report(summary, report_md, report_json)

    assert report_md.exists()
    assert report_json.exists()
