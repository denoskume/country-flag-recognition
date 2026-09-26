from pathlib import Path

from flag_recognition.dataset import (
    discover_country_images,
)


def test_country_discovery_is_recursive_and_extension_filtered(
    tmp_path: Path,
):
    france = tmp_path / "france"
    ghana = tmp_path / "ghana"
    nested = france / "photos"

    nested.mkdir(parents=True)
    ghana.mkdir()

    (france / "flag.jpg").write_bytes(b"x")
    (nested / "street.PNG").write_bytes(b"x")
    (ghana / "flag.webp").write_bytes(b"x")
    (ghana / "notes.txt").write_text(
        "ignore",
        encoding="utf-8",
    )

    discovered = discover_country_images(
        tmp_path
    )

    assert sorted(discovered) == [
        "france",
        "ghana",
    ]
    assert [
        path.name
        for path in discovered["france"]
    ] == [
        "flag.jpg",
        "street.PNG",
    ]
    assert [
        path.name
        for path in discovered["ghana"]
    ] == [
        "flag.webp",
    ]
