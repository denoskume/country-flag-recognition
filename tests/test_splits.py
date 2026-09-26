from pathlib import Path

from flag_recognition.splits import (
    SplitConfig,
    build_split_manifest,
    choose_unseen_classes,
)


def make_class_paths(
    country: str,
    count: int,
) -> list[Path]:
    return [
        Path("data/raw")
        / country
        / f"image_{index:03d}.jpg"
        for index in range(count)
    ]


def make_canonical_paths(
    country: str,
    count: int,
) -> list[Path]:
    return [
        Path("data/raw")
        / country
        / f"canonical_{index:02d}.jpg"
        for index in range(count)
    ]


def test_unseen_partition_is_reproducible():
    classes = [
        "france",
        "ghana",
        "japan",
        "brazil",
        "canada",
    ]

    seen_a, unseen_a = (
        choose_unseen_classes(
            classes,
            unseen_fraction=0.20,
            seed=42,
        )
    )
    seen_b, unseen_b = (
        choose_unseen_classes(
            classes,
            unseen_fraction=0.20,
            seed=42,
        )
    )

    assert seen_a == seen_b
    assert unseen_a == unseen_b
    assert set(seen_a).isdisjoint(
        unseen_a
    )


def test_unseen_classes_never_enter_training_or_validation():
    class_to_images = {
        country: make_class_paths(
            country,
            8,
        )
        for country in (
            "france",
            "ghana",
            "japan",
            "brazil",
            "canada",
        )
    }

    manifest = build_split_manifest(
        class_to_images,
        SplitConfig(
            unseen_fraction=0.20,
            validation_fraction=0.20,
            test_fraction=0.20,
            minimum_images_per_seen_class=5,
            seed=7,
        ),
    )

    unseen = manifest[
        manifest["regime"]
        == "unseen"
    ]

    assert not unseen.empty
    assert set(
        unseen["partition"]
    ) == {"test"}

    seen = manifest[
        manifest["regime"]
        == "seen"
    ]

    assert {
        "train",
        "validation",
        "test",
    } <= set(
        seen["partition"]
    )


def test_every_source_path_appears_once():
    class_to_images = {
        country: make_class_paths(
            country,
            7,
        )
        for country in (
            "france",
            "ghana",
            "japan",
            "brazil",
        )
    }

    manifest = build_split_manifest(
        class_to_images,
        SplitConfig(
            unseen_fraction=0.25,
            validation_fraction=0.20,
            test_fraction=0.20,
            minimum_images_per_seen_class=5,
            seed=12,
        ),
    )

    assert manifest[
        "path"
    ].is_unique
    assert len(
        manifest
    ) == 28


def test_seen_canonical_variants_are_training_only():
    class_to_images = {
        country: (
            make_class_paths(
                country,
                6,
            )
            + make_canonical_paths(
                country,
                12,
            )
        )
        for country in (
            "france",
            "ghana",
            "japan",
            "brazil",
            "canada",
        )
    }

    manifest = build_split_manifest(
        class_to_images,
        SplitConfig(
            unseen_fraction=0.20,
            validation_fraction=0.20,
            test_fraction=0.20,
            minimum_images_per_seen_class=5,
            seed=3,
        ),
    )

    seen_canonical = manifest[
        (
            manifest["regime"]
            == "seen"
        )
        & (
            manifest["source_type"]
            == "canonical_augmented"
        )
    ]

    assert not seen_canonical.empty
    assert set(
        seen_canonical[
            "partition"
        ]
    ) == {"train"}
    assert set(
        seen_canonical[
            "evaluation_scope"
        ]
    ) == {
        "training_support"
    }


def test_seen_validation_and_test_are_real_world_only():
    class_to_images = {
        country: (
            make_class_paths(
                country,
                6,
            )
            + make_canonical_paths(
                country,
                12,
            )
        )
        for country in (
            "france",
            "ghana",
            "japan",
            "brazil",
            "canada",
        )
    }

    manifest = build_split_manifest(
        class_to_images,
        SplitConfig(
            unseen_fraction=0.20,
            validation_fraction=0.20,
            test_fraction=0.20,
            minimum_images_per_seen_class=5,
            seed=9,
        ),
    )

    evaluation = manifest[
        (
            manifest["regime"]
            == "seen"
        )
        & (
            manifest["partition"]
            .isin(
                [
                    "validation",
                    "test",
                ]
            )
        )
    ]

    assert not evaluation.empty
    assert set(
        evaluation[
            "source_type"
        ]
    ) == {"real_world"}
    assert set(
        evaluation[
            "evaluation_scope"
        ]
    ) == {
        "real_world_benchmark"
    }
