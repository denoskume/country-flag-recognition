"""Image transforms shared by training, evaluation and deployment."""

from torchvision import transforms


def build_train_transform(image_size: int = 224):
    """Moderate augmentation for realistic flag-image variation."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.20,
                    )
                ],
                p=0.7,
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomPerspective(
                distortion_scale=0.20,
                p=0.35,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_eval_transform(image_size: int = 224):
    """Deterministic preprocessing for validation, test and deployment."""
    resize_size = round(image_size * 256 / 224)

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
