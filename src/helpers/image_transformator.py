"""Image transformation utilities using the Albumentations library."""

import albumentations as A


class ImageTransformator:
    """Apply resize, padding, and optional augmentation to images with COCO bboxes.

    Parameters
    ----------
    image_size : tuple[int, int]
        Target (height, width) after resizing and padding.
    augmentation : bool
        If True, include random horizontal flips, brightness/contrast
        adjustments, and small affine shifts.
    """

    def __init__(
        self, image_size: tuple[int, int] = (320, 320), augmentation: bool = False
    ):
        transforms = []

        if augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.2,
                ),
            ])

        transforms.extend([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0], min_width=image_size[1]
            ),
        ])

        self.transform_pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="coco", label_fields=["labels"]
            ),
        )

    def transform(self, image, bboxes, labels):
        """Apply the transformation pipeline to an image and its annotations.

        Parameters
        ----------
        image : np.ndarray
            RGB image array.
        bboxes : list[list[float]]
            COCO-format bounding boxes [x, y, w, h].
        labels : list[int]
            Category IDs corresponding to each bbox.

        Returns
        -------
        dict
            Transformed image, bboxes, and labels.
        """
        return self.transform_pipeline(
            image=image, bboxes=bboxes, labels=labels
        )
