import albumentations as A

class ImageTransformator:
    def __init__(self, image_size=(320, 320), augmentation=False):

        transforms = []

        if augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.2
                )
            ])

        transforms.extend([
            A.LongestMaxSize(
                max_size=max(image_size)
            ),

            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1]
            )
        ])

        self.transform_pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='coco',
                label_fields=['labels']
            )
        )

    def transform(self, image, bboxes, labels):
        transformed = self.transform_pipeline(
            image=image,
            bboxes=bboxes,
            labels=labels
        )

        return transformed