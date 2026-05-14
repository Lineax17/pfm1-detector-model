from src.helpers.coco_merger import CocoMerger


TRAIN_FOLDERS = [
    "vid-mine-obv-1",
    "vid-mine-obv-2",
    "vid-mine-obv-5",
    "vid-other-1"
]

VAL_FOLDERS = [
    "vid-mine-obv-3",
]

TEST_FOLDERS = [
    "vid-mine-obv-4",
]


def main():

    merger = CocoMerger()

    merger.merge(
        folders=TRAIN_FOLDERS,
        output_folder="train"
    )

    merger.merge(
        folders=VAL_FOLDERS,
        output_folder="val"
    )

    merger.merge(
        folders=TEST_FOLDERS,
        output_folder="test"
    )


if __name__ == "__main__":
    main()