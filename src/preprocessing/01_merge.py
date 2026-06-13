from src.helpers.coco_merger import CocoMerger


TRAIN_FOLDERS = [
    "vid-mine-obv-1",
    "vid-mine-obv-3",
    "vid-mine-obv-4",
    "vid-mine-obv-5",
    "vid-mine-obv-6",
    "vid-mine-obv-7",
    "vid-mine-obv-8",
    "vid-mine-obv-10",
    "vid-mine-obv-11",
    "vid-mine-obv-12",
    "vid-mine-obv-14",
    "vid-mine-obv-16",
    "vid-mine-obv-17",
    "vid-mine-obv-18",
    "vid-mine-obv-19",
    "vid-mine-obv-20",
    "vid-mine-obv-21",
    "vid-mine-obv-22",
    "vid-mine-obv-23",
    "vid-mine-obv-24",
    "vid-mine-obv-25",
    "vid-mine-obv-26",
    "vid-mine-obv-27",
    "vid-mine-obv-29",
    "vid-mine-obv-31",
    "vid-mine-obv-32",
    "vid-mine-obv-33",
    "vid-other-1",
    "vid-other-2",
    "vid-other-3",
    "vid-other-6",
    "vid-other-8",
    "vid-other-9"
]

VAL_FOLDERS = [
    "vid-mine-obv-2",
    "vid-mine-obv-9",
    "vid-mine-obv-13",
    "vid-mine-obv-15",
    "vid-mine-obv-28",
    "vid-mine-obv-30",
    "vid-other-4",
    "vid-other-5",
    "vid-other-7"
]

TEST_FOLDERS = [
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