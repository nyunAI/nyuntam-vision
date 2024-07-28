from typing import Any


def __getattr__(name: str) -> Any:
    # classification
    if name == "ClassificationDatasetFactory":
        from .classification import DatasetFactory as ClassificationDatasetFactory

        return ClassificationDatasetFactory

    # object detection
    elif name == "ObjectDetectionDatasetFactory":
        from .object_detection import DatasetFactory as ObjectDetectionDatasetFactory

        return ObjectDetectionDatasetFactory

    # segmentation
    elif name == "SegmentationDatasetFactory":
        from .segmentation import DatasetFactory as SegmentationDatasetFactory

        return SegmentationDatasetFactory

    # tracking (commented out in original)
    # elif name == "TrackingDatasetFactory":
    #     from trailmet.datasets.tracking import DatasetFactory as TrackingDatasetFactory
    #     return TrackingDatasetFactory

    # pose estimation
    elif name == "PoseEstimationDatasetFactory":
        from .pose_estimation import DatasetFactory as PoseEstimationDatasetFactory

        return PoseEstimationDatasetFactory

    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
