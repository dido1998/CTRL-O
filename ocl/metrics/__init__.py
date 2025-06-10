"""Package for metrics.

The implemetation of metrics are grouped into submodules according to their
datatype and use

 - [ocl.metrics.bbox][]: Metrics for bounding boxes
 - [ocl.metrics.masks][]: Metrics for masks
 - [ocl.metrics.tracking][]: Metrics for multiple object tracking.
 - [ocl.metrics.diagnosis][]: Metrics for diagnosing model training.
 - [ocl.metrics.dataset][]: Metrics that are computed on the whole dataset.

"""

from ocl.metrics.acc import AccMetric, MSEMetric
from ocl.metrics.bbox import BboxCorLocMetric, BboxRecallMetric, UnsupervisedBboxIoUMetric
from ocl.metrics.dataset import DatasetSemanticMaskIoUMetric, SklearnClustering
from ocl.metrics.diagnosis import (
    CategoricalEntropy,
    SlotMaskOccupancy,
    TensorStatistic,
    TwoTensorStatistic,
)
from ocl.metrics.masks import (
    ARIMetric,
    AverageBestOverlapMetric,
    BestOverlapObjectRecoveryMetric,
    BindingHits,
    MaskCorLocMetric,
    PanopticQuality,
    PanopticRecognitionQuality,
    PanopticSegmentationQuality,
    PatchARIMetric,
    UnsupervisedMaskIoUMetric,
)
from ocl.metrics.tracking import MOTMetric

__all__ = [
    "UnsupervisedBboxIoUMetric",
    "BboxCorLocMetric",
    "BboxRecallMetric",
    "DatasetSemanticMaskIoUMetric",
    "SklearnClustering",
    "TensorStatistic",
    "TwoTensorStatistic",
    "SlotMaskOccupancy",
    "CategoricalEntropy",
    "ARIMetric",
    "PatchARIMetric",
    "UnsupervisedMaskIoUMetric",
    "MaskCorLocMetric",
    "AverageBestOverlapMetric",
    "BestOverlapObjectRecoveryMetric",
    "PanopticQuality",
    "PanopticSegmentationQuality",
    "PanopticRecognitionQuality",
    "MOTMetric",
    "BindingHits",
    "AccMetric",
    "MSEMetric",
]
