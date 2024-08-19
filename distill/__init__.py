def initialize_initialization(algoname, task):
    if algoname == "KDTransfer":
        from .responsekd import KDTransfer

        return KDTransfer
    if algoname == "MMRazorDistill":
        if task == "object_detection":
            from .mmrazordistill import MMRazorDistillObjectDetection

            return MMRazorDistillObjectDetection
        elif task == "segmentation":
            from .mmrazordistill import MMRazorDistillSegmentation

            return MMRazorDistillSegmentation
        elif task == "pose_detection":
            from .mmrazordistill import MMRazorDistillPoseEstimation

            return MMRazorDistillPoseEstimation
