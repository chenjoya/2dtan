"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "tacos_train":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/train.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_val":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/val.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "video_dir": "tacos/videos",
            "ann_file": "tacos/annotations/test.json",
            "feat_file": "tacos/features/tall_c3d_features.hdf5",
        },
        "activitynet_train":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/train.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },
        "activitynet_val":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/val.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },
        "activitynet_test":{
            "video_dir": "activitynet1.3/videos",
            "ann_file": "activitynet1.3/annotations/test.json",
            "feat_file": "activitynet1.3/features/activitynet_v1-3_c3d.hdf5",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["video_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
