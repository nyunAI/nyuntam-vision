def find_folder(cfg):
    model = cfg.split(".")[0]
    model = model.split("_")[0]
    if "faster-rcnn" in model:
        return "faster_rcnn"
    elif "retinanet" in model:
        return "retinanet"
    elif "atss" in model:
        return "atss"
    elif "autoassign" in model:
        return "autoassign"
    elif "boxinst" in model:
        return "boxinst"
    elif "bytetrack" in model:
        return "bytetrack"
    elif "cascade-rcnn" in model:
        return "cascade_rcnn"
    elif "cascade-rpn" in model:
        return "cascade_rpn"
    elif "centernet" in model:
        return "centernet"
    elif "conditional-detr" in model:
        return "conditional_detr"
    elif "crowddet" in model:
        return "crowddet"
    elif "dab_detr" in model:
        return "dab_detr"
    elif "dcn" in model:
        return "dcn"
    elif "dcnv2" in model:
        return "dcnv2"
    elif "ddod" in model:
        return "ddod"
    elif "ddq" in model:
        return "ddq"
    elif "deepfashion" in model:
        return "deepfashion"
    elif "deepsort" in model:
        return "deepsort"
    elif "deformable-detr" in model:
        return "deformable_detr"
    elif "detectors" in model:
        return "detectors"
    elif "detr" in model:
        return "detr"

    elif "double-heads" in model:
        return "double_heads"
    elif "dsdl" in model:
        return "dsdl"
    elif "dyhead" in model:
        return "dyhead"
    elif "dynamic-rcnn" in model:
        return "dynamic_rcnn"
    elif "effcientnet" in model:
        return "efficientnet"
    elif "empirical-attention" in model:
        return "empirical_attention"
    elif "fast-rcnn" in model:
        return "fast_rcnn"
    elif "fcos" in model:
        return "fcos"
    elif "foveabox" in model:
        return "foveabox"
    elif "fpg" in model:
        return "fpg"
    elif "free-anchor" in model:
        return "free_anchor"
    elif "fsaf" in model:
        return "fsaf"
    elif "gcnet" in model:
        return "gcnet"
    elif "gfl" in model:
        return "gfl"
    elif "ghm" in model:
        return "ghm"
    elif "glip" in model:
        return "glip"
    elif "gn+ws" in model:
        return "gn+ws"
    elif "grid-rcnn" in model:
        return "grid_rcnn"
    elif "groie" in model:
        return "groie"
    elif "grounding-dino" in model:
        return "grounding_dino"
    elif "guided-anchoring" in model:
        return "guided_anchoring"
    elif "hrnet" in model:
        return "hrnet"
    elif "htc" in model:
        return "htc"
    elif "instaboost" in model:
        return "instaboost"
    elif "lad" in model:
        return "lad"
    elif "ld" in model:
        return "ld"
    elif "legacy-1" in model:
        return "legacy_1.x"
    elif "libra-rcnn" in model:
        return "libra_rcnn"
    elif "lvis" in model:
        return "lvis"
    elif "mask2former" in model:
        return "mask2former"
    elif "mask2former-vis" in model:
        return "mask2former_vis"
    elif "mask-rcnn" in model:
        return "mask_rcnn"
    elif "maskformer" in model:
        return "maskformer"
    elif "masktrack-rcnn" in model:
        return "masktrack_rcnn"
    elif "misc" in model:
        return "misc"
    elif "mm-grounding-dino" in model:
        return "mm_grounding_dino"
    elif "ms-rcnn" in model:
        return "ms_rcnn"
    elif "nas-fcos" in model:
        return "nas_fcos"
    elif "nas-fpn" in model:
        return "nas_fpn"
    elif "objects365" in model:
        return "objects365"
    elif "ocsort" in model:
        return "ocsort"
    elif "openimages" in model:
        return "openimages"
    elif "paa" in model:
        return "paa"
    elif "pafpn" in model:
        return "pafpn"
    elif "panoptic-fpn" in model:
        return "panoptic_fpn"
    elif "pascal-voc" in model:
        return "pascal_voc"
    elif "pisa" in model:
        return "pisa"
    elif "point-rend" in model:
        return "point_rend"
    elif "pvt" in model:
        return "pvt"
    elif "qdtrack" in model:
        return "qdtrack"
    elif "queryinst" in model:
        return "queryinst"
    elif "regnet" in model:
        return "regnet"
    elif "reid" in model:
        return "reid"
    elif "reppoints" in model:
        return "reppoints"
    elif "res2net" in model:
        return "res2net"
    elif "rpn" in model:
        return "rpn"
    elif "rtmdet" in model:
        return "rtmdet"
    elif "sabl" in model:
        return "sabl"
    elif "scnet" in model:
        return "scnet"
    elif "seesaw-loss" in model:
        return "seesaw_loss"
    elif "selfsup-pretrain" in model:
        return "selfsup_pretrain"
    elif "simple-copy-paste" in model:
        return "simple_copy_paste"
    elif "soft-teacher" in model:
        return "soft_teacher"
    elif "solo" in model:
        return "solo"
    elif "solov2" in model:
        return "solov2"
    elif "sort" in model:
        return "sort"
    elif "sparse-rcnn" in model:
        return "sparse_rcnn"
    elif "ssd" in model:
        return "ssd"
    elif "strong-baselines" in model:
        return "strong_baselines"
    elif "strongsort" in model:
        return "strongsort"
    elif "tood" in model:
        return "tood"
    elif "tridentnet" in model:
        return "tridentnet"
    elif "v3det" in model:
        return "v3det"
    elif "vfnet" in model:
        return "vfnet"
    elif "wider-face" in model:
        return "wider_face"
    elif "yoloact" in model:
        return "yoloact"
    elif "yolof" in model:
        return "yolof"
    elif "yolox" in model:
        return "yolox"
    elif "yolo" in model:
        return "yolo"
    else:
        raise ValueError(f"folder not found for {model}")


def return_cwd_config(teacher_config, teacher_pth, student_config):
    folder = find_folder(student_config)
    teacher_folder = find_folder(teacher_config)
    return f"""
_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# default_scope = 'mmrazor'
teacher_ckpt = '{teacher_pth}'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path= 'mmdet::{folder}/{student_config}',
        pretrained=False),
    teacher=dict(
        cfg_path= 'mmdet::{teacher_folder}/{teacher_config}',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_cwd_fpn0=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn1=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn2=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn3=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn4=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10)),
        loss_forward_mappings=dict(
            loss_cwd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_cwd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_cwd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_cwd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=3)),
            loss_cwd_fpn4=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=4),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=4)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

"""


def return_pkd_config(teacher_config, teacher_pth, student_config):
    folder = find_folder(student_config)
    teacher_folder = find_folder(teacher_config)
    return f"""_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = '{teacher_pth}'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::{folder}/{student_config}',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::{teacher_folder}/{teacher_config}',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=6, resize_stu=True)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')"""


def find_seg_folder(cfg):
    model = cfg.split(".")[0]
    model = model.split("_")[0]
    if "ann" in model:
        return "ann"
    elif "apcnet" in model:
        return "apcnet"
    elif "beit" in model:
        return "beit"
    elif "bisenetv1" in model:
        return "bisenetv1"
    elif "bisenetv2" in model:
        return "bisenetv2"
    elif "ccnet" in model:
        return "ccnet"
    elif "cgnet" in model:
        return "cgnet"
    elif "convnext" in model:
        return "convnext"
    elif "danet" in model:
        return "danet"
    elif "ddrnet" in model:
        return "ddrnet"
    elif "deeplabv3" in model:
        return "deeplabv3"
    elif "deeplabv3plus" in model:
        return "deeplabv3plus"
    elif "dmnet" in model:
        return "dmnet"
    elif "dnlnet" in model:
        return "dnlnet"
    elif "dpt" in model:
        return "dpt"
    elif "dsdl" in model:
        return "dsdl"
    elif "emanet" in model:
        return "emanet"
    elif "encnet" in model:
        return "encnet"
    elif "erfnet" in model:
        return "erfnet"
    elif "fastfcn" in model:
        return "fastfcn"
    elif "fastscnn" in model:
        return "fastscnn"
    elif "fcn" in model:
        return "fcn"
    elif "gcnet" in model:
        return "gcnet"
    elif "hrnet" in model:
        return "hrnet"
    elif "icnet" in model:
        return "icnet"
    elif "isanet" in model:
        return "isanet"
    elif "knet" in model:
        return "knet"
    elif "mae" in model:
        return "mae"
    elif "mask2former" in model:
        return "mask2former"
    elif "mobilenet-v2" in model:
        return "mobilenet_v2"
    elif "mobilenet-v3" in model:
        return "mobilenet_v3"
    elif "nonlocal-net" in model:
        return "nonlocal_net"
    elif "ocrnet" in model:
        return "ocrnet"
    elif "pidnet" in model:
        return "pidnet"
    elif "point_rend" in model:
        return "point_rend"
    elif "poolformer" in model:
        return "poolformer"
    elif "psanet" in model:
        return "psanet"
    elif "pspnet" in model:
        return "pspnet"
    elif "resnest" in model:
        return "resnest"
    elif "san" in model:
        return "san"
    elif "segformer" in model:
        return "segformer"
    elif "segmenter" in model:
        return "segmenter"
    elif "segnext" in model:
        return "segnext"
    elif "sem-fpn" in model:
        return "sem_fpn"
    elif "setr" in model:
        return "setr"
    elif "stdc" in model:
        return "stdc"
    elif "swin" in model:
        return "swin"
    elif "twins" in model:
        return "twins"
    elif "unet" in model:
        return "unet"
    elif "upernet" in model:
        return "upernet"
    elif "vit" in model:
        return "vit"
    elif "vpd" in model:
        return "vpd"
    else:
        raise ValueError(f"folder not found for {model}")


def return_pkd_seg_config(teacher_config, teacher_pth, student_config):
    folder = find_folder(student_config)
    teacher_folder = find_folder(teacher_config)
    return f"""_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]
_base_.val_evaluator['metric'] = ['bbox','segm']
_base_.test_evaluator['metric'] = ['bbox','segm']
teacher_ckpt = '{teacher_pth}'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::{folder}/{student_config}',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::{teacher_folder}/{teacher_config}',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=6, resize_stu=True)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)))))

find_unused_parameters = True

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
"""


def find_folder_mmpose(cfg):
    model = cfg.split(".")[0]
    model = model.split("_")[0]
    if "associative-embedding" in model:
        return "associative_embedding"
    elif "cid" in model:
        return "cid"
    elif "dekr" in model:
        return "dekr"
    elif "edpose" in model:
        return "edpose"
    elif "integral-regression" in model:
        return "integral_regression"
    elif "rtmo" in model:
        return "rtmo"
    elif "rtmpose" in model:
        return "rtmpose"
    elif "simcc" in model:
        return "simcc"
    elif "topdown-heatmap" in model:
        return "topdown_heatmap"
    elif "topdown-regression" in model:
        return "topdown_regression"
    elif "yoloxpose" in model:
        return "yoloxpose"
    else:
        raise ValueError(f"folder not found for {model}")


def return_pkd_pose_config(teacher_config, teacher_pth, student_config):
    folder = find_folder_mmpose(student_config)
    teacher_folder = find_folder_mmpose(teacher_config)
    return f"""_base_ = [
    'mmpose::_base_/default_runtime.py'
]
train_cfg = dict(max_epochs=600, val_interval=20, dynamic_intervals=[(580, 1)])
teacher_ckpt = '{teacher_pth}'  # noqa: E501
metafile = 'mmpose::_base_/datasets/coco.py'
input_size = (640, 640)
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({{'neck.encoder': dict(lr_mult=0.05)}})),
    clip_grad=dict(max_norm=0.1, norm_type=2))
model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmpose::body_2d_keypoint/{folder}/coco/{student_config}',
        pretrained=False),
    teacher=dict(
        cfg_path='mmpose::body_2d_keypoint/{teacher_folder}/coco/{teacher_config}',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=6, resize_stu=True)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)))))
                
train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
data_mode = 'bottomup'
data_root = 'data/'
# train datasets
dataset_coco = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=train_pipeline_stage1,
)
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_coco)
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader
# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator
find_unused_parameters = True
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
"""


def find_folder_mmyolo(cfg):
    model = cfg.split(".")[0]
    model = model.split("_")[0]
    if "yolov5" in model:
        return "yolov5"
    elif "yolov6" in model:
        return "yolov6"
    elif "yolov7" in model:
        return "yolov7"
    elif "yolov7" in model:
        return "yolov7"
    elif "yolov8" in model:
        return "yolov8"
    elif "yolox" in model:
        return "yolox"
    elif "rtmdet" in model:
        return "rtmdet"
    else:
        raise ValueError(f"folder not found for {model}")


def return_pkd_mmyolo_config(teacher_config, teacher_pth, student_config):
    folder = find_folder_mmyolo(student_config)
    teacher_folder = find_folder_mmyolo(teacher_config)
    return f"""default_scope = 'mmyolo'
base_ = ['mmyolo::_base_/default_runtime.py', 'mmyolo::_base_/det_p5_tta.py']
teacher_ckpt = '{teacher_pth}'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::{folder}/{student_config}',
        pretrained=False),
    teacher=dict(
        cfg_path='mmyolo::{teacher_folder}/{teacher_config}',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=6, resize_stu=True),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=6, resize_stu=True)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)))))


find_unused_parameters = True

data_root = 'data/root/'
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'  
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'
train_batch_size_per_gpu = 32
train_num_workers = 10
persistent_workers = True
img_scale = (640, 640) 
random_resize_ratio_range = (0.1, 2.0)
mosaic_max_cached_images = 40
mixup_max_cached_images = 20
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 32
val_num_workers = 10
save_checkpoint_intervals = 10
max_epochs = 300
num_epochs_stage2 = 20
val_interval_stage2 = 1
base_lr = 0.004
lr_start_factor = 1.0e-5
dsl_topk = 13
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0
weight_decay = 0.05
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmyolo',
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals,
    dynamic_intervals=[(max_epochs - num_epochs_stage2, val_interval_stage2)])
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
test_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
"""
