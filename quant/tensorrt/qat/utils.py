def write_deploy_cfg(
    insize,
    score_threshold,
    confidence_threshold,
    iou_threshold,
    max_box,
    pre_top_k,
    keep_top_k,
    cache_path,
):
    cfg = f"""deploy_cfg = dict(
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['input'],
        output_names=['dets', 'labels'],
        input_shape=None,
        optimize=True,
        dynamic_axes=dict(
            input=dict({{
                0: 'batch',
                2: 'height',
                3: 'width'
            }}),
            dets=dict({{
                0: 'batch',
                1: 'num_dets'
            }}),
            labels=dict({{
                0: 'batch',
                1: 'num_dets'
            }}))),
    codebase_config=dict(
        type='mmdet',
        task='ObjectDetection',
        model_type='end2end',
        post_processing= dict(
            score_threshold={score_threshold},
            confidence_threshold={confidence_threshold},  # for YOLOv3
            iou_threshold={iou_threshold},
            max_output_boxes_per_class={max_box},
            pre_top_k={pre_top_k},
            keep_top_k={keep_top_k},
            background_label_id=-1,
        )),
    backend_config=dict(
        type='tensorrt',
        common_config=dict(
            fp16_mode=False,
            max_workspace_size=1073741824,
            int8_mode=True,
            explicit_quant_mode=True),
        model_inputs=[
            dict(
                input_shapes=dict(
                    input=dict(
                        min_shape=[1, 3, {insize}, {insize}],
                        opt_shape=[1, 3, {insize}, {insize}],
                        max_shape=[1, 3, {insize}, {insize}])))
        ]),
    function_record_to_pop=[
        'mmdet.models.detectors.single_stage.SingleStageDetector.forward',
        'mmdet.models.detectors.two_stage.TwoStageDetector.forward',
        'mmdet.models.detectors.single_stage_instance_seg.SingleStageInstanceSegmentor.forward',  # noqa: E501
        'torch.cat'
    ])"""
    with open(f"{cache_path}/current_base_tensorrt_deploy_config.py", "w") as f:
        f.write(cfg)


def build_quantization_config(
    checkpoint,
    cache_path,
    epochs,
    val_interval,
    opt,
    momentum,
    lr,
    scheduler,
    factor,
    decay,
):
    return f"""_base_ = [
    'modified_cfg.py',
    'current_base_tensorrt_deploy_config.py']

float_checkpoint = '{checkpoint}'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    architecture=_base_.model,
    deploy_cfg=_base_.deploy_cfg,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.TensorRTQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',
            ])))

optim_wrapper = dict(
    optimizer=dict(lr={lr}, momentum={momentum},  type='{opt}', weight_decay={decay}))

# learning policy
param_scheduler = dict(
    _delete_=True, type='{scheduler}', factor={factor}, by_epoch=True)

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.QATEpochBasedLoop',
    max_epochs={epochs},
    val_interval={val_interval})
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))

custom_hooks = []"""


def build_mmdeploy_config(insize, cache_path):
    config = f"""_base_ = ['../_base_/base_static.py', '../../_base_/backends/tensorrt-int8.py']

onnx_config = dict(input_shape=(320, 320))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, {insize}, {insize}],
                    opt_shape=[1, 3, {insize}, {insize}],
                    max_shape=[1, 3, {insize}, {insize}])))
    ])"""
    with open(f"{cache_path}/current_tensorrt_deploy_config.py", "w") as f:
        f.write(config)
