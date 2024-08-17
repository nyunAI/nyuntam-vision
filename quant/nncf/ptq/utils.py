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
        input_shape=None,
        input_names=['input'],
        output_names=['dets', 'labels'],
        optimize=True,
        dynamic_axes={{
            'input': {{
                0: 'batch',
                2: 'height',
                3: 'width'
            }},
            'dets': {{
                0: 'batch',
                1: 'num_dets',
            }},
            'labels': {{
                0: 'batch',
                1: 'num_dets',
            }},
        }}),
    backend_config=dict(
        type='openvino',
        model_inputs=[dict(opt_shapes=dict(input=[1, 3, {insize}, {insize}]))]),
    codebase_config=dict(
        type='mmdet',
        task='ObjectDetection',
        model_type='end2end',
        post_processing=dict(
            score_threshold={score_threshold},
            confidence_threshold={confidence_threshold},  # for YOLOv3
            iou_threshold={iou_threshold},
            max_output_boxes_per_class={max_box},
            pre_top_k={pre_top_k},
            keep_top_k={keep_top_k},
            background_label_id=-1,
        )),
    function_record_to_pop=[
        'mmdet.models.detectors.single_stage.SingleStageDetector.forward',
        'mmdet.models.detectors.two_stage.TwoStageDetector.forward',
        'mmdet.models.detectors.single_stage_instance_seg.SingleStageInstanceSegmentor.forward'  # noqa: E501
    ])"""
    with open(f"{cache_path}/current_base_openvino_deploy_config.py", "w") as f:
        f.write(cfg)


def build_quantization_config(
    checkpoint,
    cache_path,
):
    return f"""_base_ = [
    '{cache_path}/modified_cfg.py',
    'current_base_openvino_deploy_config.py']

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=_base_.val_dataloader,
    calibrate_steps=32,
)

float_checkpoint = '{checkpoint}'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
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
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',

            ])))

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

custom_hooks = []
"""


def build_mmdeploy_config(insize, cache_path):
    config = f"""_base_ = ['vision/core/utils/mmdeployconfigs/mmdet/_base_/base_dynamic.py', 'vision/core/utils/mmdeployconfigs/_base_/backends/openvino.py']

onnx_config = dict(input_shape=None)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, {insize}, {insize}]))])
"""
    with open(f"{cache_path}/current_openvino_deploy_config.py", "w") as f:
        f.write(config)
