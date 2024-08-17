def make_deploy_config(insize, cache_path):
    config = f"""_base_ = [
    'vision/core/utils/mmdeployconfigs/mmdet/_base_/base_dynamic.py', 'vision/core/utils/mmdeployconfigs/_base_/backends/tensorrt-int8.py']

backend_config = dict(
        common_config=dict(max_workspace_size=1 << 30),
        model_inputs=[
            dict(
                input_shapes=dict(
                    input=dict(
                        min_shape=[1, 3, 64, 64],
                        opt_shape=[1, 3, {insize}, {insize}],
                        max_shape=[1, 3, {insize}, {insize}])))
        ])"""
    with open(f"{cache_path}/current_tensorrt_quant_config.py", "w") as f:
        f.write(config)


def make_deploy_config_mmyolo(insize, cache_path):
    config = f"""_base_ = ['mmyolo::deploy/base_static.py']
onnx_config = dict(input_shape=(640, 640))
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True, max_workspace_size=1 << 30, int8_mode=True),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, {insize}, {insize}],
                    opt_shape=[1, 3, {insize}, {insize}],
                    max_shape=[1, 3, {insize}, {insize}])))
    ])
calib_config = dict(create_calib=True, calib_file='calib_data.h5')
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
"""
    with open(f"{cache_path}/current_tensorrt_quant_config.py", "w") as f:
        f.write(config)


def create_demo_img(dataloader):
    from torchvision.utils import save_image

    x, y = next(iter(dataloader))
    x = x[0]
    save_image(x, "demo_img.png")


def write_modified_test_loader_config(path, cache_path):
    cfg = f"""_base_ = '{path}'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=False,
        use_mini_pad=False,
    ),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, batch_shapes_cfg=None))"""
    with open(f"{cache_path}/modified_pretrain_cfg_tensorrt_quantization.py", "w") as f:
        f.write(cfg)
