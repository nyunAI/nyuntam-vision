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
    '{cache_path}/modified_cfg.py',
    'current_base_openvino_deploy_config.py']

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
        type='mmrazor.OpenVINOQuantizer',
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
