def register_custom_timm_models():
    from .vanillanet import VanillaNet
    from timm.models.registry import register_model

    @register_model
    def vanillanet_5(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[128 * 4, 256 * 4, 512 * 4, 1024 * 4], strides=[2, 2, 2], **kwargs
        )
        return model

    @register_model
    def vanillanet_6(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4],
            strides=[2, 2, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_7(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4],
            strides=[1, 2, 2, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
            strides=[1, 2, 2, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_9(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 4,
                128 * 4,
                256 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                1024 * 4,
                1024 * 4,
            ],
            strides=[1, 2, 2, 1, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_10(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 4,
                128 * 4,
                256 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                1024 * 4,
                1024 * 4,
            ],
            strides=[1, 2, 2, 1, 1, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_11(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 4,
                128 * 4,
                256 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                1024 * 4,
                1024 * 4,
            ],
            strides=[1, 2, 2, 1, 1, 1, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_12(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 4,
                128 * 4,
                256 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                1024 * 4,
                1024 * 4,
            ],
            strides=[1, 2, 2, 1, 1, 1, 1, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_13(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 4,
                128 * 4,
                256 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                512 * 4,
                1024 * 4,
                1024 * 4,
            ],
            strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_13_x1_5(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 6,
                128 * 6,
                256 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                1024 * 6,
                1024 * 6,
            ],
            strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
            **kwargs
        )
        return model

    @register_model
    def vanillanet_13_x1_5_ada_pool(pretrained=False, in_22k=False, **kwargs):
        model = VanillaNet(
            dims=[
                128 * 6,
                128 * 6,
                256 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                512 * 6,
                1024 * 6,
                1024 * 6,
            ],
            strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
            ada_pool=[0, 38, 19, 0, 0, 0, 0, 0, 0, 10, 0],
            **kwargs
        )
        return model
