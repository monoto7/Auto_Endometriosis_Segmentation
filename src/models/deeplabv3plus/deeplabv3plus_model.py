import segmentation_models_pytorch as smp


def build_deeplabv3plus_model(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation=None
):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )

    return model