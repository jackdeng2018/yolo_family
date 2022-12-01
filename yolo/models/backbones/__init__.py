from .resnet import resnet18, resnet50, resnet101, resnet152


def build_backbone(model_name='r18', pretrained=False, freeze=None, img_size=224):
    if model_name == 'r18':
        print('Backbone: ResNet-18 ...')
        model = resnet18(pretrained=pretrained)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'r50':
        print('Backbone: ResNet-50 ...')
        model = resnet50(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'r101':
        print('Backbone: ResNet-101 ...')
        model = resnet101(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'r152':
        print('Backbone: ResNet-152 ...')
        model = resnet152(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]

    return model, feature_channels, strides
