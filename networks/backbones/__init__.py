from .resnet import ResNet, ResNetv2, Bottleneck
from utils.pyt_utils import load_model
from transformers import CLIPSegForImageSegmentation

def get_backbone(norm_layer, pretrained_model=None, is_restore=False, backbone='resnet101', relu_l3=True, relu_l4=True, **kwargs):
    if backbone == 'resnet101':
        model = ResNet(Bottleneck,[3, 4, 23, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet101')
    elif backbone == 'resnet50':
        model = ResNet(Bottleneck,[3, 4, 6, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet50')
    elif backbone == 'resnet50v2':
        model = ResNetv2(Bottleneck,[3, 4, 6, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet50v2')
    elif backbone == 'resnet101v2':
        model = ResNetv2(Bottleneck,[3, 4, 23, 3], norm_layer=norm_layer, relu_l3=relu_l3, relu_l4=relu_l4, **kwargs)
        print('Backbone:resnet101v2')
    elif backbone == 'clip':
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        print('Backbone:CLIP')

    else:
        raise RuntimeError('unknown backbone: {}'.format(backbone))
    if pretrained_model is not None:
        model = load_model(model, pretrained_model, is_restore)
    return model
