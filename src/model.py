from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
from effdet.efficientdet import HeadNet
import torch
import gc


def get_model(variant, model_dir, checkpoint_path=None):

    config = get_efficientdet_config(f'tf_efficientdet_{variant}')
    net = EfficientDet(config, pretrained_backbone=False)
    
    if checkpoint_path is None:
        if variant == 'd0':
            checkpoint_path = f'{model_dir}/efficientdet_d0-d92fd44f.pth'
        elif variant == 'd1':
            checkpoint_path = f'{model_dir}/efficientdet_d1-4c7ebaf2.pth'
        elif variant == 'd2':
            checkpoint_path = f'{model_dir}/efficientdet_d2-cb4ce77d.pth'
        elif variant == 'd3':
            checkpoint_path = f'{model_dir}/efficientdet_d3-b0ea2cbc.pth'
        elif variant == 'd4':
            checkpoint_path = f'{model_dir}/efficientdet_d4-5b370b7a.pth'
        elif variant == 'd5':
            checkpoint_path = f'{model_dir}/efficientdet_d5-ef44aea8.pth'
            
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)

    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
        
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])

    return DetBenchTrain(net, config)


def load_model_for_eval(checkpoint_path, variant):
    config = get_efficientdet_config(f'tf_efficientdet_{variant}')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)    
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval();
    
    return net.cuda()