from loss.criterion import CELoss, OrthLoss
from loss.atm_loss import SegLossPlus

def get_loss(args):
    if 'ekt' in args.model:
        criterion = OrthLoss(ignore_index=args.ignore_label)
    else:
        criterion = CELoss(ignore_index=args.ignore_label)
        # criterion = SegLossPlus(num_classes=16, dec_layers=3, mask_weight=100.0, dice_weight=1.0, loss_weight=1.0)
    
    # criterion = CELoss(ignore_index=args.ignore_label)
    return criterion