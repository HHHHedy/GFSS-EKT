import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from networks.pspnet_ekt import GFSS_Model

def plot_weights_distribution(model, save_path='weight_distribution.png'):
    # get weights of all layers
    para_list = {}
    for name, param in model.named_parameters():
        para_list[name] = param

    std_n = torch.std(para_list['classifier_n.4.weight'], dim=1, keepdim=True)
    std_b = torch.std(para_list['classifier.4.weight'], dim=1, keepdim=True)
    print('std_n:', std_n)
    print('std_b:', std_b)
    print('alpha:', para_list['bias_linear.alpha'])
    weights = []
    for name, param in model.named_parameters():
        print(f'{name}: {param.shape}')
        if name == 'classifier.4.weight':
            # save_path = save_path.replace('.png', '_base.png')
            
            param = param
            weights.append(param.data.flatten().cpu().numpy())
        elif name == 'classifier_n.4.weight':
            # save_path = save_path.replace('.png', '_novel.png')
            factor = para_list['bias_linear.alpha']
            param = param.detach() * factor
            weights.append(param.data.flatten().cpu().numpy())

    # weights = [param.data.flatten().cpu().numpy() for param in model.classifier[4].weight]

    # drawing
    plt.figure(figsize=(10, 6))
    plt.title('Weight Distribution of Classifier')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.grid(True)

    '''for i, w in enumerate(weights):
        # plt.hist(w, bins=500, alpha=0.5, label=f'Layer {i+1}')
        sns.kdeplot(w, fill=True, label=f'Layer {i+1}')'''
    sns.kdeplot(weights[0], fill=True, label='Base', common_norm=True)
    sns.kdeplot(weights[1], fill=True, label='Novel', common_norm=True)

    plt.legend()
    plt.savefig(save_path)
    print(f'Weight distribution saved as {save_path}')

def plot_weight_norm(model, save_path='weight_norm.png', bar_width=0.2, spacing=0.1):
    # get weights of all layers
    para_list = {}
    for name, param in model.named_parameters():
        para_list[name] = param

    # std_n = torch.std(para_list['classifier_n.4.weight'], dim=1, keepdim=True)
    # std_b = torch.std(para_list['classifier.4.weight'], dim=1, keepdim=True)
    # print('std_n:', std_n)
    # print('std_b:', std_b)
    # print('alpha:', para_list['bias_linear.alpha'])
    
    l2_norms = []
    for name, param in model.named_parameters():
        print(f'{name}: {param.shape}')
        if name == 'classifier.4.weight':
            l2_norm = torch.norm(param, p=2).item()
            l2_norms.append(l2_norm)
        elif name == 'classifier_n.4.weight':
            l2_norm = torch.norm(param, p=2).item()
            l2_norms.append(l2_norm)

    # weights = [param.data.flatten().cpu().numpy() for param in model.classifier[4].weight]
    print('norm:',l2_norms)
    # drawing
    positions = [0.1, 0.1 + bar_width + spacing]
    plt.figure(figsize=(8, 6))
    plt.bar(positions, l2_norms, width=bar_width, alpha=0.5, color=['skyblue', 'salmon'])
    plt.title('Total Weight L2 Norm of Classifier Layers')
    plt.xlabel('Layer Type')
    plt.ylabel('L2 Norm')
    plt.xticks([p + bar_width / 2 for p in positions], ['Base', 'Novel'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, max(l2_norms) * 1.1)  # Adding some space above the tallest bar for clarity

    '''for i, w in enumerate(weights):
        # plt.hist(w, bins=500, alpha=0.5, label=f'Layer {i+1}')
        sns.kdeplot(w, fill=True, label=f'Layer {i+1}')'''
    plt.savefig(save_path)
    print(f'Weight norm saved as {save_path}')

if __name__ == '__main__':
    # create model instance
    model = GFSS_Model(n_base=15, fold=1, backbone='resnet50v2', dilated=True, os=8, n_novel=5, is_ft=True)
    model_path = "./exp/voc/pspnet_ekt/0/resnet50v2/Novel/only_rescale/1/best_123.pth"
    checkpoint = torch.load(model_path)
    if 'module.' in list(checkpoint.keys())[0]:
        # 去除键的前缀
        new_state_dict = {k[7:]: v for k, v in checkpoint.items()}
        # 加载去除前缀的状态字典
        model.load_state_dict(new_state_dict)
    else:
        # 如果没有前缀，直接加载状态字典
        model.load_state_dict(checkpoint)
    plot_weights_distribution(model, save_path='weight_distribution_test.png')
    plot_weight_norm(model, save_path='weight_norm_test.png')
