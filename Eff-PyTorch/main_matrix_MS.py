from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image
import time
import os
from datetime import datetime
from timm.utils import *
# 如果使用上面的Git工程的话这样导入
# from efficientnet.model import EfficientNet
# 如果使用pip安装的Efficient的话这样导入
from efficientnet_pytorch import EfficientNet
from signal_matrix_MS import *
from torch.nn import functional as F
from functools import partial
from sklearn.metrics import confusion_matrix
import seaborn as sn  #画图模块
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_random_seed(state=1):
    """
        设定随机种子
    :param state: 随机种子值
    :return: None
    """
    gens = (np.random.seed, torch.manual_seed)
    for set_state in gens:
        set_state(state)


def load_data(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1

    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=4) for x in [set_name]}
    
    return dataset_loaders[set_name]


def train_loop(epoch, dataloader, model, loss_fn, optimizer, device):
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top3_m = AverageMeter()
    batch_time_m = AverageMeter()

    pred_labels = torch.empty(0).to(device)
    labels = torch.empty(0).to(device)

    size = len(dataloader.dataset)
    end = time.time()
    model.train()
    for batch_idx, (signals, targets) in enumerate(dataloader):
        signals0 = signals[0].float().to(device)
        signals1 = signals[1].float().to(device)
        targets = targets.long().to(device)

        with torch.set_grad_enabled(True):
            # Compute prediction and loss
            preds = model(signals0, signals1)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            optimizer.step() 

            acc1, acc3 = accuracy(preds, targets, topk=(1, 3))
            losses_m.update(loss.item(), signals0.size(0))
            top1_m.update(acc1.item(), signals0.size(0))
            top3_m.update(acc3.item(), signals0.size(0))

            pred_labels = torch.cat([pred_labels, torch.argmax(preds, dim=1)], dim=0)
            labels = torch.cat([labels, targets], dim=0)

            batch_time_m.update(time.time() - end)
            end = time.time()

            print(
                'Batch: {batch_idx}  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@3: {top3.val:>7.4f} ({top3.avg:>7.4f})'.format(
                        batch_idx=batch_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top3=top3_m))


def test_loop(epoch, dataloader, model, loss_fn, device):
    """
    模型测试部分
    :param dataloader: 测试数据集
    :param model: 测试模型
    :param loss_fn: 损失函数
    :return: None
    """
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top3_m = AverageMeter()
    batch_time_m = AverageMeter()

    pred_labels = torch.empty(0).to(device)
    labels = torch.empty(0).to(device)
    
    # 用来计算abs-sum. 等于PyTorch L1Loss
    model.eval()
    end = time.time()
    with torch.no_grad(): 
        for batch_idx, (signals, targets) in enumerate(dataloader):
            signals0 = signals[0].float().to(device)
            signals1 = signals[1].float().to(device)
            targets = targets.long().to(device)

            preds = model(signals0, signals1)
            loss = loss_fn(preds, targets)
            acc1, acc3 = accuracy(preds, targets, topk=(1, 3))

            losses_m.update(loss.item(), signals0.size(0))
            top1_m.update(acc1.item(), signals0.size(0))
            top3_m.update(acc3.item(), signals0.size(0))

            pred_labels = torch.cat([pred_labels, torch.argmax(preds, dim=1)], dim=0)
            labels = torch.cat([labels, targets], dim=0)
            
            batch_time_m.update(time.time() - end)
            end = time.time()

            print(
                'Batch: {batch_idx}  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@3: {top3.val:>7.4f} ({top3.avg:>7.4f})'.format(
                        batch_idx=batch_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top3=top3_m))

    plot=plot_matrix(labels.cpu(), pred_labels.cpu(), 'signal confusion matrix')
    plt.savefig('./workers/{}_signal_conf_matrix(vote).png'.format(epoch))
    plt.close()
    
    return top1_m.avg
    

def pred_dataset(weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(weight_path).to(device).eval()
    test_dir = '/home/wanghao/频域数据/signal_python_500B_2of4_split/test'
    test_dir2 = '/home/wanghao/频域数据/signal_python_500B_4of4'
    dataset = dataset_joint(test_dir, test_dir2, test_transforms)    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    loss_fn = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        acc = test_loop(0, loader, net, loss_fn, device)
    print("***Model acc***", acc)


def plot_matrix(y_true, y_pred,title_name):
    cm = confusion_matrix(y_true, y_pred)#混淆矩阵
    #annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sn.heatmap(cm,annot=True,fmt='g',xticklabels=['sig1', 'sig2', 'sig3', 'sig4', 'sig5', 'sig6', 'sig7', 'sig8'],yticklabels=['sig1', 'sig2', 'sig3', 'sig4', 'sig5', 'sig6', 'sig7', 'sig8'], cmap="YlGnBu")
    ax.set_title(title_name) #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴


def pred_img(weight_path, img_path, device):

    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    img = Image.open(img_path).convert('RGB')
    print('Image shape: ', img.size)
    img = transform(img).to(device)
    img = img.unsqueeze(0)
    print('Input shape: ',img.shape)

    net =  torch.load(weight_path).to(device)
    net.eval()
    output = net(img)
    pred_label = torch.argmax(output, dim=1)
    print('**Pred Label**', pred_label.item(), '\n**Pred Prob**', output.tolist())


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


class Mult_Stage(nn.Module):
    def __init__(
            self,
            stage_num,
            hidden_dim,
    ):
        super().__init__()
        self.stage_num = stage_num
        self.hidden_dim = hidden_dim

        self.net1 = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1)
        num_ftrs = self.net1._fc.in_features
        self.net1._fc = nn.Linear(num_ftrs, hidden_dim)

        self.net2 = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1)
        self.net2._fc = nn.Linear(num_ftrs, hidden_dim)
        
        self.vote = nn.Linear(in_features=2, out_features=1)

    def forward(self, x1, x2):

        res1 = self.net1(x1)
        res2 = self.net2(x2)

        labels = torch.cat([res1.unsqueeze(-1), res2.unsqueeze(-1)], dim=-1)

        x = self.vote(labels).squeeze(-1)

        return x

if __name__ == '__main__':
    set_random_seed(2023)

    class_num = 8
    batch_size = 128
    lr = 0.005
    mom = 0.9
    n_epoch = 60
    input_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/wanghao/频域数据/series_signal_B_2000_split'

    # # 自动下载到本地预训练
    # net = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1)

    # # Modify the fc layer based on the number of categories
    # num_ftrs = net._fc.in_features
    # net._fc = nn.Linear(num_ftrs, class_num)

    net = Mult_Stage(stage_num=2, hidden_dim=8)


    net = net.to(device)
    print('Model:', net)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom, weight_decay=0.0004)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Param & Flops
    # from torchprofile import profile_macs
    # net.eval()
    # macs = profile_macs(net, tuple([torch.randn([1, 1, 224, 224]),torch.randn([1, 1, 224, 224])]).to(device))
    # net.train()
    # print('Model Flops:', macs, 'Param Count:', sum([m.numel() for m in net.parameters()]))

    # Dataloaders
    train_dir = '/home/wanghao/频域数据/500B_stft25_vote/4of4_split/train'
    test_dir = '/home/wanghao/频域数据/500B_stft25_vote/4of4_split/test'
    train_dir2 = '/home/wanghao/频域数据/500B_stft25_vote/2of4'
    test_dir2 = '/home/wanghao/频域数据/500B_stft25_vote/2of4'
    train_dataset = dataset_joint(train_dir, train_dir2, train_transforms)
    test_dataset = dataset_joint(test_dir, test_dir2, test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    # train_loader = load_data(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)
    # test_loader = load_data(data_dir=data_dir, batch_size=batch_size, set_name='test', shuffle=False)
    print('Train Size:', len(train_loader.dataset), 'Test Size:', len(test_loader.dataset))
    
    # Train & Test 
    best_acc = 0
    worker_dir = './workers'
    now = datetime.now()  # 获得当前时间
    timestr = now.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(worker_dir, timestr)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epoch):
        # Train & Test 
        print(f"\n----------Epoch {epoch + 1}----------")
        print("\n================Train================")
        train_loop(epoch, train_loader, net, loss_fn, optimizer, device)
        print("\n================Test=================")
        acc = test_loop(epoch, test_loader, net, loss_fn, device)

        # checkpoints
        if acc > best_acc:
            best_acc = acc
            best_net_wts = net.state_dict()
            # 保存
            net.load_state_dict(best_net_wts)
            save_path = os.path.join(save_dir, str(best_acc) + '.pth')
            torch.save(net, save_path)
            print('***Save model*** best_acc:', best_acc, 'save_path:', save_path)
        else:
            print('***Not Save model*** best_acc:', best_acc)



    weight_path = '/home/wanghao/EfficientNet-PyTorch-master/workers/20230903-092418/72.82303370786516.pth'
    pred_dataset(weight_path)



