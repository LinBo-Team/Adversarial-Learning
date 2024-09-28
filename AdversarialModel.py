#  Supplementary adversarial training experiments
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from datautil import loadmat
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def our_args():
    parser = argparse.ArgumentParser(description='Domain invariant representation learning strategy')
    parser.add_argument('--num_classes', type=int, default=4, help='padeborn:6;CEFL:4')
    parser.add_argument('--batchSize', type=int, default=256, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--normalization', type=bool, default=False, help='normalization')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bottleneck', type=int, default=256, help='bottleneck')
    parser.add_argument('--lambda_grl', type=float, default=0.2, help='lambda_grl')
    args = parser.parse_args()
    return args


# 特征提取器
class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
        self.featurizer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = self.featurizer(x)
        return x.view(x.size(0), -1)


# 瓶颈层
class Bottleneck(nn.Module):
    def __init__(self, fea_num):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Linear(128 * 318, fea_num)

    def forward(self, x):
        return self.bottleneck(x)


# 分类器
class Classifier(nn.Module):
    def __init__(self, fea_num, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(fea_num, num_classes)

    def forward(self, x):
        return self.classifier(x)


# 域判别器（用于对抗训练）
class DomainDiscriminator(nn.Module):
    def __init__(self, fea_num):
        super(DomainDiscriminator, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(fea_num, 100),
            nn.ReLU(),
            nn.Linear(100, 3)  # 3 classes: source domain
        )

    def forward(self, x):
        return self.domain_classifier(x)


# 梯度反转层
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None


# 定义对抗学习模型
class DomainAdversarialNetwork(nn.Module):
    def __init__(self, fea_num, num_classes):
        super(DomainAdversarialNetwork, self).__init__()
        self.featurizer = Featurizer()
        self.bottleneck = Bottleneck(fea_num)
        self.classifier = Classifier(fea_num, num_classes)
        self.domain_discriminator = DomainDiscriminator(fea_num)

    def forward(self, x, lambda_grl=1.0):
        features = self.featurizer(x)
        features = self.bottleneck(features)
        class_outputs = self.classifier(features)
        domain_features = GradientReversalLayer.apply(features, lambda_grl)
        domain_outputs = self.domain_discriminator(domain_features)
        return class_outputs, domain_outputs


# 模型训练
def modeltrain(source1_batch, source2_batch, source3_batch):
    for epoch in range(30):  # 100个epoch
        model.train()
        for batch_idx, ((source1_data, source1_labels), (source2_data, source2_labels), (source3_data, source3_labels)) in enumerate(zip(source1_batch, source2_batch, source3_batch)):
            # 前向传播
            class1_outputs, domain1_outputs_source = model(torch.unsqueeze(source1_data, dim=1), lambda_grl)
            class2_outputs, domain2_outputs_source = model(torch.unsqueeze(source2_data, dim=1), lambda_grl)
            class3_outputs, domain3_outputs_source = model(torch.unsqueeze(source2_data, dim=1), lambda_grl)

            # 计算分类损失
            class1_loss = classification_loss_fn(class1_outputs, torch.squeeze(source1_labels-1))
            class2_loss = classification_loss_fn(class2_outputs, torch.squeeze(source2_labels - 1))
            class3_loss = classification_loss_fn(class3_outputs, torch.squeeze(source3_labels - 1))
            class_loss = (class1_loss+class2_loss+class3_loss)/3

            # 计算域判别器损失
            domain1_loss_source = domain_loss_fn(domain1_outputs_source, source1_domain_labels)
            domain2_loss_source = domain_loss_fn(domain2_outputs_source, source2_domain_labels)
            domain3_loss_source = domain_loss_fn(domain3_outputs_source, source3_domain_labels)
            domain_loss = (domain1_loss_source + domain2_loss_source+domain3_loss_source) / 3

            # 总损失
            total_loss = class_loss + domain_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # print(f'Epoch {epoch + 1}: Total Loss: {total_loss.item():.4f}')
        total_loss_history.append(total_loss.item())
        class_loss_history.append(class_loss.item())
        adversarial_loss_history.append(domain_loss.item())
    return model


def modeltest(model, testloader):
    model.eval()  # 进入评估模式
    for batch1_idx, (target1_data, target1_labels) in enumerate(testloader):
        with torch.no_grad():  # 禁用梯度计算
            target_class_outputs, _ = model(torch.unsqueeze(target1_data, dim=1))
            _, predicted = torch.max(target_class_outputs, 1)

    accuracy = accuracy_score(torch.squeeze(target1_labels-1).numpy(), predicted.numpy())
    f1 = f1_score(torch.squeeze(target1_labels-1).numpy(), predicted.numpy(), average='weighted')

    print(f"Accuracy on target domain: {accuracy:.4f}")
    print(f"F1 Score on target domain: {f1:.4f}")

    return predicted


def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 创建数据集和数据加载器
def create_dataset(path, Normalize):
    # 创建数据集和数据加载器
    dataset = loadmat.TrainDataset(path, Normalize)
    return dataset


# 定义双Y轴绘图函数
def plot_losses_with_dual_axis(total_loss_history, class_loss_history, domain_loss_history):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制总损失和对抗损失（共享一个y轴）
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total/Domain Loss', color='blue')
    l1, = ax1.plot(total_loss_history, label='Total Loss', color='blue', linewidth=2)
    l2, = ax1.plot(domain_loss_history, label='Domain Loss', color='red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    # 创建另一个y轴，绘制分类损失
    ax2 = ax1.twinx()
    ax2.set_ylabel('Class Loss', color='green')
    l3, = ax2.plot(class_loss_history, label='Class Loss', color='green', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='green')

    # 添加图例，将两个轴的曲线图例合并
    fig.legend([l1, l2, l3], ['Total Loss', 'Domain Loss', 'Class Loss'], loc="upper right")

    # 添加标题
    fig.tight_layout()
    plt.title('Loss Curves with Dual Y-axis and Legends')
    plt.show()


if __name__ == "__main__":

    # 固定随机种子
    set_random_seed()
    # 导入参数
    args = our_args()
    # 超参数设置
    fea_num = args.bottleneck  # 特征数量
    num_classes = args.num_classes  # 类别数量
    lambda_grl = args.lambda_grl  # 梯度反转层的lambda

    # 模拟训练过程的 loss 数据存储
    total_loss_history = []
    class_loss_history = []
    adversarial_loss_history = []

    # 初始化模型
    model = DomainAdversarialNetwork(fea_num, num_classes)

    # 定义损失函数和优化器
    classification_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 创建数据加载器
    dataset1 = create_dataset('.\Labbearing/condition3.mat', Normalize=args.normalization)
    trainloader1 = DataLoader(dataset1, batch_size=args.batchSize, shuffle=args.shuffle, drop_last=True)
    dataset2 = create_dataset('.\Labbearing/condition4.mat', Normalize=args.normalization)
    trainloader2 = DataLoader(dataset2, batch_size=args.batchSize, shuffle=args.shuffle, drop_last=True)
    dataset3 = create_dataset('.\Labbearing/condition1.mat', Normalize=args.normalization)
    trainloader3 = DataLoader(dataset3, batch_size=args.batchSize, shuffle=args.shuffle, drop_last=True)
    dataset4 = create_dataset('.\Labbearing/condition2.mat', Normalize=args.normalization)

    trainset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])
    trainlaoder = DataLoader(trainset, batch_size=args.batchSize, shuffle=False, drop_last=True)

    testloader = DataLoader(dataset4, batch_size=2000, shuffle=True, drop_last=True)


    # # 获取源域数据、标签、域标签
    source1_domain_labels = torch.zeros(args.batchSize).long()  # 源域标签为0

    # # 获取目标域数据、域标签（无分类标签）
    # target_data = torch.randn(32, 1, 2560)  # 示例目标域数据
    source2_domain_labels = torch.ones(args.batchSize).long()  # 目标域标签为1
    source3_domain_labels = 2 * torch.ones(args.batchSize).long()  # 目标域标签为1
    model = modeltrain(trainloader1, trainloader2, trainloader3)

    # 假设目标域有测试标签 target_test_labels
    modeltest(model, testloader)

    plot_losses_with_dual_axis(total_loss_history, class_loss_history, adversarial_loss_history)
    loss_data = {
        "Epoch": list(range(1, len(total_loss_history) + 1)),
        "Total Loss": total_loss_history,
        "adversarial Loss": adversarial_loss_history,
        "Class Loss": class_loss_history
    }

    # 将数据转换为DataFrame
    loss_df = pd.DataFrame(loss_data)

    # 保存为CSV文件
    csv_file_path = './output/loss_history.csv'
    loss_df.to_csv(csv_file_path, index=False)

    csv_file_path  # 输出保存的CSV文件路径

