import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
import torchvision
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 数据载入与预处理
    trans_train = torchvision.transforms.Compose(
        [torchvision.transforms.RandomResizedCrop(224),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    trans_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.CIFAR10(root='../ImageClassification/data/cifar10', train=True, download=True, transform=trans_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    test_data = torchvision.datasets.CIFAR10(root='../ImageClassification/data/cifar10', train=False, download=False, transform=trans_valid)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Cifar-10中的各种类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'ship', 'truck')

    dataiter = iter(train_loader)

    images, labels = next(dataiter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LR = 0.001

    # 选择模型
    print("press 1 for ViT model")
    print("press 2 for CNN model")
    print("press 3 for VGG model")
    print("press 4 for Resnet model")
    print("Please choose your model:", end=" ")
    model = None
    num = input()
    if (num == '1'):
        print('===> Model building ......\n')
        from model import ViT
        model = ViT.ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 1024,
            pool = 'cls',
            channels = 3,
            dim_head = 64,
            dropout = 0.,
            emb_dropout = 0.,
            )
        print('Model has been build successfully!\n')
        print('Model structure: \n')
        print(model)
        optimizer = torch.optim.Adagrad(model.parameters(), lr = LR)
        criterion = torch.nn.CrossEntropyLoss()

    elif num == '2':
        print('===> Model building ......\n')
        from model import CNN
        model = CNN.CNN()
        print('Model has been build successfully!\n')
        print('Model structure: \n')
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr = LR)
        criterion = torch.nn.CrossEntropyLoss()

    elif num == '3':
        print('===> Model building ......\n')
        from model import VGG
        model = VGG.VGG('VGG19')
        print('Model has been build successfully!\n')
        print('Model structure: \n')
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)
        criterion = torch.nn.CrossEntropyLoss()

    elif num == '4':
        print('===> Model building ......\n')
        from model import ResNet
        model = ResNet.ResNet18()
        print('Model has been build successfully!\n')
        print('Model structure: \n')
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr = LR)
        criterion = torch.nn.CrossEntropyLoss()

    def train(model, train_loader, optimizer, criterion, device,):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 初始化损失值为0
        correct = 0  # 初始化预测正确的样本数为0
        total = 0  # 初始化总样本数为0
        loop = tqdm(train_loader, total=len(train_loader))  # 使用 tqdm 创建进度条
        loop.set_description(f'Training Starting ...')

        for i, (images, labels) in enumerate(loop):  # 遍历训练数据集，并使用 enumerate 获取批次索引
            images, labels = images.to(device), labels.to(device)  # 将输入数据和标签移动到指定的设备上（例如GPU）

            optimizer.zero_grad()  # 清零梯度，以防止梯度累积
            outputs = model(images)  # 前向传播：通过模型获取预测输出

            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 更新模型参数

            # 计算损失和准确率
            running_loss += loss.item()  # 累加损失值
            _, predicted = outputs.max(1)  # 获取预测结果中的最大值及其对应的索引
            total += labels.size(0)  # 累加样本数量
            correct += predicted.eq(labels).sum().item()  # 累加预测正确的样本数量

            # 更新进度条显示
            loop.set_description(f'Training Process [{i+1}/{len(train_loader)}]')
            loop.set_postfix(loss=running_loss/(i+1), acc=correct/total)

        # 计算平均损失和准确率
        train_loss = running_loss / len(train_loader)  # 计算平均损失
        train_acc = correct / total  # 计算平均准确率

        return train_loss, train_acc
    def test(model, test_loader, criterion, device):
        model.eval()  # 将模型设置为评估模式
        running_loss = 0.0  # 初始化损失值为0
        correct = 0  # 初始化预测正确的样本数为0
        total = 0  # 初始化总样本数为0

        loop = tqdm(test_loader, total=len(test_loader))  # 使用 tqdm 创建进度条
        loop.set_description(f'Testing Starting ...')

        with torch.no_grad():  # 在评估过程中，不需要计算梯度
            for i, (images, labels) in enumerate(loop):  # 遍历测试数据集，并使用 enumerate 获取批次索引
                images, labels = images.to(device), labels.to(device)  # 将输入数据和标签移动到指定的设备上（例如GPU）

                outputs = model(images)  # 前向传播：通过模型获取预测输出
                loss = criterion(outputs, labels)  # 计算损失值

                # 计算损失和准确率
                running_loss += loss.item()  # 累加损失值
                _, predicted = outputs.max(1)  # 获取预测结果中的最大值及其对应的索引
                total += labels.size(0)  # 累加样本数量
                correct += predicted.eq(labels).sum().item()  # 累加预测正确的样本数量

                # 更新进度条显示
                loop.set_description(f'Testing Process [{i+1}/{len(test_loader)}]')
                loop.set_postfix(loss=running_loss/(i+1), acc=correct/total)

        # 计算平均损失和准确率
        test_loss = running_loss / len(test_loader)  # 计算平均损失
        test_acc = correct / total  # 计算平均准确率

        return test_loss, test_acc

    def run(EPOCHS, model, train_loader, test_loader, optimizer, criterion, device):
        num_epochs = EPOCHS  # 设置训练 epoch 数
        test_loss_v = []
        test_acc_v = []

        for epoch in range(1, num_epochs + 1):
            print(f'Epoch: {epoch}')
            print('===>Train Start')
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            print('===>Test Start')
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_loss_v.append(test_loss)
        test_acc_v.append(test_acc)

        # 可视化训练损失和准确率
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), test_loss_v, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Testing Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), test_acc_v, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
        save_dir = './saved_models'
        # 如果目录不存在，创建目录
        os.makedirs(save_dir, exist_ok=True)
        # 定义保存路径和文件名
        if num == '1':
            model_save_path = os.path.join(save_dir, 'vit_model.pth')
            # 保存模型
            torch.save(model.state_dict(), model_save_path)
            print(f"ViT model saved to {model_save_path}")
            plt.savefig('vit_model.png')
        elif num == '2':
            model_save_path = os.path.join(save_dir, 'cnn_model.pth')
            # 保存模型
            torch.save(model.state_dict(), model_save_path)
            print(f"CNN model saved to {model_save_path}")
            plt.savefig('CNN.png')
        elif num == '3':
            model_save_path = os.path.join(save_dir, 'vgg_model.pth')
            # 保存模型
            torch.save(model.state_dict(), model_save_path)
            print(f"VGG model saved to {model_save_path}")
            plt.savefig('vgg.png')
        elif num == '4':
            model_save_path = os.path.join(save_dir, 'resnet18_model.pth')
        # 保存模型
            torch.save(model.state_dict(), model_save_path)
            print(f"ResNet18 model saved to {model_save_path}")
            plt.savefig('ResNet18.png')

    EPOCH = 20
    run(20, model,train_loader, test_loader, optimizer, criterion, device)
 
