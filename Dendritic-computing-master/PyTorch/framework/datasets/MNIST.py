# 开摩托的小猫
import torchvision
from torchvision import transforms


def mnist_dataset(args):

    # Data augmentation
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道
    transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root=args.data_root,train=True, download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    return {"train":trainset, "test":testset}