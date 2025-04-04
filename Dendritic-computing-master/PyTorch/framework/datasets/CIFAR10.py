# 开摩托的小猫
import torchvision
def CIFAR10_dataset(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform)

    return {"train":trainset, "test":testset}