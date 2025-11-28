from torchvision import transforms

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    return train_tf, test_tf
