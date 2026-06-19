import os
import random
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_url
from torchvision import transforms
import torchvision


url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
download_url(url, '.', 'cifar-10-python.tar.gz', None)

import tarfile
with tarfile.open('cifar-10-python.tar.gz', 'r:gz') as tar:
    tar.extractall(path='.')


transform = transforms.Compose([
    transforms.ToTensor()
])

cifar_dataset = datasets.CIFAR10(root='./', train=True, download=False, transform=transform)

num_images = len(cifar_dataset)
indices = list(range(num_images))
random.shuffle(indices)
split = int(0.8 * num_images)
train_indices, test_indices = indices[:split], indices[split:]

classes = cifar_dataset.classes
for class_name in classes:
    os.makedirs(os.path.join('cifar10', 'train', class_name), exist_ok=True)
    os.makedirs(os.path.join('cifar10', 'test', class_name), exist_ok=True)


for i in train_indices:
    image, label = cifar_dataset[i]
    class_name = classes[label]
    image_path = os.path.join('cifar10', 'train', class_name, f'image_{i}.png')
    torchvision.utils.save_image(image, image_path)

for i in test_indices:
    image, label = cifar_dataset[i]
    class_name = classes[label]
    image_path = os.path.join('cifar10', 'test', class_name, f'image_{i}.png')
    torchvision.utils.save_image(image, image_path)

print("CIFAR-10 dataset organized into training and testing folders successfully.")
