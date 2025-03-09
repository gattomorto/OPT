import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

model_path = r'models\video_model.pth'


def SubsetMNIST(type):
    base_dir = "datasets"
    os.makedirs(base_dir, exist_ok=True)
    train = False
    if type == "training":
        train = True
    return torchvision.datasets.MNIST(base_dir, train=train, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))


class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        self.domain_bounds = torch.stack([torch.zeros(784), torch.ones(784)], dim=1)
        self.input_shape = torch.Size([1,1,28,28])
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)


def train():
    n_epochs =  20#20
    batch_size_train = 128
    model = VideoModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_set = SubsetMNIST("training")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            print("Loss: ",loss.item())

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model


def test(model):
    test_set = SubsetMNIST("testing")
    test_loader = torch.utils.data.DataLoader(test_set)
    model.eval()
    #test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            #test_loss += F.nll_loss(output, target)
            pred = output.argmax(-1)
            correct += pred.eq(target).sum().item()
    print("Accuracy: ", correct/len(test_loader.dataset))


def model_init(load_model = True):
    if load_model:
        model = VideoModel()
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        model = train()

    model.eval()
    return model


def attack_set_init(model):
    model.eval()
    correct_examples = []
    correct_labels = []
    set = SubsetMNIST("testing")
    data_loader = torch.utils.data.DataLoader(set)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model(data)
            pred = output.argmax(-1)
            correct_mask = pred.eq(target.data.view_as(pred)).squeeze()
            correct_examples.append(data[correct_mask])
            correct_labels.append(target[correct_mask])

    correct_examples_tensor = torch.cat(correct_examples, dim=0)
    correct_labels_tensor = torch.cat(correct_labels, dim=0)
    correctly_classified = [(example, label) for example, label in zip(correct_examples_tensor, correct_labels_tensor)]
    attack_set = correctly_classified

    return attack_set


def show_image(x):
    x = x.squeeze().detach().numpy()
    # x = (x * 0.3081) + 0.1307
    plt.figure(figsize=(5, 5))
    # plt.title(title, fontsize=14, weight='bold')
    plt.imshow(x, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()