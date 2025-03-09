import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import sounddevice as sd
import shutil
import os

model_path = r'models\audio_model.pth'

def filter_SPEECHCOMMANDS():
    digit_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    base_dir = "./datasets/SpeechCommands/speech_commands_v0.02"
    def remove_paths(file_path):
        with open(os.path.join(base_dir, file_path), "r") as file:
            lines = file.readlines()
        filtered_lines = [line for line in lines if line.strip().split("/")[0] in digit_words]
        if len(lines) != len(filtered_lines):
            print("Removing paths...")
            with open(os.path.join(base_dir, file_path), "w") as file:
                file.writelines(filtered_lines)

    def remove_folders():
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item not in digit_words:
                print(f"Removing folder: {item_path}")
                shutil.rmtree(item_path)

    remove_paths("testing_list.txt")
    remove_paths("validation_list.txt")
    remove_folders()

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        base_dir = "datasets"
        os.makedirs(base_dir, exist_ok=True)
        super().__init__(base_dir, download=True)

        SC_compressed_file_path = base_dir + "/speech_commands_v0.02.tar.gz"
        if os.path.exists(SC_compressed_file_path):
            print("Removing ", SC_compressed_file_path)
            os.remove(SC_compressed_file_path)

        filter_SPEECHCOMMANDS()
        def load_list(filename):
            #self._path = ./SpeechCommands\speech_commands_v0.02
            #filepath = ./SpeechCommands\speech_commands_v0.02\testing_list.txt
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                # lista di 'SpeechCommands\\speech_commands_v0.02\\right\\a69b9b3e_nohash_0.wav'
                paths = [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
                return paths

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def collate_fn(batch):
    word_to_digit = { "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    tensors, targets = [], []
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
    for waveform, _, label, *_ in batch:
        waveform = transform(waveform).to(torch.float64)
        tensors += [waveform]
        targets += torch.tensor([word_to_digit[label]])

    # list: 20 -> Tensor: (20,1,8000)
    tensors = [item.t() for item in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.)
    tensors = tensors.permute(0, 2, 1)

    # [tensor(9), tensor(4), tensor(3), tensor(2), tensor(6), ... -> tensor([9, 4, 3, 2, 6,...
    targets = torch.stack(targets)

    return tensors, targets

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.domain_bounds = torch.stack([-1*torch.ones(8000), torch.ones(8000)], dim=1)
        self.input_shape = torch.Size([1,1,8000])
        stride = 16
        n_channel = 32
        self.conv1 = nn.Conv1d(1, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, 10)

    def forward(self, x):
        x = self.conv1(x) # (1,8000) vs (50,1,8000)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def train():
    model = AudioModel()
    model.train()
    train_set = SubsetSC("training")

    batch_size = 256
    n_epoch = 50 #50
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(1, n_epoch + 1):
        print("epoch: ", epoch)
        #################
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = F.nll_loss(output[:, 0, :], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.6f}")
        #################
        scheduler.step()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model

def test(model):
    test_set = SubsetSC("testing")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
    )
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.unsqueeze(1)
            output = model(data)
            pred = output.argmax(-1)
            correct_mask = pred.eq(target).squeeze()
            correct += correct_mask.sum().item()
    print(f"Accuracy: ", correct/len(test_loader.dataset))

def play_sound(x,sample_rate=8000):
    waveform = x.squeeze().detach().numpy()
    sample_rate = sample_rate
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()

def model_init(load_model=True):
    model = AudioModel()
    if load_model:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        model = train()

    model.eval()
    return model

def attack_set_init(model):
    set = SubsetSC("testing")
    data_loader = torch.utils.data.DataLoader(set, collate_fn=collate_fn, batch_size=128, shuffle=True)
    correct_examples = []
    correct_labels = []
    with torch.no_grad():
        for data, target in data_loader:
            target = target.unsqueeze(1)
            output = model(data)
            pred = output.argmax(-1)
            correct_mask = pred.eq(target).squeeze()
            correct_examples.append(data[correct_mask])
            correct_labels.append(target[correct_mask])

    correct_examples_tensor = torch.cat(correct_examples, dim=0).unsqueeze(1)
    correct_labels_tensor = torch.cat(correct_labels, dim=0).squeeze()
    attack_set = [(example, label) for example, label in zip(correct_examples_tensor, correct_labels_tensor)]

    return attack_set
