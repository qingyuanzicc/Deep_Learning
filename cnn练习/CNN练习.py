import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

batch_size = 64
lr = 1e-4
num_epochs = 100
in_channel = 1
out_class = 10

train_data = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False)
print(train_data.train_labels.size())
import matplotlib.pyplot as plt
print(type(train_loader))
plt.imshow(next(iter(train_loader))[0][0].squeeze(0), cmap="gray")
plt.show()
# class Cnn(nn.Module):
#     def __init__(self, in_channel, out_class):
#         super(Cnn, self).__init__()
#         self.feature_extraction = nn.Sequential(
#             nn.Conv2d(in_channel, 8, 3, 1, 1),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(8, 16, 5, 1, 2),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2)
#         )
#         self.classification = nn.Sequential(
#             nn.Linear(784, 500),
#             nn.ReLU(True),
#             nn.Linear(500, 100),
#             nn.Sigmoid(),
#             nn.Linear(100, out_class)
#         )
#
#     def forward(self, x):
#         x1 = self.feature_extraction(x)
#         x2 = x1.view(x1.size(0), -1)
#         x3 = self.classification(x2)
#         return x3
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Cnn(in_channel, out_class).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr)
#
# for epoch in range(num_epochs):
#     total_loss = 0.0
#     total_acc = 0.0
#     print('*' * 30)
#     for i, data in enumerate(train_loader):
#         img, label = data
#         img = img.to(device)
#         label = label.to(device)
#
#         out = model(img)
#         loss = criterion(out, label)
#         total_loss += loss
#         num_correct = (torch.argmax(out, dim=1) == label).sum()
#         total_acc += num_correct
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print('epoch:[{}/{}] loss: {}, acc: {}'.format(
#         epoch + 1,
#         num_epochs,
#         total_loss.item()/train_data.train_labels.size(0),
#         total_acc.item()/train_data.train_labels.size(0)
#     ))
