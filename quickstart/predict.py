import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from modules.neural_network import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

classes = test_data.classes

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
