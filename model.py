import torch
import os
import cv2
import numpy as np

class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cnn_layers = torch.nn.Sequential(
        # Convolutional Layer. 64 filter
        torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding="valid"),
        # Relu layer sau convolution
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
        torch.nn.ReLU(),
        # Bỏ bớt 10% output từ filter layer trước 
        torch.nn.Dropout2d(0.1),
        # Max pooling 2x2
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(128, 256, kernel_size=3, padding="valid"),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.1),
        torch.nn.MaxPool2d(3),
        # Từ kết quả neuron phân bố 2D thành 1D
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        # Bắt đầu layer Fully Connected của Neural Network bình thường
        # Số 2304 này được lấy bằng việc chạy thử CNN trước và xem lỗi cần số input neuron của layer là bao nhiêu. Có thể tự tính, nhưng phức tạp hơn. Có thể sử dụng nn.LazyLinear để không cần quan tâm đến số input neuron
        torch.nn.Linear(2304, 256),
        torch.nn.ReLU(),
        # Bỏ 10% neuron ngẫu nhiên. 
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 10),
        # Layer output dự đoán ảnh thuộc label nào
        torch.nn.Softmax()
    )
  def forward(self, X):
      # Giống với bài blog trước về Linear Regression
      # Chạy toàn bộ layer 
      result = self.cnn_layers(X)
      return result

device = torch.device('cpu')
cnn = CNN()
cnn.load_state_dict(torch.load("model.pth", map_location=device))
cnn.eval()

i = 1
while os.path.exists(f"./data_test/{i}.png"):
  img = cv2.imread(f"./data_test/{i}.png", cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (28, 28))
  img = img.reshape(1, 1, 28, 28)
  img = np.invert(img)
  img = torch.tensor(img, dtype=torch.float32)
  img = img.to(device)
  pred = cnn(img)
  pred = torch.argmax(pred, dim=1)
  print(f"Label: {i}, Predict: {pred.item()}")
  i+=1