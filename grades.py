import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

innerDim = 64

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()

        # Define Layers
        self.seq_layers = nn.Sequential(
                nn.Linear(input_dim, innerDim),
                nn.Sigmoid(),
                nn.Linear(innerDim, innerDim),
                nn.ReLU(),
                nn.Linear(innerDim, innerDim),
                nn.ReLU(),
                nn.Linear(innerDim, output_dim) 
        )
        
    def forward(self, x):
        y_pred = self.seq_layers(x)
        return y_pred


def main():

    rate = 0.01
    X = torch.tensor( [ [5., 9.], [4., 8.], [3., 6.], [5., 8.], [1., 4.], [2., 6.], [8., 8.], [7., 8.], [2., 5.], [6., 9.]], dtype=torch.float32)
    Y = torch.tensor( [ [0.92], [0.91], [0.82], [0.95], [0.74], [0.75], [0.96], [0.94], [0.80], [0.91]  ] , dtype=torch.float32)

    n_samples, n_features = X.shape     # X is a 10x2

    net = Net(n_features, 1)
    EPOCHS = 1000

    loss = torch.nn.MSELoss()     # define loss function as MSE
    optimizer = torch.optim.SGD(net.parameters(), lr=rate)

    for epoch in range(EPOCHS):
        y_pred = net(X)
        l = loss(y_pred, Y)
      
        # Print Loss
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}: MSE Loss: {l.item()}')

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    
    # testers 
    test_tense = torch.tensor( [ [3, 9], [1, 3], [7, 7], [4, 5], [8, 9] ], dtype=torch.float32)

    print()
    pred = net(test_tense);
    i = 0
    for element in pred:
        print(f'Study Hours: {test_tense[i][0]}   Sleep Hours: {test_tense[i][1]}   Prediction: ', element.item())
        i+=1 

if __name__ == "__main__":
    main()
