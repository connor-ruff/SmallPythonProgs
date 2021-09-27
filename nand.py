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
                nn.Linear(innerDim, output_dim)
        )
        
    def forward(self, x):
        y_pred = (self.seq_layers(x))

        return (y_pred)


def main():

    rate = 0.01
    X = torch.tensor( [  [0,0], [0,1], [1,0], [1,1]  ], dtype=torch.float32)
    Y = torch.tensor( [  [1],    [1],   [1],   [0]  ], dtype=torch.float32)

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
    test_tense = torch.tensor( [ [0,0], [0,1], [1,0], [1,1] ], dtype=torch.float32)
    pred = net(test_tense);

    # Process Response and Flatten to Binary
    for i in range(4):
        distToZero = abs(pred[i])
        distToOne = abs(1 - pred[i])
        if distToZero < distToOne:
            pred[i] = 0
        else:
            pred[i] = 1
            

    print()
    i = 0
    for element in pred:
        print(f'Inputs: [{test_tense[i][0]}, {test_tense[i][1]}]  --->  ', int(element.item()))
        i+=1 

if __name__ == "__main__":
    main()
