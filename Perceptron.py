#create a model that will be able to decipher points we place on a graph
import torch
import torch.nn as nn #accessing neural network base class from torch library (writing nn instead of torch.nn)
import torch.optim as optim # manage gradient descendent and back propagation steps by automation
import numpy as np
import matplotlib.pyplot as plt # graphing
import time
from matplotlib.widgets import Button # interactive graphing
from networkx.classes import non_edges

#setting up to use GPU
# cuda version 12.9
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-cache-dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("gpu")

#initiliazations of lists
points = [] # grid points (x, y)
labels = [] # 0, 1 / blue,red

#placeholder for model temp
model = None

def onclick(event):

    if event.inaxes == ax: # ax will be assigned via plot later
        if event.button == 1: # left click
            plt.scatter(event.xdata, event.ydata, color='red')
            points.append([event.xdata, event.ydata]) # inputs into the array an array, [[x, y], [x, y]]
            labels.append(0)
        elif event.button == 3: # right click
            plt.scatter(event.xdata, event.ydata, color='blue')
            points.append([event.xdata, event.ydata])
            labels.append(1)
        elif event.button == 2:
             train_model()


        plt.draw() ##UPDATING PLOT, NOT REDRAWING


class Perceptron(nn.Module): # Perceptron is a child of nn.module
    def __init__(self):
        super(Perceptron, self).__init__() #override parent constructor
        self.fc1 = nn.Linear(2, 25)
        self.fc2 = nn.Linear(25, 5)
        self.fcOut = nn.Linear(5, 1)
        #self.fc = nn.Linear(2, 1) # creating a single layer of the neural network with 2 inputs and 1 output

    def forward(self, x): # x is the input tensor, that takes both the x and y
        #sending our weights bias through the activation function to get a number between 0 and 1
        z = torch.tanh(self.fc1(x))
        y = torch.tanh(self.fc2(z))
        return torch.sigmoid(self.fcOut(y)) # sigmoid activation function for rapidly moves between 0 and 1

def train_model():
     # make sure to use model
    # print("Training")

    global model
    #conversion of points and labels into numpy matrix's
    position = np.array(points, dtype=np.float32)
    target_labels = np.array(labels, dtype=np.float32).reshape(-1, 1) # rashaping [0, 1, 0, 1, 1] into [[1], [0], [0]]

    #normalaizing the x/y data  to between 0 and 1
    position_min = position.min(axis=0)
    position_max = position.max(axis=0)
    normalized_position = (position - position_min) / (position_max - position_min)

    #convert normalaized data and labers to pytorch tensors
    inputs = torch.tensor(normalized_position, dtype=torch.float32).to(device) #to device tells torch to use gpu (defined at the top)
    desired_outputs = torch.tensor(target_labels, dtype=torch.float32).to(device)

    if model is None:
        model = Perceptron().to(device)

    criterion = nn.MSELoss() # bianary cross-entropy loss || set value to 0 or 1, making a binary output
    optimizer = optim.SGD(model.parameters(), lr=2) # lr is learning rate
    # optimizer = optim.Adam(model.parameters(), lr=0.1)  similar results


    epochs = 500 #number of training attempts
    for epoch in range(epochs):
        outputs = model(inputs) # passing the inputs through the model...this is calling he forward function(Perceptron)
        loss = criterion(outputs, desired_outputs) #calculate the avg loss across all the points of this epoch
        optimizer.zero_grad() # clear out any gradient values from prev epoch from backward()
        loss.backward() # Performing backpropagation to calc our new gradients based on the loss function
        optimizer.step() # Take the gradients and use them to change the weights and biases of the model

        # showcase every 50 epeochs current loss vallue

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], loss: {loss.item():.4f}")
            plot_decision_boundary(model, position_min, position_max)
            # fig.canvas.draw()
            fig.canvas.flush_events()
            #plt.pause(0.25) also works
            time.sleep(0.1)


    # plot_decision_boundary(model, position_min, position_max)

def plot_decision_boundary(m, position_min, position_max):
    plt.cla() #clear the plot
    print("thinknig")
    #generate a meshgrid for the decision boundary
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    grid = np.c_[xx.ravel(), yy.ravel()] # converts lists xx and yy to same as points
    normalized_grid = (grid - position_min) / (position_max - position_min)
    grid_tensor = torch.tensor(normalized_grid, dtype=torch.float32).to(device)

    #pred the value for each point
    with torch.no_grad():
        zz = m(grid_tensor).cpu().reshape(xx.shape)

    #plotting the decision boundary, showing the bg with the inverted and muted colours
    #xx, yy tells us where to draw each value in zz, the RdBu cmaps the 0 to red, 0.5 to white, 1 to blue
    plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.1)

    #scatter the points drawing each of them ([:,0] slices a list of all the x values, [:,1] slice all the y values)
    #bwr maps 0 to blue, 0.5 to white, 1 to red
    #edgecolor = k for black outline
    #s = 100 sets the size
    plt.scatter(np.array(points)[:,0], np.array(points)[:,1], c=labels, cmap="RdBu", alpha=0.5, edgecolors="k", s=200)

    for i, point in enumerate(points):
        if labels[i] == 0:
            plt.scatter(point[0], point[1], color='red')
        else:
            plt.scatter(point[0], point[1], color='blue')

    plt.title("Perceptron Decision Boundary")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    plt.draw() #UPDATING PLOT, NOT REDRAWING




#initialization

fig, ax = plt.subplots()
plt.title("Click to add points, left = blue, right = green")
plt.xlim([-1, 1])
plt.ylim([-1, 1])

#connect mouse clicks to onclick function
print("done")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.ion()
plt.show()

