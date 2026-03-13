#----IMPORTS-----------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

#----DATA-LOADING------------------------------------------------------------------------------
SEED = 4096                           #setting seed for randomizing dataset
torch.manual_seed(SEED)               #sets torch seed manually
if torch.cuda.is_available():         #if gpu processing is avaliable
    torch.cuda.manual_seed_all(SEED)  #set the cuda seed manually
    
np.random.seed(SEED)                  #set seed to numpy random number generator

file_path = 'iris.data'               #specify path of data file(csv)
df = pd.read_csv(                     #read data file to dataframe object
    file_path,
    header=None,
    names=['SLength', 'SWidth', 'PLength', 'PWidth', 'class'],    #format dataframe based on columns
)
#df.head()                             #display first 5 rows of dataframe

#df['class'].astype('category')      #display "class" column to "category" datatype

df['class'] = df['class'].astype('category')    #set dataframe "class" column as "category" datatype
df['class'] = df['class'].cat.codes             #sets "class" column values to categorical codes, i.e. 0-2 for 3 possible string values
#df.head()

n = len(df.index)                               #sets n to the number of rows
print(n)                                        #prints the number of rows
shuffled_indices = np.random.permutation(n)     #creates a shuffled set of numbers equal to the number of rows
df = df.iloc[shuffled_indices]                  #shuffles dataframe based on shuffled_indices generated
df.head()

x = df.iloc[:, :4].values.astype(np.float32)    #sets all rows, columns 0-4 to a float, stores in x
y = df.iloc[:, -1].values.astype(np.int64)      #sets all rows, last column(class) to an int type, stores in y

mu = x.mean(axis=0)                             #sets mu to the mean of x dataframe
span = x.max(axis=0) - x.min(axis=0)            #span = the range of values in x

def rescale(inputs):                            #function to normalize values in dataframe between -1 and 1
    return (inputs - mu) / span                 #something like the absolute mean difference, except no the absolute value

x = rescale(x)                                  #normalize x dataframe
print(x[:5])

num_train = int(n * .6)                         #defines an integer 60% the size of the dataset
num_test = n - num_train                        #degines an integer 40% the size of the dataset

x_train = x[:num_train]                         #defines training set dataframe(60% of dataset)
y_train = y[:num_train]                         #defines the labels for the training set
x_test = x[-num_test:]                          #test set(40% of the dataset)
y_test = y[-num_test:]                          #labels for test set

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

class NpDataset(Dataset):                             #Dataset class for encapsulating data and related functions
    def __init__(self, data, label):                  #
        assert len(data) == len(label)                #
        self.data = torch.from_numpy(data)            #
        self.label = torch.from_numpy(label).long()   #
        
    def __getitem__(self, index):                     #
        return self.data[index], self.label[index]    #
    
    def __len__(self):                                #
        return len(self.label)

train_dataset = NpDataset(x_train, y_train)           #
test_dataset = NpDataset(x_test, y_test)              #

train_dataloader = DataLoader(                        #
    train_dataset,                                    #
    batch_size=128,                                   #
    shuffle=False                                     #
)
test_dataloader = DataLoader(                         #
    test_dataset,                                     #
    batch_size=128,                                   #
    shuffle=False                                     #
)

len(train_dataloader.dataset)                         #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     #torch device object represents a device on which a tensor will be processed
print(device)


#----NETWORK-----------------------------------------------------------------------------------
class IrisNN(nn.Module):                    #neural network class containing layers, activation functions (nn.module is a base class for neural network modules)
    def __init__(self):                     #constructor
        super(IrisNN, self).__init__()      #
        
        #each of these represent a layer
        self.fn1 = nn.Linear(4, 6)          #4 features, 6 nodes in hidden layer
        self.fn2 = nn.Linear(6, 3)          #6 nodes in hidden layer, 3 output features
        
    def forward(self, x):                   #forward propagation
        x = F.relu(self.fn1(x))             #activation function for x, shapes it to layer 1 and stores in x
        x = self.fn2(x)                     #same thing for second layer but with no activation function
        return x                            #return x
    
model = IrisNN()                            #
model.to(device)                            #

#this is just a test of the nn class
x, y = next(iter(train_dataloader))         #dataloader is an iterator, so using next, iter, you can access tensor in dataset
x = x[:5].to(device)                        #passing 5 samples from training set
score = model(x)                            #score is the nn created by passing in x dataloader(or dataframe?)
print(score)                                #

loss_fn = nn.CrossEntropyLoss()                                               #loss function(pytorch CrossEntropyLoss function)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  #optimizer(Adam algorithm, learning rade 0.01, weight decay of 0.01)

def train():                                                                  #training function
    model.train()                                                             #
    
    for x, y in train_dataloader:                                             #iterate through dataloader
        x = x.to(device)                                                      #makes sure model is on the set device
        y = y.to(device)                                                      #""
        n = x.size(0)                                                         #get batch size
        
        optimizer.zero_grad()                                                 #reset gradient
        score = model(x)                                                      #returns activation function from our IrisNN class 
        loss = loss_fn(score, y)                                              #compute loss
        
        loss.backward()                                                       #backward propagate to compute gradient
        optimizer.step()                                                      #use optimizer to update parameters
        
        predictions = score.max(1, keepdim=True)[1]                           #gets number of correct predictions in each field(?)
        num_correct = predictions.eq(y.view_as(predictions)).sum().item()     #returns the number of correct predictions as a scalar value
        
    acc = num_correct / n                                                     #accuracy by dividing number correct by total predictions
    return loss, acc                                                          #returns the loss and accuracy

def evaluate():                                                               #
    model.eval()                                                              #
    
    with torch.no_grad():                                                     #
        for x, y in test_dataloader:                                          #
            x = x.to(device)                                                  #
            y = y.to(device)                                                  #
            n = x.size(0)                                                     #
            score = model(x)                                                  #
            loss = loss_fn(score, y)                                          #
            predictions = score.max(1, keepdim=True)[1]                       #
            num_correct = predictions.eq(y.view_as(predictions)).sum().item() #
        
    acc = num_correct / n                                                     #
    return loss, acc                                                          #

max_epochs = 200                                                              #200 epochs
for epoch in range(max_epochs):                                               #
    tr_loss, tr_acc = train()                                                 #
    eva_loss, eva_acc = evaluate()                                            #
    
    print(f'[{epoch}/{max_epochs}] Train loss:{tr_loss:.4f} acc:{tr_acc*100:.2f}% - Test loss:{eva_loss:.4f} acc:{eva_acc*100:.2f}%')

for param in model.parameters():         #
    print(param)                         #