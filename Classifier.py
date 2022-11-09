import torch
from PIL import Image
from torch import nn
import torchvision
from torch import optim
import torchvision.transforms as transformer
import numpy as np


#define Normalisation function
trans = transformer.Compose(
    [transformer.ToTensor(),
     transformer.Normalize( [0.5], [0.5])])

batch_size=4


# get data and define dataloaders
training_set = torchvision.datasets.MNIST(root='./', 
                                        train=True,
                                        download=False,
                                        transform=trans
                                        )

training_loader = torch.utils.data.DataLoader(training_set, 
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=2
                                            )

testing_set = torchvision.datasets.MNIST(root='./', 
                                        train=False,
                                        download=False, 
                                        transform=trans
                                        )

testing_loader = torch.utils.data.DataLoader(testing_set,
                                            batch_size=1,
                                            shuffle=False, num_workers=2
                                            )



#defining the feed foward neural net
network= nn.Sequential(
                        nn.Linear(784,128)
                        ,nn.ReLU(),
                        nn.Linear(128,128)
                        ,nn.ReLU(),
                        nn.Linear(128,10)

                    )


#defining Optimisor function as Stochastic Gradient Descent 
opti=optim.SGD(network.parameters(),lr=1e-2)
#defining Loss Function 
Loss_func=nn.CrossEntropyLoss()


#training
iterations=5
for i in range(iterations):
    for instance in training_loader:
        img,label=instance

        #use model to predict output with foward pass
        #inside brackets reshapes image using view
        netw_out=network(img.view(img.size(0),-1))
        #calculate diff between output and label with loss function
        netw_loss=Loss_func(netw_out,label)
        #zeroing gradients for backward pass
        network.zero_grad()
        #perform backward propagation
        netw_loss.backward()
        opti.step()
        #print("LOSS: ",netw_loss.item())
print("TRAINING DONE...")
#test training with data
with torch.no_grad():
    losses=0
    correct_out=0
    numExamples=len(testing_loader)
    for example in testing_loader:
        img,label=example
        img=img.view(img.size(0),-1)
        netw_out=network(img)
        netw_loss=Loss_func(netw_out,label).item()
        losses+=netw_loss
        nu,netw_out=torch.max(netw_out,1)
        correct_out+=(netw_out==label).sum()
    print("Accuracy : ", (correct_out/numExamples).item()*100  , " Mean Loss :",losses/numExamples)

PATH=input("Please enter a filepath : ")
while PATH!="exit":
    with torch.no_grad():
        img= Image.open(PATH)
        img=trans(img)
        img=img.view(img.size(0),-1)
        output=network(img)
        print(torch.max(output,1)[1].item())
        PATH=input("Please enter a filepath : ")


