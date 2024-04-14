import torch
import torch.nn as nn

from tqdm import tqdm

def train(data_loader, model, optimizer, device): #does training for one epoch.
    
    model.train() #put the model in training mode

    #iterating over every batch
    for data in data_loader:

        inputs=data['image']
        targets=data['targets']

        
        #move everything to device
        inputs=inputs.to(device, dtype=torch.float)
        targets=targets.to(device, dtype=torch.float)

        #zero-grad the optimizer
        optimizer.zero_grad()

        #forward output of model
        outputs=model(inputs)

        # print(f"Output shape= {outputs.shape}, targets shape={targets.shape}")

        #calculate loss, change loss function here
        loss=nn.BCEWithLogitsLoss()(outputs, targets)

        #backpropogation
        loss.backward()

        #optimize the step
        optimizer.step()

def evaluate(data_loader, model, device): #evaluates the model

    model.eval()#evaluation mode

    final_targets=[]
    final_outputs=[]

    with torch.no_grad():

        for data in data_loader:

            inputs=data["image"]
            targets=data["targets"]

            inputs=inputs.to(device, dtype=torch.float)
            targets=targets.to(device, dtype=torch.float)            

            # print("evaluate input size", inputs.shape)

            #generate prediction
            output=model(inputs)

            #converting to list
            targets=targets.detach().cpu().numpy().tolist()
            output=output.detach().cpu().numpy().tolist()

            #extend the original list
            final_targets.extend(targets)
            final_outputs.extend(output)
            
            # print(final_outputs, final_targets)
    return final_outputs, final_targets