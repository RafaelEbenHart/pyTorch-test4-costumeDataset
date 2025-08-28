import torch
from torch import nn
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt

# prediction
# torch.manual_seed(42)
def evalModel(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0,0
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for X,y in dataLoader:
            X,y = X.to(device),y.to(device)
            # prediksi
            yPred = model(X)
            # akumulasi loss dan Acc per batch
            loss += lossFn(yPred,y)
            acc += accuracy_fn(y_true=y,
                               y_pred=yPred.argmax(dim=1))

        # scale loss and acc to find avg per batch
        loss /= len(dataLoader)
        acc /= len(dataLoader)
    result = {"ModelName" : model.__class__.__name__,# hanya bisa berkerja jika mmodel dibuat dengan class
            "ModelLoss" : loss.item(),
            "ModelAcc" : acc}
    return result

# time
def printTrainTime(start:float,
                   end:float,
                   device: torch.device = None):
    """print perbandingan antara start dan end time"""
    totalTime = end-start
    print(f"Train time on {device}: {totalTime:.3f} seconds")
    return totalTime

# train loop
def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accFn,
              perBatch: None):
    """Performs training with model trying to learn on dataLoader"""
    trainLoss,trainAcc = 0, 0
    device = next(model.parameters()).device
    for batch, (X,y) in enumerate(dataLoader):
        model.train()
        X,y = X.to(device),y.to(device) # put tu target device
        yPred = model(X) # forward pass
        # Calculate loss and acc
        loss = lossFn(yPred,y)
        trainLoss += loss
        trainAcc += accFn(y_true=y,
                               y_pred=yPred.argmax(dim=1))
        # optimizer zero grad,loss backward,optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show batch
        if perBatch:
            if batch % perBatch == 0:
                print(f"Looked at: {(batch * len(X)) + perBatch}/{len(dataLoader.dataset)} samples ")

    # calculate avg
    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    print(f"|Train Loss: {trainLoss:.5f} | Train Acc: {trainAcc:.2f}%|")

# test loop

def testStep(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn):
    """Performs testing with model trying to test on dataLoader"""
    testLoss,testAcc = 0, 0
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for X,y in dataLoader:
            X,y = X.to(device),y.to(device)
            testPred = model(X)
            testLoss += lossFn(testPred,y)
            testAcc += accFn(y_true=y,
                             y_pred=testPred.argmax(dim=1))

        testLoss /= len(dataLoader)
        testAcc /= len(dataLoader)
        print(f"|Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}%|")
    return testLoss.item(),testAcc

def makePredictions(model:torch.nn.Module,
                    data:list):
    predProbs = []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # prepare the sample (add a batch dimensio and pass to target device)
            sample = torch.unsqueeze(sample,dim=0).to(device)

            # forward pass (model output raw logits)
            predLogits = model(sample)

            # get prediction probability (logit -> preediction probability)
            predProb = torch.softmax(predLogits.squeeze(),dim=0)

            # get predprobs off the CUDA for further calculations
            predProbs.append(predProb.cpu())

    # stack the predProbs to turn list into a tensor
    return torch.stack(predProbs)

# non linear activation
class hyperTanh(nn.Module):
    def forward(self, x) :
        ex = torch.exp(x)
        enx = torch.exp(-x)
        return (ex - enx) / (ex + enx)




# Saving
def Save(path:str,name:str,model:torch.nn.Module):
    import torch
    from pathlib import Path

    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)
    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    print(f"Saving your model to {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)

# Loading
def load(model: torch.nn.Module, Saved_path:str):
    import torch
    model.load_state_dict(torch.load(f=Saved_path))

# visualize data(image)
def display_random_image(dataset: torch.utils.data.Dataset,
                          classes: list[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print(f"for display, purpose, N shouldn't be larger than 10,setting to 10 and removing shape display")
    if seed:
        random.seed(seed)
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(10,5))
    for i, targ_samples in enumerate(random_samples_idx):
        targ_img, targ_label = dataset[targ_samples][0], dataset[targ_samples][1]
        targ_img_adjust = targ_img.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(targ_img_adjust)
        plt.axis(False)
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title= title + f"\nshape: {targ_img_adjust.shape}"
        plt.title(title,fontsize=5)
    plt.show()


