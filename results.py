import torch
import pandas as pd
import matplotlib.pyplot as plt

from function import load,load_results_txt
from poltFunction import plot_loss_curves

model0_results = load_results_txt("results/model0_results.txt")
model1_results = load_results_txt("results/model1_results.txt")

model0_df = pd.DataFrame(model0_results)
model1_df = pd.DataFrame(model1_results)
# print(model0_df)

# setup plot
# train loss
plt.figure(figsize=(10,5))
epochs = range(len(model0_df))
plt.subplot(2,2,1)
plt.plot(epochs,model0_df["train_loss"], label="Model 0")
plt.plot(epochs,model1_df["train_loss"], label ="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs", fontsize = 5)
plt.legend()
# plt.show()
# test loss
plt.subplot(2,2,2)
plt.plot(epochs,model0_df["test_loss"], label="Model 0")
plt.plot(epochs,model1_df["test_loss"], label ="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs", fontsize = 5)
plt.legend()
# train acc
plt.subplot(2,2,3)
plt.plot(epochs,model0_df["train_acc"], label="Model 0")
plt.plot(epochs,model1_df["train_acc"], label ="Model 1")
plt.title("Train Acc")
plt.xlabel("Epochs", fontsize = 5)
plt.legend()
# test acc
plt.subplot(2,2,4)
plt.plot(epochs,model0_df["test_acc"], label="Model 0")
plt.plot(epochs,model1_df["test_acc"], label ="Model 1")
plt.title("Test Acc")
plt.xlabel("Epochs", fontsize = 5)
plt.legend()
plt.show()