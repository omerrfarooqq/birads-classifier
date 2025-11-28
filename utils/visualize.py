import matplotlib.pyplot as plt
import torch
import os

def save_training_plot(train_loss, val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(["Train", "Val"])
    plt.savefig("outputs/training_loss.png")
    plt.close()
