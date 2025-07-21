import os
import torch
import torch.nn as nn

def save_network(model):
    if not os.path.exists('./model'):
        os.makedirs('./model')
    torch.save(model.state_dict(), './model/best_model.pth')
    print("Model saved to ./model/best_model.pth")

def save_network_without_classifier(model):
    if not os.path.exists('./model'):
        os.makedirs('./model')
    torch.save(model.state_dict(), './model/best_model_without_classifier.pth')
    print("Model saved to ./model/best_model_without_classifier.pth")