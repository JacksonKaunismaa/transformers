import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle


def correct(pred_logits, labels, dim=1):  # doesn't really make sense unless you do teacher forcing
    if labels.shape[1] != 1:
        pred_probabilities = F.softmax(pred_logits, dim=dim)
        classifications = torch.argmax(pred_probabilities, dim=dim)
        labels_argmax = torch.argmax(labels, dim=dim)
    else:  # binary classification
        classifications = pred_logits.int()
        labels_argmax = labels
    correct = (labels_argmax == classifications)
    return correct


@torch.no_grad()
def evaluate(net, loss, valid_loader, device=None):
    epoch_va_loss = 0.0
    epoch_va_correct = 0
    net.eval()
    total_valid = 0
    for i, sample in enumerate(valid_loader):
        imgs = sample["image"].to(device).float()
        labels = sample["label"].to(device).float()
        outputs = net(imgs)
        epoch_va_loss += loss(outputs, labels).item()
        epoch_va_correct += correct(outputs, labels).sum().item()
        total_valid += labels.shape[0]
    epoch_va_accuracy = epoch_va_correct/total_valid
    return epoch_va_loss, epoch_va_accuracy


def train(net, optimizer, loss, epochs, train_loader, valid_loader, device=None):
    va_losses = []
    tr_losses = []
    va_accuracies = []
    for epoch in range(epochs):
        epoch_tr_loss = 0.0
        net.train()
        #input("Entering train")
        for i, sample in tqdm(enumerate(train_loader)):
            inputs, targets = sample.to(device)
            outputs = net(inputs)
            batch_loss = loss(targets, outputs)
            epoch_tr_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_va_loss, epoch_va_accuracy = evaluate(net, loss, valid_loader, device=device)
        epoch_summary = f'Epoch {epoch + 1}: va_loss: {epoch_va_loss}, va_accuracy: {epoch_va_accuracy}, tr_loss: {epoch_tr_loss}'
        #input("Entering extra")
        print(epoch_summary)
        if not va_losses or epoch_va_loss < min(va_losses):
            net.save_model_state_dict(optim=optimizer)
        va_losses.append(epoch_va_loss)
        tr_losses.append(epoch_tr_loss)
        va_accuracies.append(epoch_va_accuracy)
        
    return va_losses, va_accuracies, tr_losses


