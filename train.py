from tqdm import tqdm

def train(net, optimizer, loss, epochs, data_loaders, device=None):
    losses = dict(train=[], valid=[])
    for epoch in range(epochs):
        for split in ["train", "valid"]:
            epoch_loss = 0.0
            for i, sample in tqdm(enumerate(data_loaders[split])):
                inputs, targets = sample.to(device)
                outputs = net(inputs)
                batch_loss = loss(targets, outputs)
                epoch_loss += batch_loss.item()
                if split == "train":
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            losses[split].append(epoch_loss.item())
        epoch_summary = f'Epoch {epoch + 1}: ' + " ".join(f"{split[:2]}_loss: {loss[-1]:.4f}" for split,loss in losses.items())
        print(epoch_summary)

        if losses["valid"][-1] < net.best_loss:  # maybe dont save every epoch if loss improves?
            net.best_loss = losses["valid"][-1]
            net.save_model_state_dict(optim=optimizer)
    return losses


