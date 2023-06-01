from tqdm import tqdm
import torch.nn as nn
import torch
import config_objects
import network
import os
import pickle
import itertools

def run_experiment(train_dset, valid_dset, name, exp_config: config_objects.ModelCfg):
    net = network.Transformer(name, exp_config, train_dset.cfg).to(train_dset.cfg.device)
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=exp_config.learn_rate, weight_decay=exp_config.weight_decay)
    train_result = train(net, optim, loss_func, exp_config.epochs, train_dset, valid_dset)
    return train_result


def run_experiments(train_dset, valid_dset, experiment_path, hyperparams, prob_dists=None, search_type="grid", override=False, num_rand=20):
    assert search_type in ["grid", "random"]
    assert all(k in config_objects.ModelCfg().__dataclass_fields__ for k in hyperparams)
    if prob_dists is None:
        prob_dists = {}

    log_path = os.path.join(experiment_path, "log.pkl")
    if os.path.exists(log_path):
        with open(log_path, "rb") as p:
            saved = pickle.load(p)
        if saved["dset_cfg"] != train_dset.cfg: #or saved["hyperparams"] != hyperparams:
            print("Found existing log at path with different config than specified")
            if not override:  # override: set this to ignore the dset_config equality check
                return
        train_results, test_results = saved["train_results"], saved["test_results"]
    else:
        train_results, test_results = {}, {}

    model_name = "_".join(f"{{{k}}}" for k in hyperparams) + ".dict"
    model_path = os.path.join(experiment_path, model_name)

    log_dict = {"hyperparams": hyperparams,
                "dset_cfg": train_dset.cfg,
                "train_results": train_results,
                "test_results": test_results}

    hyp_keys, hyp_choices = list(hyperparams.keys()), list(hyperparams.values())
    experiment_iter = itertools.product(*hyp_choices) if search_type == "grid" else range(num_rand)
    for i, item in enumerate(experiment_iter):
        if search_type == "grid":  # here, "item" is the actual selections for the hyperparameters
            hyp_dict = dict(zip(hyp_keys, item))
        elif search_type == "random":  # here "item" is just an integer
            hyp_dict = {}
            for k,choices in hyperparams.items():
                prob_dist = prob_dists.get(k)
                if isinstance(choices, dict):
                    choices = list(choices.keys())                
                hyp_dict[k] = np.random.choice(choices, p=prob_dist)
        
        # use the named choices version of hyp_dict
        name = model_path.format(**hyp_dict)  # guarantees that the format specified in name matches the actual hyperparams

        # fetch the values associated with the named choices
        for k,choice in hyp_dict.items():
            if isinstance(choice, str):  # assume that if a hyperparameter takes a string value, it's a named choice
                hyp_dict[k] = hyperparams[k][choice]

        # use the value-only version of hyp_dict
        exp_config = config_objects.ExperimentConfig(**hyp_dict)
        if exp_config in train_results:
            print("Already completed experiment for", name)
            continue
        print("Running experiment for", name, "experiment", i+1)

        train_result, test_result = run_experiment(train_dset, valid_dset, name, exp_config)
        
        train_results[exp_config] = train_result
        test_results[exp_config] = test_result

        with open(log_path, "wb") as p:
            pickle.dump(log_dict, p)

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


