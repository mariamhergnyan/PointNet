# main function at the end

from torch.utils.data import DataLoader, Dataset
from src.models.pointnet import PointNet
from src.pointNetDataset import ModelNet, RandomRotateTransform, RandomJitterTransform, ScaleTransform
from functools import partial
from torchvision import transforms
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def train_loop_per_worker(config : dict, training_set = None, validation_set = None, n_epochs : int = 10):
    """
    Trainable loop to pass to tuner. 
    Expects training_set and validation_set Dataset objects, and a number of epochs.
    tune.with_parameters().

    Config keys:

    batch_size : int
    lr : float
    gamma : float
    model : dict of parameters passed to model init

    """

    # define loader
    
    train_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(validation_set,batch_size=config['batch_size'], shuffle=True)
    def get_model_net_40(datadir, batch_size, num_points):
        transform = transforms.Compose([
            RandomRotateTransform(),
            RandomJitterTransform(),
            ScaleTransform(),
        ])

        train_data = ModelNet(datadir, split='train', num_points=num_points, transform=transform)
        test_data = ModelNet(datadir, split='test', num_points=num_points, transform=transform)

        return train_data,test_data

    
    # choose device
    # QUESTION: DOES THIS INTERFERE WITH THE RESOURCES RAY ALLOCATES TO THE TRAINING LOOP? HOW DO THEY INTERACT?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init the net
    model = PointNet(**config["model"], use_dropout = False).to(device)
 

    # loss functions
 # Loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
   

    # optimizer
    # tune learning rate
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])   
    training_losses = []
    training_accuracies = []
    test_accuracies = []

        ############################################################### EPOCHS LOOP


    for epoch in range(n_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)  # Convert labels to torch.long

            optimizer.zero_grad()

            outputs = net(inputs)
            outputs = outputs.float()  # Ensure outputs have the correct data type
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_loss = running_loss / len(train_loader)
        training_accuracy = 100 * correct / total

        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        # Evaluate on the test set
        net.eval()
        correct = 0
        total = 0

    # Inside the evaluation loop
    with torch.no_grad():
        total_test_loss = 0.0  # Initialize total test loss to zero

        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)
            outputs = net(inputs)
            outputs = outputs.float()

            # Calculate the test loss for this batch
            loss = loss_func(outputs, labels)

            # Accumulate the test loss
            total_test_loss += loss.item()

        # Calculate the average test loss over all batches
        average_test_loss = total_test_loss / len(test_loader)
        
        # Append the average test loss to the list
        test_losses.append(average_test_loss)

#         print(f'Epoch [{epoch + 1}/{n_epochs}] '
#               f'Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}% '
#               f'Test Accuracy: {test_accuracy:.2f}%')

        ############################################# REPORT AND CHECKPOINT

                # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("pointnet", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "pointnet/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("point")
        session.report({"val_loss": test_losses, 
                       "training_accuracy": training_accuracies,
                       "train_loss" : training_losses },
                       checkpoint=checkpoint)




if __name__ == '__main__':

    # BEFORE RUNNING THIS, INITIALIZE RAY CLUSTER WITH 
    # ray start --head 
    # on the head node

    # init ray to head
    ray.init('localhost:6379')
    
    # number of trials i guess
    num_samples = 20

    config = {
                "batch_size": tune.choice([8, 16, 32, 64, 128]),
                "lr": tune.loguniform(1e-5, 1e-1),
    
                "model" : {
                        "conv1a_out" : tune.choice([16,32,64]),
                        "conv2a_out" : tune.choice([16,32,64]),
                        "conv3a_out" : tune.choice([16,32,64]),
                        "conv4a_out" : tune.choice([16,32,64])
                }
            }

    scheduler = ASHAScheduler(
    # max_t=10, # isnt this the same as setting it in the trainable function? idk
    grace_period=3,
    reduction_factor=2)

    # initialize datasets to pass
    metadata_path = '/dataNfs/modelnet10/metadata.parquet'
   
    training_set, validation_set  = get_model_net_40(metadata_path, batch_size=config['batch_size'], num_points=1024)

    # max epochs per loop
    n_epochs=10
        
    tuner = tune.Tuner(
        # wrap the training loop in this
        tune.with_parameters(
            train_loop_per_worker,
            # parameters to pass to the training loop
            training_set = training_set,
            validation_set = validation_set,
            n_epochs=n_epochs),
        # tune configurations
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            # max_concurrent_trials=8 # maybe this is the number of processes it spawns?
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    # save results

    df = results.get_dataframe()

    df.to_csv('/home/ubuntu/nndlproject/ray_results.csv')
