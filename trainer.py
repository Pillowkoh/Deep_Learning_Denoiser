import torch
import numpy as np
from tqdm import tqdm

class Trainer(object):
    """
    Trainer class to coordinate all training processes

    Args:
        - model (torch.nn.Module): Denoiser model
        - loss_fn (torch.nn.Module): Module to calculate loss between prediction and ground truth
        - optimizer (torch.nn.Module): Module to update model weights
        - epochs (int): Number of training epochs
        - config (dict): Dictionary containing all training config settings
        - scheduler (torch.nn.Module): Module to reduce learning rate if no improvement
    """
    
    def __init__(self, model, loss_fn, optimizer, epochs, config, scheduler = None):
        self.model = model
        self.loss = {"train":[], "val":[]}
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = scheduler
        self.checkpoint_frequency = 1
        self.early_stopping_epochs = 10
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.batches_per_epoch = config["batches_per_epoch"]
        self.batches_per_epoch_val = config["batches_per_epoch_val"]

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                )
            )

            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            # saving model
            if (epoch + 1) % self.checkpoint_frequency == 0:
                torch.save(
                    self.model.state_dict(), "model_{}".format(str(epoch + 1).zfill(3))
                )

            # early stopping
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]), 
                                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0
                    #print('New min: ', min_val_loss)

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), "model_final")
        return self.model


    def _epoch_train(self, dataloader):
        self.model.train()

        for i, data in enumerate(tqdm(dataloader), 0):
            # print(f"batch #{i} --- training")
            inputs = data["noisy"].to(self.device)
            clean = data["clean"].to(self.device)

            self.optimizer.zero_grad()
            running_loss = []
            denoised = self.model(inputs)

            loss = 0

            for j in range(self.batch_size):
                loss += self.loss_fn(clean[j], denoised[j])
                          
            loss/=self.batch_size

            running_loss.append(loss.item())

            loss.backward()                            
            self.optimizer.step()

            if i == self.batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                self.loss["train"].append(epoch_loss)

        print('Training epoch complete')

    def _epoch_eval(self, dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader), 0):
                inputs = data["noisy"].to(self.device)
                clean = data["clean"].to(self.device)

                denoised = self.model(inputs)

                loss = 0
                for j in range(self.batch_size):
                    loss += self.loss_fn(clean[j], denoised[j])

                loss/=self.batch_size
                running_loss.append(loss.item())

                if i == self.batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self.loss["val"].append(epoch_loss)

        print('Eval epoch complete')