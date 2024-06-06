from Train_Test import train_teacher, train_student, test
import torch
import torch.nn as nn
import torch.optim as optim

class TrainKD():
    def __init__(self, 
                 teacher_model, 
                 student_model,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 mode:str="Classification"):
        """
        Initialize the KD Toolkit.

        Args:
        teacher_model: teacher model
        student_model: student model
        mode: Choose from "Classification", "Regression"
        """

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.mode = mode

########################################################################################

    def train(self, 
              dataset, 
              alpha:float=0.5, 
              beta:float=0.5, 
              optimizer:str="Adam", 
              lr:float=0.001,
              criterion:str="CrossEntropyLoss",
              teacher_epochs:int=20,
              student_epochs:int=20,
              scheduler:str="None"):
        """
        Train the student model with knowledge distillation.

        Args:
        dataset: Tuple of train and test dataloaders
        alpha: Weight for the soft target loss
        beta: Weight for the hard target loss
        optimizer: Optimizer. Choose from "Adam", "SGD", "AdamW"
        lr: Learning rate
        criterion: Loss function. Choose from "CE", "MSE", "KL"
        teacher_epochs: Number of epochs to train the teacher model
        student_epochs: Number of epochs to train the student model
        scheduler: Learning rate scheduler. Choose from "None", "StepLR", "MultiStepLR", "ExponentialLR"

        Returns:
        student_model: trained student model
        """
        
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        train_loader, test_loader = dataset
        self.test_loader = test_loader

        if optimizer == "Adam":
            optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = optim.SGD(self.student_model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            optimizer = optim.AdamW(self.student_model.parameters(), lr=lr)
        else:
            raise ValueError("Invalid optimizer")
        
        if criterion == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion == "KL":
            self.criterion = nn.KLDivLoss()
        else:
            raise ValueError("Invalid criterion")

        if scheduler == "None":
            scheduler = None
        elif scheduler == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
        elif scheduler == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            raise ValueError("Invalid scheduler")

        self.teacher_performance = train_teacher(self.teacher_model, self.device, train_loader, optimizer, teacher_epochs, criterion, scheduler)
        self.student_performance = train_student(self.student_model, self.teacher_model, self.device, train_loader, optimizer, alpha, beta, student_epochs, scheduler)
        test(self.student_model, self.device, test_loader, criterion, mode=self.mode)
        
        return self.student_model

    def save_model(self, path:str="./models"):
        """
        Save the student model

        Args:
        path: path to save the student model. Default is "./models"
        
        Returns:
        None
        """
        import os
        from time import time

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.student_model.state_dict(), path + f"/student_model_{time}.pth")

########################################################################################

    def model_compare(self):
        """
        Compare the teacher and student models in terms of accuracy, loss and number of parameters

        Args:
        None

        Returns:
        None
        """
        from tabulate import tabulate

        if self.mode == "Classification":
            teacher_acc, teacher_loss = test(self.teacher_model, self.device, self.test_loader, self.criterion, mode=self.mode)
            student_acc, student_loss = test(self.student_model, self.device, self.test_loader, self.criterion, mode=self.mode)
        
            table = [
                ["Model", "Accuracy", "Loss", "Number of Parameters"],
                ["Teacher Model", teacher_acc, teacher_loss, sum(p.numel() for p in self.teacher_model.parameters())], 
                ["Student Model", student_acc, student_loss, sum(p.numel() for p in self.student_model.parameters())]
            ]
        
        elif self.mode == "Regression":
            teacher_loss = test(self.teacher_model, self.device, self.test_loader, self.criterion, mode=self.mode)
            student_loss = test(self.student_model, self.device, self.test_loader, self.criterion, mode=self.mode)
        
            table = [
                ["Model", "Loss", "Number of Parameters"],
                ["Teacher Model", teacher_loss, sum(p.numel() for p in self.teacher_model.parameters())], 
                ["Student Model", student_loss, sum(p.numel() for p in self.student_model.parameters())]
            ]
        
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

########################################################################################

    def plot_loss(self, path:str="./results", savefig:bool=False):
        """
        Plot the loss curve of the teacher and student models

        Args:
        path: path to save the loss curve. Default is "./results"
        savefig: Save the plot. Default is False

        Returns:
        None
        """
        import os
        import matplotlib.pyplot as plt

        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure(figsize=(10, 6))
        plt.plot(self.teacher_performance, label="Teacher Model")
        plt.plot(self.student_performance, label="Student Model")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        if savefig:
            plt.savefig(path + "/Loss_curve.png")