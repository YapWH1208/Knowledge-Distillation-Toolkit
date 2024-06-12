from Train_Test import train_teacher, train_student, test
import torch
import torch.nn as nn
import torch.optim as optim

class TrainKD():
    def __init__(self, 
                 task:str,
                 mode:str,
                 student_model,
                 teacher_model=None, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),):
        """
        Initialize the KD Toolkit.

        Args:
        task: Task to perform. Choose from "Classification", "Regression"
        mode: Mode of KD. Choose from "Online", "Offline", "Self"
        student_model: Student model
        teacher_model: Teacher model. Default is None
        device: Device to run the models. Default is 'cuda' if available else 'cpu'
        """

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device

        if task in ["Classification", "Regression"]:
            self.task = task
        else:
            raise ValueError("Invalid mode. Choose from 'Classification', 'Regression'")
        
        if mode in ["Online", "Offline", "Self"]:
            self.mode = mode
        else:
            raise ValueError("Invalid mode. Choose from 'Online', 'Offline', 'Self'")

########################################################################################

    def train(self, 
              dataset, 
              alpha:float=0.5, 
              beta:float=0.5,
              teacher_epochs:int=20,
              student_epochs:int=20):
        """
        Train the student model with knowledge distillation.

        Args:
        dataset: Tuple of train and test dataloaders
        alpha: Weight for the soft target loss
        beta: Weight for the hard target loss
        teacher_epochs: Number of epochs to train the teacher model
        student_epochs: Number of epochs to train the student model

        Returns:
        student_model: trained student model
        """

        if not self.mode == "Self":
            self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        if self.task == "Classification":
            self.create_optimizer("Adam", 0.001)
            self.create_criterion("CE")
            self.create_scheduler("None")
        elif self.task == "Regression":
            self.create_optimizer("Adam", 0.001)
            self.create_criterion("MSE")
            self.create_scheduler("None")
        
        if self.mode == "Online":
            self.train_online(dataset, alpha, beta, teacher_epochs, student_epochs)
        elif self.mode == "Offline":
            self.train_offline(dataset, alpha, beta, student_epochs)
        elif self.mode == "Self":
            self.train_self(dataset, alpha, beta, student_epochs)
        
        return self.student_model
    
########################################################################################

    def train_online(self, 
                     dataset, 
                     alpha:float=0.5, 
                     beta:float=0.5,
                     teacher_epochs:int=20,
                     student_epochs:int=20):
        """
        Train both the teacher model and student model with knowledge distillation.

        Args:
        dataset: Tuple of train and test dataloaders
        alpha: Weight for the soft target loss
        beta: Weight for the hard target loss
        teacher_epochs: Number of epochs to train the teacher model
        student_epochs: Number of epochs to train the student model

        Returns:
        student_model: trained student model
        """
        
        train_loader, test_loader = dataset
        self.test_loader = test_loader

        self.teacher_performance = train_teacher(self.teacher_model, self.device, train_loader, self.optimizer, teacher_epochs, self.criterion, self.scheduler)
        self.student_performance = train_student(self.student_model, self.teacher_model, self.device, train_loader, self.optimizer, alpha, beta, student_epochs, self.scheduler)
        test(self.student_model, self.device, test_loader, self.criterion, mode=self.task)
        
        return self.student_model
    
    def train_offline(self,
                      dataset,
                      alpha:float=0.5,
                      beta:float=0.5,
                      student_epochs:int=20):
        """
        Train the student model with knowledge distillation in offline mode.

        Args:
        dataset: Tuple of train and test dataloaders
        alpha: Weight for the soft target loss
        beta: Weight for the hard target loss
        student_epochs: Number of epochs to train the student model

        Returns:
        student_model: trained student model
        """

        train_loader, test_loader = dataset
        self.test_loader = test_loader

        self.student_performance = train_student(self.student_model, self.teacher_model, self.device, train_loader, self.optimizer, alpha, beta, student_epochs, self.scheduler)
        test(self.student_model, self.device, test_loader, self.criterion, mode=self.task)
        
        return self.student_model
    
    def train_self(self,
                   dataset,
                   alpha:float=0.5,
                   beta:float=0.5,
                   student_epochs:int=20):
        """
        Train the student model with knowledge distillation in self mode.

        Args:
        dataset: Tuple of train and test dataloaders
        alpha: Weight for the soft target loss
        beta: Weight for the hard target loss
        student_epochs: Number of epochs to train the student model

        Returns:
        student_model: trained student model
        """

        train_loader, test_loader = dataset
        self.test_loader = test_loader

        self.student_performance = train_student(self.student_model, self.student_model, self.device, train_loader, self.optimizer, alpha, beta, student_epochs, self.scheduler)
        test(self.student_model, self.device, test_loader, self.criterion, mode=self.task)
        
        return self.student_model

################################################################################

    def create_optimizer(self, optimizer_name, lr, **kwargs):
        """
        Creates an optimizer instance based on the provided name and learning rate.

        Args:
        optimizer_name: Name of the optimizer. Choose from "Adam", "SGD", "AdamW", "Adadelta", "Adagrad", "Adamax", "RMSprop"
        lr: Learning rate
        **kwargs: Additional arguments for the optimizer

        Returns:
        optimizer: Optimizer instance
        """

        optimizers = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "AdamW": optim.AdamW,
            "Adadelta": optim.Adadelta,
            "Adagrad": optim.Adagrad,
            "Adamax": optim.Adamax,
            "RMSprop": optim.RMSprop
        }

        if optimizer_name not in optimizers:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        self.optimizer = optimizers[optimizer_name](self.student_model.parameters(), lr=lr, **kwargs)
    
################################################################################

    def create_criterion(self, criterion_name):
        """
        Creates a criterion instance based on the provided name.

        Args:
        criterion_name: Name of the criterion. Choose from "CE", "MSE", "KL", "BCE", "BCEWithLogits", "CTC", "NLL", "Poisson", "SmoothL1"
        **kwargs: Additional arguments for the criterion

        Returns:
        criterion: Criterion instance
        """

        criterion = {
            "CE": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss(),
            "KL": nn.KLDivLoss(),
            "BCE": nn.BCELoss(),
            "BCEWithLogits": nn.BCEWithLogitsLoss(),
            "CTC": nn.CTCLoss(),
            "NLL": nn.NLLLoss(),
            "Poisson": nn.PoissonNLLLoss(),
            "SmoothL1": nn.SmoothL1Loss(),
        }

        if criterion_name not in criterion:
            raise ValueError(f"Invalid criterion name: {criterion_name}")
        
        self.criterion = criterion[criterion_name]

################################################################################

    def create_scheduler(self, scheduler_name, **kwargs):
        """
        Creates a scheduler instance based on the provided name.

        Args:
        scheduler_name: Name of the scheduler. Choose from "None", "StepLR", "MultiStepLR", "ExponentialLR", "ReduceLROnPlateau", "CyclicLR", "OneCycleLR", "CosineAnnealingLR"
        **kwargs: Additional arguments for the scheduler

        Returns:
        scheduler: Scheduler instance
        """
        
        if self.optimizer is None:
            raise ValueError("Optimizer not found. Please create an optimizer first.")
        
        scheduler = {
            "None": None,
            "StepLR": optim.lr_scheduler.StepLR,
            "MultiStepLR": optim.lr_scheduler.MultiStepLR,
            "ExponentialLR": optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
            "CyclicLR": optim.lr_scheduler.CyclicLR,
            "OneCycleLR": optim.lr_scheduler.OneCycleLR,
            "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR
        }

        if scheduler_name not in scheduler:
            raise ValueError(f"Invalid scheduler name: {scheduler_name}")
        
        
        
        self.scheduler = scheduler[scheduler_name](self.optimizer, **kwargs) if scheduler_name != "None" else None

########################################################################################

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

        if self.task == "Classification":
            teacher_acc, teacher_loss = test(self.teacher_model, self.device, self.test_loader, self.criterion, mode=self.task)
            student_acc, student_loss = test(self.student_model, self.device, self.test_loader, self.criterion, mode=self.task)
        
            table = [
                ["Model", "Accuracy", "Loss", "Number of Parameters"],
                ["Teacher Model", teacher_acc, teacher_loss, sum(p.numel() for p in self.teacher_model.parameters())], 
                ["Student Model", student_acc, student_loss, sum(p.numel() for p in self.student_model.parameters())]
            ]
        
        elif self.task == "Regression":
            teacher_loss = test(self.teacher_model, self.device, self.test_loader, self.criterion, mode=self.task)
            student_loss = test(self.student_model, self.device, self.test_loader, self.criterion, mode=self.task)
        
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
        from time import time
        import matplotlib.pyplot as plt

        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure(figsize=(10, 6))

        if self.mode == "Online":
            plt.plot(self.teacher_performance, label="Teacher Model")
            
        plt.plot(self.student_performance, label="Student Model")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        if savefig:
            plt.savefig(path + f"/Loss_curve_{time}.png")