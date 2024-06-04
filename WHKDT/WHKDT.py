from Train_Test import train_teacher, train_student, test
import torch
import torch.nn as nn
import torch.optim as optim

class TrainKD():
    def __init__(self, 
                 teacher_model, 
                 student_model):
        """
        Initialize the KD Toolkit.

        Args:
        teacher_model: teacher model
        student_model: student model
        """

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

        if optimizer == "Adam":
            optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = optim.SGD(self.student_model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            optimizer = optim.AdamW(self.student_model.parameters(), lr=lr)
        else:
            raise ValueError("Invalid optimizer")
        
        if criterion == "CE":
            criterion = nn.CrossEntropyLoss()
        elif criterion == "MSE":
            criterion = nn.MSELoss()
        elif criterion == "KL":
            criterion = nn.KLDivLoss()
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

        train_teacher(self.teacher_model, self.device, train_loader, optimizer, teacher_epochs, criterion, scheduler)
        train_student(self.student_model, self.teacher_model, self.device, train_loader, optimizer, alpha, beta, student_epochs, scheduler)
        test(self.student_model, self.device, test_loader, criterion, mode="Classification")
        
        return self.student_model

    def save_model(self, path:str):
        """
        Save the student model

        Args:
        path: path to save the student model
        
        Returns:
        None
        """
        import os
        from time import time

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.student_model.state_dict(), path + f"/student_model_{time}.pth")