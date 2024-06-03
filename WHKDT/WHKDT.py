from Train_Test import train_teacher, train_student, test
import torch
import torch.nn as nn
import torch.optim as optim

class TrainKD():
    def __init__(self, 
                 teacher_model, 
                 student_model):
        """
        Train student model using knowledge distillation.

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
              criterion:nn.Module=nn.CrossEntropyLoss(),
              teacher_epochs:int=20,
              student_epochs:int=20):
        """
        Train student model using knowledge distillation.

        Args:
        dataset: tuple of train_loader and test_loader
        alpha: weight for soft target
        beta: weight for hard target
        optimizer: optimizer to use
        lr: learning rate
        criterion: loss function
        teacher_epochs: number of epochs to train teacher model
        student_epochs: number of epochs to train student model

        Returns:
        Trained student model
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

        train_teacher(self.teacher_model, self.device, train_loader, optimizer, teacher_epochs, criterion)
        train_student(self.student_model, self.teacher_model, self.device, train_loader, optimizer, alpha, beta, student_epochs)
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