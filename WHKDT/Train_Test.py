from Knowledge_Distillation import knowledge_distillation_loss
import torch.nn.functional as F
import torch
from tqdm import tqdm

########################################################################################

def train_teacher(model, 
                  device, 
                  train_loader, 
                  optimizer, 
                  epochs,
                  criterion):
    
    model.train()
    train_loss = []

    for epoch in range(epochs):
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]") as pbar:
            for _, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': loss.item()})
        train_loss.append(loss.item())
    
    return train_loss

########################################################################################

def train_student(student_model, 
                  teacher_model, 
                  device, 
                  train_loader, 
                  optimizer, 
                  alpha, 
                  beta, 
                  epochs):
    
    student_model.train()
    teacher_model.eval()

    train_loss = []

    for epoch in range(epochs):
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]") as pbar:
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                teacher_output = teacher_model(data)
                student_output = student_model(data)
                loss = knowledge_distillation_loss(teacher_output, student_output, target, alpha, beta)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': loss.item()})
        train_loss.append(loss.item())
    
    return train_loss

########################################################################################

# Test function for both models
def test(model, 
         device, 
         test_loader, 
         criterion,
         mode:str="Classification"):
    
    model.eval()

    loss = 0
    correct = 0

    if mode not in ["Classification", "Regression"]:
        raise ValueError("Invalid mode")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            if mode == "Classification":
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    if mode == "Classification":
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    else:
        print(f'Test set: Average loss: {loss:.4f}')


########################################################################################