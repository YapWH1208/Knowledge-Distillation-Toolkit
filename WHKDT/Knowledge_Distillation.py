import torch.nn.functional as F

def knowledge_distillation_loss(teacher_logits, student_logits, true_labels, alpha=0.5, beta=0.5):
    """
    Compute the knowledge distillation loss as a linear combination of the soft target loss and the mean squared error loss.

    Args:
    teacher_logits: output of teacher model
    student_logits: output of student model
    true_labels: true labels
    alpha: weight for soft target loss
    beta: weight for mean squared error loss

    Returns:
    knowledge distillation loss
    """
    soft_target_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction='batchmean')
    mse_loss = F.mse_loss(student_logits, F.one_hot(true_labels, num_classes=student_logits.size(1)).float(), reduction='mean')
    distillation_loss = alpha * soft_target_loss + beta * mse_loss
    
    return distillation_loss