import torch

def get_accuracy_counts(outputs: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """
    Calculates the number of correct Top-1 and Top-5 predictions for a batch.
    
    Args:
        outputs (torch.Tensor): Logits from the model (batch_size, num_classes)
        labels (torch.Tensor): True labels (batch_size,)
        
    Returns:
        tuple: (top1_correct_count, top5_correct_count)
    """
    with torch.no_grad():
        # maxk: (batch_size, 5) - indices of the top 5 predictions
        _, maxk = torch.topk(outputs, 5, dim=1)

        # labels_resize: (batch_size, 1) - true labels
        labels_resize = labels.view(-1, 1)

        # correct_matrix: (batch_size, 5) 
        # correct_matrix[i, j] = 1 if the j-th prediction of the i-th sample is correct
        correct_matrix = maxk == labels_resize

        # total top1 correct is just the sum of the first column of the correct_matrix
        top1 = correct_matrix[:, 0].sum().item()
        
        # total top5 correct is the sum of all columns (any match in top 5)
        top5 = correct_matrix.sum().item()
        
        return top1, top5