from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_train_valid_loss(train_loss: List[float], valid_loss: List[float], plot_path):
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, 'g', label='Train Loss')
    plt.plot(epochs, valid_loss, 'b', label='Valid Loss')
    plt.title('Training & Validation results')
    plt.xlabel('Epochs')
    plt.xticks([x - 1 for x in epochs[::10]])
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid()
    plt.savefig(plot_path, dpi=300)
    plt.clf()


def plot_metrics(accuracy, inform, mcc, plot_path):
    for phase in ['train', 'valid']:
        accuracy_p, inform_p, mcc_p = accuracy[phase], inform[phase], mcc[phase]
        epochs = list(range(1, len(accuracy_p) + 1))
        plt.plot(epochs, accuracy_p, 'r', label='Accuracy')
        plt.plot(epochs, inform_p, 'g', label='Informedness')
        plt.plot(epochs, mcc_p, 'b', label="Matthew's CC")
        plt.title(f'Classification Metrics ({phase})')
        plt.xlabel('Epochs')
        plt.xticks([x - 1 for x in epochs[::10]])
        plt.ylabel('Metrics')
        plt.ylim((-1, 1))
        plt.legend()
        plt.grid()
        plt.savefig(plot_path.replace('.png', f'_{phase}.png'), dpi=300)
        plt.clf()


def metrics(pred_b, y_b) -> torch.Tensor:
    predicted_res = pred_b.clone().detach()
    predicted_res = torch.gt(predicted_res, 0.5).detach()
    bool_pred_b = torch.gt(predicted_res, 0.5)
    int_pred_b = bool_pred_b.int()
    int_y_b = y_b.int()

    true_positive_tensor = int_pred_b & int_y_b
    true_positive = torch.mean(true_positive_tensor.float()).detach().item()

    false_positive_tensor = int_pred_b & (1 - int_y_b)
    false_positive = torch.mean(false_positive_tensor.float()).detach().item()

    true_negative_tensor = (1 - int_pred_b) & (1 - int_y_b)
    true_negative = torch.mean(true_negative_tensor.float()).detach().item()

    false_negative_tensor = (1 - int_pred_b) & int_y_b
    false_negative = torch.mean(false_negative_tensor.float()).detach().mean()

    return torch.tensor([true_positive, false_positive, true_negative, false_negative])


def informedness(metrics_arr):
    tp, fp, tn, fn = metrics_arr
    return (tp / (tp + fn)) + (tn / (tn + fp)) - 1


def matthews_correlation_coefficient(metrics_arr):
    tp, fp, tn, fn = metrics_arr
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0
    return numerator / denominator


def metrics_to_rates(metrics_tensor: torch.Tensor) -> List[float]:
    arr = np.asarray(metrics_tensor)
    tp, fp, tn, fn = arr
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = 1 - tnr
    fnr = 1 - tpr
    return [tpr, fpr, tnr, fnr]
