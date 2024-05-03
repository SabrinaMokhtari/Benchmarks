import torch
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from torchmetrics.functional.classification import multilabel_accuracy
import numpy as np
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix
import logging
from datetime import datetime
import time
import pandas as pd

from numpy.linalg import norm
import datetime
from statistics import mean
from opacus.grad_sample.functorch import make_functional
from torch.func import grad, grad_and_value, vmap
from opacus.grad_sample import GradSampleModule

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits  # with sigmoid

    def forward(self, y_pred, y_true, reduction='mean'):  # TODO: handle the tensor shapes
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        return self.criterion(y_pred, y_true, reduction=reduction)


def get_device():
    use_cuda = torch.cuda.is_available()
    print("use_cuda", use_cuda)
    # assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, train_loader, optimizer, ema, max_physical_batch_size=128, grad_sample_mode="no_op"):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    train_pred_cpu = []
    train_true_cpu = []

    losses = []

    print(f'Train Loader: {len(train_loader)}, {train_loader.batch_size}')

    if grad_sample_mode == "no_op":
        print("Using grad_sample_mode: no_op")
        # Functorch prepare
        fmodel, _fparams = make_functional(model)

        def compute_loss_stateless_model(params, sample, target):
            # import pdb; pdb.set_trace()
            repeated_target = target.repeat(sample.shape[0], 1)
            repeated_target = repeated_target.squeeze(1)

            predictions = fmodel(params, sample)
            loss = F.cross_entropy(predictions, repeated_target)
            return loss

        ft_compute_grad = grad_and_value(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # Using model.parameters() instead of fparams
        # as fparams seems to not point to the dynamically updated parameters
        params = list(model.parameters())

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer
    ) as memory_safe_data_loader:
        print(f"training on {len(memory_safe_data_loader)} batches of size {max_physical_batch_size}")
        for batch_idx, (data, target) in enumerate(memory_safe_data_loader):

            data, target = data.to(device), target.to(device) # squeeze for ffcv

            if grad_sample_mode == "no_op":
                # print("Using grad_sample_mode: no_op")
                per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                    params, data, target
                )
                per_sample_grads = [g.detach() for g in per_sample_grads]
                loss = torch.mean(per_sample_losses)
                for p, g in zip(params, per_sample_grads):
                    p.grad_sample = g
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            if ema:
                ema.update()

            with torch.no_grad():
                if grad_sample_mode == "no_op":
                    orignial_data = data[:, 0]
                else:
                    orignial_data = data
                original_output = model(orignial_data)

                pred = original_output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                num_examples += len(data)

                # calculate AUC
                probs = torch.softmax(original_output, dim=1)
                train_pred_cpu.append(probs.cpu().detach().numpy())
                train_true_cpu.append(target.cpu().detach().numpy())
            
            del data, target, original_output, orignial_data
            torch.cuda.empty_cache()

    train_acc = 100. * correct / num_examples

    train_loss = mean(losses)


    train_pred_cpu = np.concatenate(train_pred_cpu)
    train_true_cpu = np.concatenate(train_true_cpu)
    val_auc_mean = roc_auc_score(train_true_cpu, train_pred_cpu, average='weighted', multi_class='ovr')
    print(f'Train set: Average loss: {train_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)', f'AUC: {val_auc_mean}')

    # Get the current time
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {time_string}")

    return train_loss, train_acc, val_auc_mean # change 0 to multi_label_acc for chexpert

def CheXpert_train(model, train_loader, optimizer, ema, max_physical_batch_size=128, grad_sample_mode="no_op"):
    device = next(model.parameters()).device
    model.train()
    train_loss = 0
    num_examples = 0

    CELoss = CrossEntropyLoss()
    train_pred = []
    train_true = []

    train_pred_cpu = []
    train_true_cpu = []

    losses = []

    print(f'Train Loader: {len(train_loader)}, {train_loader.batch_size}')

    if grad_sample_mode == "no_op":
        # Functorch prepare
        fmodel, _fparams = make_functional(model)

        def compute_loss_stateless_model(params, sample, target):
            # batch = sample.unsqueeze(0)
            # targets = target.unsqueeze(0)
            # import pdb; pdb.set_trace()

            repeated_target = target.repeat(sample.shape[0], 1)

            predictions = fmodel(params, sample)
            loss = CELoss(predictions, repeated_target)
            return loss

        ft_compute_grad = grad_and_value(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # Using model.parameters() instead of fparams
        # as fparams seems to not point to the dynamically updated parameters
        params = list(model.parameters())


    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer
    ) as memory_safe_data_loader:
        print(f"training on {len(memory_safe_data_loader)} batches of size {max_physical_batch_size}")
        for batch_idx, (data, target) in enumerate(memory_safe_data_loader):

            print("Data Shape: ", data.shape, target.shape)

            data, target = data.to(device), target.to(device) # squeeze for ffcv

            if grad_sample_mode == "no_op":
                print("Using grad_sample_mode: no_op")
                per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                    params, data, target
                )
                per_sample_grads = [g.detach() for g in per_sample_grads]
                loss = torch.mean(per_sample_losses)
                for p, g in zip(params, per_sample_grads):
                    p.grad_sample = g
            else:
                output = model(data)
                loss = CELoss(output, target)
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if ema:
                ema.update()

            losses.append(loss.item())

            with torch.no_grad():
                if grad_sample_mode == "no_op":
                    orignial_data = data[:, 0]
                else:
                    orignial_data = data
                original_output = model(orignial_data)
                # import pdb; pdb.set_trace()
                # calculate multi-label accuracy
                train_pred.append(original_output)
                train_true.append(target)

                # calculate AUC
                train_pred_cpu.append(original_output.cpu().detach().numpy())
                train_true_cpu.append(target.cpu().detach().numpy())
            
            del data, target, original_output, orignial_data
            torch.cuda.empty_cache()



    train_loss = mean(losses)
        
    train_pred = torch.cat(train_pred)
    train_true = torch.cat(train_true)
    multi_label_acc = multilabel_accuracy(train_pred, train_true, num_labels=train_true.size(dim=1), average=None).cpu()

    train_pred_cpu = np.concatenate(train_pred_cpu)
    train_true_cpu = np.concatenate(train_true_cpu)
    val_auc_mean = roc_auc_score(train_true_cpu, train_pred_cpu, average='weighted', multi_class='ovr')

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {time_string}")
    
    print(f'Train set: Average loss: {train_loss:.4f}', f'Accuracy: {multi_label_acc}', f'AUC: {val_auc_mean:.4f}')

    return train_loss, multi_label_acc, val_auc_mean # change 0 to multi_label_acc for chexpert

def test(model, test_loader, ema):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    ema_loss = 0
    correct = 0
    test_CELoss = CrossEntropyLoss()
    test_pred = []
    test_true = []

    test_pred_cuda = []
    test_true_cuda = []

    ema_pred = []
    ema_true = []

    ema_pred_cuda = []
    ema_true_cuda = []

    ema_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device) # squeeze for ffcv
            output = model(data)
            test_probs = torch.softmax(output, dim=1)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

            test_pred.append(test_probs.cpu().detach().numpy()) # change test_probs to output for other than eyepacs
            test_true.append(target.cpu().detach().numpy())

            if ema:
                # Validation: with EMA
                # the .average_parameters() context manager
                # (1) saves original parameters before replacing with EMA version
                # (2) copies EMA parameters to model
                # (3) after exiting the `with`, restore original parameters to resume training later
                with ema.average_parameters():
                    logits = model(data)
                    ema_loss += F.cross_entropy(logits, target, reduction='sum').item()
                    ema_pred_temp = logits.max(1, keepdim=True)[1]
                    ema_correct += ema_pred_temp.eq(target.view_as(ema_pred_temp)).sum().item()
                    
                    ema_probs = torch.softmax(logits, dim=1)
                    ema_pred.append(ema_probs.cpu().detach().numpy()) # change test_probs to output for other than eyepacs
                    ema_true.append(target.cpu().detach().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc_mean = roc_auc_score(test_true, test_pred, average='weighted', multi_class='ovr')

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}', f'Test Accuracy {correct}/{num_examples}: {test_acc:.4f}', f'AUC: {val_auc_mean}')

    if ema:
        ema_true = np.concatenate(ema_true)
        ema_pred = np.concatenate(ema_pred)
        ema_auc_mean = roc_auc_score(ema_true, ema_pred, average='weighted', multi_class='ovr')

        ema_loss /= num_examples
        ema_acc = 100. * ema_correct / num_examples
        print(f'EMA set: Average loss: {ema_loss:.4f}', f'Test Accuracy {ema_correct}/{num_examples}: {ema_acc:.4f}', f'AUC: {ema_auc_mean}')

    # threshold = 0.5
    # binary_labels_test = (test_pred > threshold).astype(int)
    # confusion_matrix = multilabel_confusion_matrix(test_true, binary_labels_test)
    # for i, cm in enumerate(confusion_matrix):
    #     print(f"Confusion matrix for label {i}:")
    #     print(cm)

    if ema:
        return test_loss, test_acc, val_auc_mean, ema_loss, ema_acc, ema_auc_mean # change 0 to multi_label_acc for chexpert
    else:
        return test_loss, test_acc, val_auc_mean, 0, 0, 0
    # return test_loss, test_acc, val_auc_mean # change 0 to multi_label_acc for chexpert

def CheXpert_test(model, test_loader, ema):
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    ema_loss = 0
    correct = 0
    num_examples = 0
    test_CELoss = CrossEntropyLoss()
    test_pred = []
    test_true = []

    test_pred_cuda = []
    test_true_cuda = []

    ema_pred = []
    ema_true = []

    ema_pred_cuda = []
    ema_true_cuda = []

    # import pdb; pdb.set_trace()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device) # squeeze for ffcv
            output = model(data)

            test_loss += test_CELoss(output, target, reduction='sum').item()
            num_examples += len(data)

            test_pred.append(output.cpu().detach().numpy()) # change test_probs to output for other than eyepacs
            test_true.append(target.cpu().detach().numpy())

            test_pred_cuda.append(output)
            test_true_cuda.append(target)

            if ema:
                # Validation: with EMA
                # the .average_parameters() context manager
                # (1) saves original parameters before replacing with EMA version
                # (2) copies EMA parameters to model
                # (3) after exiting the `with`, restore original parameters to resume training later
                with ema.average_parameters():
                    logits = model(data)
                    ema_loss += test_CELoss(logits, target, reduction='sum').item()
                

                    ema_pred.append(logits.cpu().detach().numpy()) # change test_probs to output for other than eyepacs
                    ema_true.append(target.cpu().detach().numpy())

                    ema_pred_cuda.append(logits)
                    ema_true_cuda.append(target)

    test_loss /= num_examples

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc_mean = roc_auc_score(test_true, test_pred, average='weighted', multi_class='ovr')

    test_pred_cuda = torch.cat(test_pred_cuda)
    test_true_cuda = torch.cat(test_true_cuda)
    multi_label_acc = multilabel_accuracy(test_pred_cuda, test_true_cuda, num_labels=test_true_cuda.size(dim=1), average=None).cpu()

    print(f'Test set: Loss: {test_loss}', f'Multi-Label Accuracy: {multi_label_acc}', f'val_auc_mean: {val_auc_mean}')

    if ema:
        ema_loss /= num_examples

        ema_true = np.concatenate(ema_true)
        ema_pred = np.concatenate(ema_pred)
        ema_auc_mean = roc_auc_score(ema_true, ema_pred, average='weighted', multi_class='ovr')

        ema_pred_cuda = torch.cat(ema_pred_cuda)
        ema_true_cuda = torch.cat(ema_true_cuda)
        ema_multi_label_acc = multilabel_accuracy(ema_pred_cuda, ema_true_cuda, num_labels=ema_true_cuda.size(dim=1), average=None).cpu()

        print(f'EMA set: Loss: {ema_loss}', f'Multi-Label Accuracy: {ema_multi_label_acc}', f'val_auc_mean: {ema_auc_mean}')

    if ema:
        return test_loss, multi_label_acc, val_auc_mean, ema_loss, ema_multi_label_acc, ema_auc_mean # change 0 to multi_label_acc for chexpert
    else:
        return test_loss, multi_label_acc, val_auc_mean, 0, 0, 0