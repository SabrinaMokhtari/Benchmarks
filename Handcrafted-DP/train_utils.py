import torch
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager

def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, train_loader, optimizer, n_acc_steps=1, max_physical_batch_size=256):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    #bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    #print(f"training on {num_batches} batches of size {bs}")
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
      for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
          optimizer.zero_grad()

          if batch_idx > num_batches - 1:
              break

          data, target = data.to(device), target.to(device)

          output = model(data)

          loss = F.cross_entropy(output, target)
          

          pred = output.max(1, keepdim=True)[1]
          correct += pred.eq(target.view_as(pred)).sum().item()
          train_loss += F.cross_entropy(output, target, reduction='sum').item()
          num_examples += len(data)

          loss.backward()
          optimizer.step()

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc
