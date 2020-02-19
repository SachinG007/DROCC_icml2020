import torch
import torch.nn.functional as F

def eval(model, device, data_loader, print_string):
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float)
            target= target.to(torch.long)
            target = torch.squeeze(target)
            output = model(data)
            loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    print(print_string + ': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy
