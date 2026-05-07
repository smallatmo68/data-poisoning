import torch
from torch.utils.data import DataLoader

from barbar import Bar

def get_predictions(model, dataset, batch_size=128, num_workers=0):
    all_preds = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    correct_preds = 0
    with torch.no_grad():
        for b_x, b_y in Bar(dataloader):
            y_preds = torch.softmax(model(b_x), dim=1)
            correct_preds += torch.sum(y_preds.argmax(dim=1) == b_y).item()
            all_preds.extend(y_preds.detach().cpu().numpy())
    acc = correct_preds / len(dataset)
    return all_preds, acc
