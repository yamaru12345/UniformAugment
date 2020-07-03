import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader


def predict(model, dataset, batch_size, device=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred = []
    with torch.no_grad():
        for data in dataloader:
            # デバイスの指定
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)

            # 順伝播
            outputs = model.forward(inputs)

            # 予測値算出
            pred.append(outputs)
  
    pred = torch.cat(pred)

    return pred


def visualize_logs(log, fig_path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log['train']).T[0], label='train')
    ax[0].plot(np.array(log['valid']).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log['train']).T[1], label='train')
    ax[1].plot(np.array(log['valid']).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    fig.savefig(fig_path)
