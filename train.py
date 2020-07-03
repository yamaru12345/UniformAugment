import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def calculate_loss_and_accuracy(model, dataset, batch_size, device=None, criterion=None):
  """損失・正解率を計算"""
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  loss = 0.0
  correct = 0
  with torch.no_grad():
    for data in dataloader:
      # デバイスの指定
      inputs = data[0].to(device)
      labels = data[1].to(device)

      # 順伝播
      outputs = model.forward(inputs)

      # 損失計算
      if criterion != None:
        loss += criterion(outputs, labels).item()

      # 正解率計算
      pred = torch.argmax(outputs, dim=-1)
      correct += (pred == labels).sum().item()
      
  return loss / len(dataloader), correct / len(dataset)
  

def train_model(model_id, dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, base_dir, sampler=None, device=None, patience=15):
  """モデルの学習を実行し、損失・正解率のログを返す"""
  # デバイスの指定
  model.to(device)

  # dataloaderの作成
  if sampler != None:
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
  else:
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

  # スケジューラの設定
  #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

  # 学習
  log_train = []
  log_valid = []
  for epoch in range(num_epochs):
    # 開始時刻の記録
    s_time = time.time()
    
    if epoch == int(num_epochs / 3):
      # 前半ベストをセットして上流をunfreeze
      model.load_state_dict(torch.load(Path(base_dir, f'state_dict_{model_id}.pt'), map_location=device))
      for param in model.parameters():
        param.requires_grad = True
      for g in optimizer.param_groups:
        g['lr'] = 1e-5

    # 訓練モードに設定
    model.train()
    for data in dataloader_train:
      # 勾配をゼロで初期化
      optimizer.zero_grad()

      # 順伝播 + 誤差逆伝播 + 重み更新
      inputs = data[0].to(device)
      labels = data[1].to(device)
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    
    # 評価モードに設定
    model.eval()

    # 損失と正解率の算出
    loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, batch_size, device, criterion=criterion)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, batch_size, device, criterion=criterion)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # チェックポイントの保存
    if epoch == 0:
      saved = True
      min_loss = loss_valid
      torch.save(model.state_dict(), Path(base_dir, f'state_dict_{model_id}.pt'))
      counter = 0
    elif loss_valid < min_loss:
      saved = True
      min_loss = loss_valid
      torch.save(model.state_dict(), Path(base_dir, f'state_dict_{model_id}.pt'))
      counter = 0
    else:
      saved = False
      counter += 1

    # 終了時刻の記録
    e_time = time.time()

    # ログを出力
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, saved: {saved}, {(e_time - s_time):.4f}sec') 
      
    # スケジューラを1ステップ進める
    #scheduler.step()
    
    # Early Stopping
    if counter == patience:
      break

  return {'train': log_train, 'valid': log_valid}
