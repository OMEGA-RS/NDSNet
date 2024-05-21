# import cv2
import os
import torch
import time
import copy
import numpy as np
from skimage import io
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from Defined_Functions import create_folder, normalization, ND, FLICM, postprocess, calculate_metric
from SSL_Dataset import SSLDataset
from CD_Dataset import CDDataset
from sklearn import metrics
from SSL_SimSiam import SSL, CDNet
import matplotlib.pyplot as plt


# Default parameter
batch_size = 256
patch_size = 5
max_epoch = 10
isTrain = True
isTest = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Start time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
start_time = time.time()

# -------------------------------------------------------------------------------------
# Step 0: Load Data
DATA_PATH = 'Data/Coastline'

SAVE_PATH = create_folder(DATA_PATH)

im1 = io.imread(os.path.join(DATA_PATH, 'im1.bmp'))
im2 = io.imread(os.path.join(DATA_PATH, 'im2.bmp'))
ref = io.imread(os.path.join(DATA_PATH, 'im3.bmp'), 'gray')

imgH, imgW = ref.shape

# --------------------------------------------------------------------------------------
# Step 1: Segment the DI
ND_DI = ND(im1, im2)
pre_map = FLICM(ND_DI, 3)

# --------------------------------------------------------------------------------------
# Step 2: Generate self-supervised training set
SSL_train_dataset = SSLDataset(im1, im2, pre_map, patch_size, isTrain)
SSL_train_dataloader = DataLoader(SSL_train_dataset, batch_size=batch_size)

# --------------------------------------------------------------------------------------
# Step 3: Self-supervised pretraining
SSL_model = SSL().to(device)
SSL_optimizer = torch.optim.Adam(SSL_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=5e-4)

print('-------------------Self-Supervised Pretraining-----------------------')

tmp_loss = 0.0

for epoch in range(1, max_epoch + 1):
    SSL_model.train()
    train_loss = 0
    epoch_start_time = time.time()

    for i, data in enumerate(SSL_train_dataloader):
        x1, x2 = SSL_model.set_input(data)
        SSL_optimizer.zero_grad()

        data_dict = SSL_model.forward(x1, x2)
        batch_loss = data_dict['loss'].mean()

        batch_loss.backward()
        SSL_optimizer.step()

        train_loss += batch_loss.item()

    train_loss = train_loss / len(SSL_train_dataloader)

    print(
        "End of epoch %d / %d \t Training loss is: %.4f \t Time Taken: %.4f s"
        % (epoch, max_epoch, train_loss, time.time() - epoch_start_time)
    )

    if abs(train_loss) > tmp_loss:
        tmp_loss = abs(train_loss)
        best_epoch = epoch
        best_SSL_model_wts = copy.deepcopy(SSL_model.state_dict())

print('\nBest epoch is :{:d}'.format(best_epoch))
print('Best SSL model loss is :{:.4f}'.format(tmp_loss))

# save model
SAVE_SSL_MODEL_PATH = os.path.join(SAVE_PATH, TIME + 'SSL_model.pt')
torch.save(best_SSL_model_wts, SAVE_SSL_MODEL_PATH)

# -----------------------------------------------------------------------------------
# Step 4: Generate change detection training and test dataset
CD_train_dataset = CDDataset(im1, im2, pre_map, patch_size, isTrain)
CD_train_dataloader = DataLoader(CD_train_dataset, batch_size=batch_size)

CD_test_dataset = CDDataset(im1, im2, pre_map, patch_size, isTest)
CD_test_dataloader = DataLoader(CD_test_dataset, batch_size=batch_size)

# -------------------------------------------------------------------------------------
# Step 5: Setup optimizer and loss function
CD_model = CDNet().to(device)

encoder_state_dict = {k.replace('encoder.', ''): v for k, v in best_SSL_model_wts.items() if 'encoder' in k}
CD_model.encoder.load_state_dict(encoder_state_dict)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CD_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=5e-4)

# ----------------------------------------------------------------------------------------
# Step 6: Supervised fine-tuning
print('--------------------Supervised Fine-tuning-----------------------')

KC = 0.0

for epoch in range(1, max_epoch + 1):
    CD_model.train()
    train_loss = 0
    train_real_label = []
    train_pred_label = []
    epoch_start_time = time.time()

    for i, data in enumerate(CD_train_dataloader):
        x1, x2, y = CD_model.set_input(data)
        y = y.long()
        optimizer.zero_grad()

        outputs = CD_model.forward(x1, x2)
        loss = criterion(outputs, y)
        train_pred = torch.argmax(outputs, dim=1)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_real_label.append(y)
        train_pred_label.append(train_pred)

    train_pred_label = torch.cat(train_pred_label).cpu().numpy()
    train_real_label = torch.cat(train_real_label).cpu().numpy()

    tmp_loss = train_loss/len(CD_train_dataloader)

    print(
        "End of epoch %d / %d \t Training loss is: %.4f \t Time Taken: %.4f s"
        % (epoch, max_epoch, tmp_loss, time.time() - epoch_start_time)
    )

    tmp_KC = metrics.cohen_kappa_score(train_real_label, train_pred_label)
    print("train tmp_KC = : {:f}\n".format(tmp_KC))

    if tmp_KC > KC:
        KC = tmp_KC
        optimal_epoch = epoch
        best_CD_model_wts = copy.deepcopy(CD_model.state_dict())

print('\nBest epoch is :{:d}'.format(optimal_epoch))
print('Best CD model KC is :{:.4f}'.format(KC))
#
# save CD_model
SAVE_CD_MODEL_PATH = os.path.join(SAVE_PATH, TIME + 'CD_model.pt')
torch.save(best_CD_model_wts, SAVE_CD_MODEL_PATH)

# ---------------------------------------------------------------------------
# Step 7: Change detection
print('--------------------Change Detection-----------------------')

CD_model.load_state_dict(torch.load(SAVE_CD_MODEL_PATH))

with torch.no_grad():
    CD_model.eval()
    pred_label = []

    for i, data in enumerate(CD_test_dataloader):
        x1, x2, y = CD_model.set_input(data)
        outputs = CD_model(x1, x2)
        test_pred = torch.argmax(outputs, dim=1)

        pred_label.append(test_pred)

    pred_label = torch.cat(pred_label).cpu().numpy()

# --------------------------------------------------------------------------------
# Step 8: Performance evaluation
map_label = np.zeros((imgH * imgW))
pre_map_label = pre_map.flatten()

map_label[pre_map_label == 0] = 0
map_label[pre_map_label == 255] = 1
map_label[pre_map_label == 128] = pred_label.flatten()

map = (map_label.reshape(imgH, imgW)).astype(np.uint8)
map = postprocess(map)

im_gt = normalization(ref)
message = calculate_metric(im_gt, map, "CM")

# Total time
end_time = time.time()
total_time = end_time - start_time
print('\n Total cost time is : {:.6f}s'.format(total_time))

# Show results
plt.figure()

plt.subplot(131)
plt.title("ND_DI")
plt.imshow(ND_DI, cmap="Greys_r")
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.title("CM")
plt.imshow(map, cmap="Greys_r")
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.title("reference")
plt.imshow(ref, cmap="Greys_r")
plt.xticks([]), plt.yticks([])

plt.show()

# ---------------------------------------------------------------------------------
# Step 9: Save change detection results
with open(Path(SAVE_PATH, "evaluate.txt"), "a") as f:
    f.write(message + "\n")

    f.close()

Map = (map * 255).astype(np.uint8)

io.imsave(SAVE_PATH + '/CM.bmp', Map)
