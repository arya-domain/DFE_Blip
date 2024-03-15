from Models.DFE import DFE
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from Utils.Dataset import DFE_dataset, BLIP_Dataset, ImageCaptioningDataset
from Models.BLIP import BLIP
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils.Scores import DFE_score, blip_score
from torch.utils.data import random_split
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


gpu_dfe = 0
gpu_blip = 1
version = 1

name = "Trento"
# name = "Houston"
# name = "MUUFL"
# name = "Augsburg"

print(f"Multi Model Training Version : {name}_{version}")

#########################################Starting DFE Training###########################################################
path_data = f"dataset/{name}-data.mat"
path_label = f"dataset/{name}-label.mat"
model_DFE_save = f"weights/DFE/autoencoder_{name}_{version}.h5"
os.makedirs("weights/DFE/", exist_ok=True)
os.makedirs("weights/blip/model/", exist_ok=True)
os.makedirs("weights/blip/processor/", exist_ok=True)

X, y = DFE_dataset(path_data, path_label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
with tf.device(f"/device:GPU:{gpu_dfe}"):
    model_DFE = DFE(name)
    checkpoint = ModelCheckpoint(
        model_DFE_save, monitor="loss", save_best_only=True, mode="min", verbose=1
    )
    model_DFE.fit(
        X_train,
        [X_train, y_train],
        epochs=250,
        batch_size=32,
        validation_data=(X_test, [X_test, y_test]),
        callbacks=[checkpoint],
    )
    # model_DFE.save(model_DFE_save)
    model_DFE = load_model(model_DFE_save)
    predicted_images, labels = model_DFE.predict(X)

    DFE_validation_score = DFE_score(y, labels)

#########################################Starting BLIP Training###########################################################

model_save = (
    f"weights/blip/model/best_model_large_{name}_{version}.pt"  # model saving name
)
processor_save = f"weights/blip/processor/best_processor_large_{name}_{version}.pt"  # processor saving name


print("Preparing Blip's Dataset -")
dataset = BLIP_Dataset(predicted_images, path_label, name)

length_train = int(len(dataset) * 0.1)  # splitting 1:9
length_val = len(dataset) - length_train
train_dataset, val_dataset = random_split(dataset, [length_train, length_val])


print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")


device = f"cuda:{gpu_blip}" if torch.cuda.is_available() else "cpu"
processor, model = BLIP()
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
print("Model Loaded")

# Trainer
epochs = 200
batch_size = 2
patience = 50
epoch_losses = []
best_loss = float("inf")
count_since_last_improvement = 0

print("DataLoader Loading...")
train_dataloader = ImageCaptioningDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataloader, shuffle=True, batch_size=batch_size)
print("DataLoader Ready")

for epoch in range(epochs):
    print("Epoch:", epoch)
    batch_losses = []
    for idx, batch in tqdm(
        enumerate(train_dataloader),
        desc="Batches :",
        leave=False,
        total=len(train_dataloader),
    ):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
        )
        loss = outputs.loss
        batch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate and append the average loss for this epoch
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(epoch_loss)
    print("Average Loss for Epoch {}: {:.4f}".format(epoch, epoch_loss))

    # Check if the loss has improved
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        count_since_last_improvement = 0

        # Save the best model
        torch.save(model, model_save)
        torch.save(processor, processor_save)
    else:
        count_since_last_improvement += 1

    # Check if the patience limit is reached
    if count_since_last_improvement >= patience:
        print(f"Loss has not improved for {patience} epochs. Stopping training.")
        break


blip_validation_score = blip_score(model_save, processor_save, device, val_dataset, name)
