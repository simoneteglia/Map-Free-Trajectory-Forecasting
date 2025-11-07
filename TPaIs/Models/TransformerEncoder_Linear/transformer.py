import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os 


TARGET_FRAMES = 9
pl.seed_everything(9999)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2105):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class VehiclePredictorTransformer(pl.LightningModule):
    def __init__(self, feature_dim=8,embedding_dim=32, max_num_vehicles=0, history_len=0, future_len=0, hidden_dim=64, num_layers_encoder=4, lr=0.001, predicted_features=2):
        super(VehiclePredictorTransformer, self).__init__()

        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.positional_encoding = PositionalEncoding(d_model=embedding_dim, max_len=1000)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_dim, 
                                                                    dropout=0.1, activation='gelu', batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers_encoder, norm=self.layer_norm)

        self.criterion = nn.MSELoss()

        self.lr = lr

        self.history_len = history_len
        self.future_len = future_len
        self.predicted_features = predicted_features

        self.mlp = nn.Sequential(
            nn.Linear(history_len*max_num_vehicles*embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, future_len*max_num_vehicles*predicted_features)  # Predicting for 24 future frames, 1 vehicle, 13 possible vehicles, each with (x, y)
        )

    
    def forward(self, x, active_vehicle_mask):
        # x shape: (num_frames, num_vehicles, feature_dim)
          
        B, F, V, f = x.size()  # B: batch size, F: num_frames, V: num_vehicles, f: feature_dim

        x = x.view(B, F * V, f)
        active_vehicle_mask = ~active_vehicle_mask.view(B, F * V)
        
        # print("Original X Shape:", x.shape)
        x = self.embedding(x)
        # print("Embedding Shape:", x.shape)

        x = self.positional_encoding(x)  # Add positional encoding
        
        x = self.transformer_encoder(x, src_key_padding_mask=active_vehicle_mask)

        x = x.reshape((B, F*V*32))
        future_positions = self.mlp(x)
        future_positions = future_positions.reshape((B, self.future_len, V, self.predicted_features))
        # print("Prediction Shape:", future_positions.shape)

        return future_positions
    
    def training_step(self, batch):
        inputs, targets, inputs_mask, targets_mask = batch
        outputs = self(inputs, inputs_mask)

        valid_outputs = outputs[targets_mask]
        valid_targets = targets[targets_mask]

        loss = self.criterion(valid_outputs, valid_targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        inputs, targets, inputs_mask, targets_mask = batch
        outputs = self(inputs, inputs_mask)

        valid_outputs = outputs[targets_mask]
        valid_targets = targets[targets_mask]

        minAde = self.minADE_FDE(valid_outputs.detach().cpu().numpy(), valid_targets.detach().cpu().numpy())

        loss = self.criterion(valid_outputs, valid_targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_minADE", minAde, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx): 
        inputs, targets, inputs_mask, targets_mask = batch
        outputs = self(inputs, inputs_mask)

        valid_outputs = outputs[targets_mask]
        valid_targets = targets[targets_mask]

        minAde = self.minADE_FDE(valid_outputs.detach().cpu().numpy(), valid_targets.detach().cpu().numpy())

        if batch_idx == 0: 
            with open("test_predictions.txt", "w") as f:
                f.write("Predictions VS Ground Truth:\n")
                for i in range( min(10, valid_outputs.size(0) )  ):
                    f.write(f"Predictions: {np.round(valid_outputs[i].detach().cpu().numpy(), 3)} \n Grund Truth: {np.round(valid_targets[i].detach().cpu().numpy(),3)}\n")
        
        loss = self.criterion(valid_outputs, valid_targets)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_minADE", minAde, on_step=True, on_epoch=True, prog_bar=True)        
        return loss

    def minADE_FDE(self, preds, gt):
        """
        preds: (K, T, 2)
        gt: (T, 2)
        """
        # Compute L2 distances at each timestep
        l2_dist = np.linalg.norm(preds - gt[:, :2], axis=1)  # shape: (K, T)

        # ADE for each prediction
        ade_all = np.mean(l2_dist)  # (K,)

        return ade_all

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class TrajectoriesDatasetGenerator(Dataset):
    def __init__(self, dataframe, history_len, future_len, tot_analyzed_frames, predicted_features):
        
        self.df = dataframe
        self.df_frame_grouped = self.df.groupby("frame_id")
        
        self.history_len = history_len
        self.future_len = future_len
        self.tot_analyzed_frames = tot_analyzed_frames
        self.predicted_features = predicted_features
        
        self.dataset_len = self.tot_analyzed_frames - self.history_len - self.future_len

        self.feature_cache = {}
        self.dataset = []

        self.masks = {}
        self.preprocess_features()

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        """
        Get a batch of data for a given index (frame).
        
        :param idx: The index (frame) for which to retrieve the data.
        :return: A dictionary with 'input' (feature tensor) and 'target' (future position tensor).
        """
        print("Index di chiamata: ", index)
        features = self.feature_cache[index]
        features = torch.tensor(features, dtype=torch.float)  # (num_window_frames, vehicle_features, num_vehicles)
        features = features.permute(0, 2, 1) # (num_window_frames, num_vehicles, vehicle_features)
        
        inputs = features[:self.history_len,:,:] # (history_len, num_vehicles, vehicle_features)
        inputs_mask = self.masks[index][:self.history_len, :] # (history_len, num_vehicles)

        targets = features[self.history_len:,:,:][:,:,:self.predicted_features] # (future_len, num_vehicles, vehicle_features)        
        targets_mask = self.masks[index][self.history_len:, :] # (future_len, num_vehicles)

        return inputs, targets, inputs_mask, targets_mask

    def preprocess_features(self):

        for i in tqdm(range(0, self.dataset_len), desc="Creating vehicles features per frame"):
            self.feature_cache[i] = self.create_temp_window_features(i)

        max_vehicles_per_frame = 0
        for index in self.feature_cache:
            feat = self.feature_cache[index]
            if feat.shape[2] > max_vehicles_per_frame:
                max_vehicles_per_frame = feat.shape[2]

        self.max_vehicles_per_frame = max_vehicles_per_frame
        self.pad_features(max_vehicles_per_frame)

        return
    
    def create_temp_window_features(self, i):

        track_ids = set()
        for k in range(i, i + self.history_len + self.future_len + 1):
            frame_df = self.df_frame_grouped.get_group(k)
            track_ids.update(frame_df['track_id'].values)
 
        #sort track ids to maintain consistent ordering
        sorted_track_ids = sorted(track_ids)
        track2index = {track_id: idx for idx, track_id in enumerate(sorted_track_ids)}

        features = np.zeros((self.history_len+self.future_len, 8, len(sorted_track_ids)), dtype=np.float32)

        ## (i+1) perché così partiamo dal frame 1 e possiamo calcolare i displacements come differenza tra frame i e i-1
        ## avere una finestra di self.history_len = 9 è buono dato che i dati sono a 6fps e quindi consideriamo 1.5s di storia
        for t, j in enumerate(range(i+1, i+self.history_len+self.future_len+1)):
            frame_df = self.df_frame_grouped.get_group(j)

            for _, row in frame_df.iterrows():
                idx = track2index[row['track_id']]

                # Skip if the vehicle was not present in the previous frame
                if self.df_frame_grouped.get_group(j-1)[self.df_frame_grouped.get_group(j-1)['track_id'] == row['track_id']].empty:
                    continue

                features[t, 0, idx] = row['x'] - self.df_frame_grouped.get_group(j-1)[self.df_frame_grouped.get_group(j-1)['track_id'] == row['track_id']]['x'].item() 
                features[t, 1, idx] = row['y'] - self.df_frame_grouped.get_group(j-1)[self.df_frame_grouped.get_group(j-1)['track_id'] == row['track_id']]['y'].item() 
                features[t, 2, idx] = row['vx']
                features[t, 3, idx] = row['vy']
                features[t, 4, idx] = row['ax']
                features[t, 5, idx] = row['ay']
                features[t, 6, idx] = row['width']
                features[t, 7, idx] = row['length']
        
        return features
    
    def pad_features(self, max_vehicles_per_frame):

        for i in self.feature_cache:
            feat = self.feature_cache[i]
            assert feat.ndim == 3
            assert feat.shape[1] == 8

            batch_size = feat.shape[0]
            original_num_vehicles = feat.shape[2]

            mask = np.zeros((batch_size, max_vehicles_per_frame), dtype=bool)
            mask[:, :original_num_vehicles] = True
            self.masks[i] = mask

            if feat.shape[2] != max_vehicles_per_frame:
                feat_padded = np.pad(feat, pad_width=((0,0), (0, 0), (0, max_vehicles_per_frame-feat.shape[2])), 
                              mode="constant", constant_values=0.0)
                
                self.feature_cache[i] = feat_padded
                        
        return 

class TrajectoryDatasetArray(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, xm, ym = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(xm, dtype=torch.bool), torch.tensor(ym, dtype=torch.bool)

class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TrajectoryDatasetArray(self.train_data)
        self.val_dataset = TrajectoryDatasetArray(self.val_data)
        self.test_dataset = TrajectoryDatasetArray(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=100, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=100, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=100, persistent_workers=True, pin_memory=True)
    
def main(mode, root_path, history_len, future_len, tot_analyzed_frames, predicted_features):
    
    # CREA DATASET
    if (mode == 'create_data'):
    
        df = pd.read_csv(root_path+"Dataset/Original/sind_smoothed_tracks.csv")
        
        dataset = TrajectoriesDatasetGenerator(df, history_len, future_len, tot_analyzed_frames, predicted_features)

        inputs = []
        targets = []
        inputs_masks = []
        targets_masks = []

        for i in range(0,len(dataset)):
            data = dataset[i]
            inputs.append(data[0])
            targets.append(data[1])
            inputs_masks.append(data[2])
            targets_masks.append(data[3])

        inputs = np.array(inputs) 
        targets = np.array(targets)
        inputs_masks = np.array(inputs_masks)
        targets_masks = np.array(targets_masks)
        max_num_vehicles = np.array([dataset.max_vehicles_per_frame])
        
        np.save(root_path+"Dataset/Processed/Traj_Inputs",inputs)
        np.save(root_path+"Dataset/Processed/Traj_Targets",targets)
        np.save(root_path+"Dataset/Processed/Traj_Inputs_Masks",inputs_masks)
        np.save(root_path+"Dataset/Processed/Traj_targets_Masks",targets_masks)
        np.save(root_path+"Dataset/Processed/Max_Num_Vehicles",max_num_vehicles)

    # AVVIA TRAINING
    elif (mode == 'load_data'):
       
        inputs = np.load(root_path+"Dataset/Processed/Traj_Inputs.npy", allow_pickle=True)
        targets = np.load(root_path+"Dataset/Processed/Traj_Targets.npy", allow_pickle=True)
        inputs_masks = np.load(root_path+"Dataset/Processed/Traj_Inputs_Masks.npy", allow_pickle=True)
        targets_masks = np.load(root_path+"Dataset/Processed/Traj_targets_Masks.npy", allow_pickle=True)
        max_num_vehicles = np.load(root_path+"Dataset/Processed/Max_Num_Vehicles.npy", allow_pickle=True)

        model = VehiclePredictorTransformer(feature_dim=8, max_num_vehicles=max_num_vehicles[0], 
                                            history_len=history_len, future_len=future_len, hidden_dim=64, num_layers_encoder=4, predicted_features=predicted_features)

        X_train, X_temp, y_train, y_temp, xm_train, xm_temp, ym_train, ym_temp = train_test_split(inputs, targets, inputs_masks, targets_masks, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test, xm_val, xm_test, ym_val, ym_test = train_test_split(X_temp, y_temp, xm_temp, ym_temp, test_size=0.3, random_state=42)

        train_data = list(zip(X_train, y_train, xm_train, ym_train))
        val_data   = list(zip(X_val, y_val, xm_val, ym_val))
        test_data  = list(zip(X_test, y_test, xm_test, ym_test))

        data_module = TrajectoryDataModule(train_data, val_data, test_data, batch_size=32)

        # callbacks, logger e Trainer
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop = EarlyStopping(monitor="val_minADE", patience=20, mode="min")
        ckpt_cb    = ModelCheckpoint(
            dirpath=root_path,
            monitor="val_minADE",
            save_top_k=2,
            mode="min",
            filename=f"prova_{{epoch:04d}}-{{val_loss:.6f}}"
        )
        tb_logger  = TensorBoardLogger("logs/", name=f"prova")

        trainer = pl.Trainer(
            logger=tb_logger,
            max_epochs=300,
            callbacks=[early_stop, ckpt_cb, lr_monitor],
            devices=1,
            accelerator="gpu"
        )

        trainer.fit(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module)

    elif (mode == 'test_data'):

        Checkpointpath = root_path+"Weights/TransformerEncoder_Linear/prova_epoch=0291-val_loss=0.002218.ckpt"

        inputs = np.load(root_path+"Dataset/Processed/Traj_Inputs.npy", allow_pickle=True)
        targets = np.load(root_path+"Dataset/Processed/Traj_Targets.npy", allow_pickle=True)
        inputs_masks = np.load(root_path+"Dataset/Processed/Traj_Inputs_Masks.npy", allow_pickle=True)
        targets_masks = np.load(root_path+"Dataset/Processed/Traj_targets_Masks.npy", allow_pickle=True)
        max_num_vehicles = np.load(root_path+"Dataset/Processed/Max_Num_Vehicles.npy", allow_pickle=True)

        model = VehiclePredictorTransformer.load_from_checkpoint(
                    Checkpointpath,
                    feature_dim=8, max_num_vehicles=max_num_vehicles[0], 
                    history_len=history_len, future_len=future_len, hidden_dim=64, 
                    num_layers_encoder=4, predicted_features=predicted_features,
                    map_location='cpu'
                )

        test_input = torch.tensor(inputs[0]).unsqueeze(0)
        test_mask = torch.tensor(inputs_masks[0]).unsqueeze(0)

        output = model(test_input, test_mask)

        output = output.transpose(1,2).squeeze(0)

        outdir = root_path+'Results/Predictions'
        prefix = 'pred_veicolo_'

        for i, mat in enumerate(output):  # mat: (9, 2)
            arr = mat.detach().cpu().numpy()

            # formato numerico adatto al dtype
            if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
                fmt = "%d"
            else:
                fmt = "%.6g"

            np.savetxt(
                os.path.join(outdir, f"{prefix}_{i:02d}.csv"),
                arr,
                delimiter=",",
                fmt=fmt
            )

        outdir = root_path+'Results/Targets'
        prefix = 'target_veicolo_'

        targets = torch.tensor(targets[0]).transpose(0,1)

        for i, mat in enumerate(targets):  # mat: (9, 2)

            arr = mat.detach().cpu().numpy()

            # formato numerico adatto al dtype
            if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
                fmt = "%d"
            else:
                fmt = "%.6g"

            np.savetxt(
                os.path.join(outdir, f"{prefix}_{i:02d}.csv"),
                arr,
                delimiter=",",
                fmt=fmt
            )
        
        return


if __name__ == "__main__":
    
    # mode = 'create_data'
    # mode = 'load_data'
    mode = 'test_data'
    
    history_len = 9
    future_len = 9
    predicted_features = 2
    tot_analyzed_frames = 12004

    root_path = "/workspace/TPaIs/"

    main(mode, root_path, history_len, future_len, tot_analyzed_frames, predicted_features)
