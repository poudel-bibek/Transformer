import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import torch
import torch.nn as nn
import seaborn as sns
from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# Function to downsample labels
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import torch

def downsample_labels(labels, target_length=300):
    original_length = labels.shape[1]
    x = np.linspace(0, original_length - 1, original_length)
    x_new = np.linspace(0, original_length - 1, target_length)

    downsampled_labels = np.zeros((labels.shape[0], target_length, labels.shape[2]))

    for i in range(labels.shape[0]):
        for j in range(labels.shape[2]):
            f = scipy.interpolate.interp1d(x, labels[i, :, j], kind='linear')
            downsampled_labels[i, :, j] = f(x_new)

    return downsampled_labels

def load_all_data(batch_size, base_path='./multitask/data_processed', exclude_users=['irfan_0220', 'ra']):
    imu_data_list = []
    cop_data_list = []
    pressure_data_list = []
    subject_ids = []

    users = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d not in exclude_users]
    print(f"Found users: {users}")

    # Create a mapping from subject IDs to integers
    subject_id_mapping = {user: i for i, user in enumerate(users)}
    
    # Save the mapping to a file
    with open('subject_id_mapping.txt', 'w') as f:
        for user, user_id in subject_id_mapping.items():
            f.write(f'{user}: {user_id}\n')

    for user in users:
        imu_path = os.path.join(base_path, user, 'imu.npy')
        cop_path = os.path.join(base_path, user, 'cop.npy')
        pressure_path = os.path.join(base_path, user, 'pressure.npy')

        imu = np.load(imu_path)
        cop = np.load(cop_path)
        pressure = np.load(pressure_path)

        # Downsampling the cop and pressure data
        cop_processed = downsample_labels(cop, 300)
        pressure_processed = downsample_labels(pressure, 300)

        # Keeping the IMU data as is
        imu_processed = downsample_labels(imu, 300)

        imu_data_list.append(imu_processed)
        cop_data_list.append(cop_processed)
        pressure_data_list.append(pressure_processed)
        
        # Append the numeric subject ID to the list
        subject_ids.extend([subject_id_mapping[user]] * imu_processed.shape[0])

    # Concatenate all users' data
    imu_data = np.concatenate(imu_data_list, axis=0)
    cop_data = np.concatenate(cop_data_list, axis=0)
    pressure_data = np.concatenate(pressure_data_list, axis=0)
    subject_ids = np.array(subject_ids)

    print(f"Overall IMU data shape: {imu_data.shape}")
    print(f"Overall COP data shape: {cop_data.shape}")
    print(f"Overall Pressure data shape: {pressure_data.shape}")

    # Splitting into training and test sets
    imu_train, imu_test, cop_train, cop_test, pressure_train, pressure_test, subject_train, subject_test = \
        train_test_split(imu_data, cop_data, pressure_data, subject_ids, test_size=0.2, random_state=42)

    # Further splitting the training data into training and validation sets
    imu_train, imu_val, cop_train, cop_val, pressure_train, pressure_val, subject_train, subject_val = \
        train_test_split(imu_train, cop_train, pressure_train, subject_train, test_size=0.2, random_state=42)

    # Creating TensorDatasets
    train_dataset = TensorDataset(torch.tensor(imu_train, dtype=torch.float32),
                                  torch.tensor(cop_train, dtype=torch.float32),
                                  torch.tensor(pressure_train, dtype=torch.float32),
                                  torch.tensor(subject_train, dtype=torch.int64))

    val_dataset = TensorDataset(torch.tensor(imu_val, dtype=torch.float32),
                                torch.tensor(cop_val, dtype=torch.float32),
                                torch.tensor(pressure_val, dtype=torch.float32),
                                torch.tensor(subject_val, dtype=torch.int64))

    test_dataset = TensorDataset(torch.tensor(imu_test, dtype=torch.float32),
                                 torch.tensor(cop_test, dtype=torch.float32),
                                 torch.tensor(pressure_test, dtype=torch.float32),
                                 torch.tensor(subject_test, dtype=torch.int64))

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, subject_id_mapping




train_loader, val_loader, test_loader, subject_id_mapping = load_all_data(batch_size=128)


# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output_cop: int,
                 d_output_pressure: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None):
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self.layers_decoding_cop = nn.ModuleList([Decoder(d_model,
                                                          q,
                                                          v,
                                                          h,
                                                          attention_size=attention_size,
                                                          dropout=dropout,
                                                          chunk_mode=chunk_mode) for _ in range(N)])

        self.layers_decoding_pressure = nn.ModuleList([Decoder(d_model,
                                                               q,
                                                               v,
                                                               h,
                                                               attention_size=attention_size,
                                                               dropout=dropout,
                                                               chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear_cop = nn.Linear(d_model, d_output_cop)
        self._linear_pressure = nn.Linear(d_model, d_output_pressure)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add positional encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack for cop
        decoding_cop = encoding.clone()  # Clone the encoding for separate decoders
        for layer in self.layers_decoding_cop:
            decoding_cop = layer(decoding_cop, encoding)

        # Decoding stack for pressure
        decoding_pressure = encoding.clone()
        for layer in self.layers_decoding_pressure:
            decoding_pressure = layer(decoding_pressure, encoding)

        # Output modules
        output_cop = self._linear_cop(decoding_cop)
        output_pressure = self._linear_pressure(decoding_pressure)

        return output_cop, output_pressure



# Initialize the model
d_model = 64
q = 8
v = 8
h = 8
N = 4
attention_size = 12
dropout = 0.2
chunk_mode = None
d_input = 3
d_output_cop = 4
d_output_pressure = 16
pe_period = None
pe = None
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = Transformer(d_input, d_model, d_output_cop, d_output_pressure, q, v, h, N, attention_size, dropout, chunk_mode, pe, pe_period).to(device)



# Training function
def train_and_evaluate(model, train_loader, val_loader, epochs=500, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[500, 750], gamma=0.1)
    
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')  # initialize with a high value
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for i, (imu, cop, pressure, subject_id) in enumerate(train_loader, 0):  # Notice subject_id is included
            imu, cop, pressure = imu.to(device), cop.to(device), pressure.to(device)

            optimizer.zero_grad()
            output_cop, output_pressure = model(imu)

            loss_cop = criterion(output_cop, cop)
            loss_pressure = criterion(output_pressure, pressure)

            loss_train = loss_cop + loss_pressure
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imu, cop, pressure, subject_id in val_loader:
                imu, cop, pressure = imu.to(device), cop.to(device), pressure.to(device)

                output_cop, output_pressure = model(imu)

                loss_cop = criterion(output_cop, cop)
                loss_pressure = criterion(output_pressure, pressure)

                loss_eval = loss_cop + loss_pressure
                val_loss += loss_eval.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './pth/imu_multitask_subs1.pth')  # Save the best model

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.6f}")
            print(f"Validation Loss: {avg_val_loss:.6f}")

    model.load_state_dict(best_model_weights)
    # Plotting
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

train_and_evaluate(model, train_loader, val_loader, epochs=1000, lr=0.0002)

sys.exit()