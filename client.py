from confluent_kafka import Consumer, KafkaError, KafkaException
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from flwr.client import NumPyClient
import torch.nn.functional as F
from typing import List
import torch.nn as nn
import numpy as np
import threading
import logging
import socket
import torch
import flwr
import time
import json
import os

CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Starting Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Kafka Configuration
KAFKA_SERVER = os.environ.get("KAFKA_SERVER", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "sensor_data_drone1")
BATCH_SIZE = 256
NORMALIZE_FEATURES = True

GLOBAL_MEAN = np.array([27.041304, 61.775265, 969.56335,  3.2226298], dtype=np.float32)
GLOBAL_STD  = np.array([8.085047,  24.433617, 105.70269,  2.5380642], dtype=np.float32)

# Consuming Messages
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
class KafkaConsumerApp:
    def __init__(self, kafka_topic, kafka_server):
        CLIENT_GROUP = os.environ.get("KAFKA_GROUP", "sensor_data_group")

        self.kafka_consumer = Consumer({
            'bootstrap.servers': kafka_server,
            'group.id': CLIENT_GROUP,
            'auto.offset.reset': 'earliest'
        })
        self.kafka_consumer.subscribe([kafka_topic])
        self.data_buffer = []


    def consume_data(self):
        logging.info("Starting to consume messages...")
        message_count = 0
        max_messages = 20000
        try:
            while message_count < max_messages:
                msg = self.kafka_consumer.poll(timeout=1.0)  # Poll for messages
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                    
                collected_data = msg.value().decode('utf-8')
                data_dict = json.loads(collected_data)
                self.data_buffer.append(data_dict)  # Store messages in the buffer
                # logging.info(f"✅ Collecting Message: {collected_data}")
                message_count += 1
                # logging.info(f"Total messages received by Client {CLIENT_ID}: {message_count}")
                if message_count % 5000 == 0:
                    logging.info(f"Client {CLIENT_ID} received {message_count} messages...")

        except KeyboardInterrupt:
            logging.info("Stopping Consumer...")
        finally:
            self.kafka_consumer.close()


def load_datasets_from_kafka(data_buffer, batch_size=BATCH_SIZE):
# Extracting the 4 features from each message
# Convert Kafka bugger into PyTorch DataLoaders for training and validation

    features = []
    labels = []

    for item in data_buffer:
        feat = [item["temperature"],    # item["timestamp"]
                item["humidity"],
                item["pressure"],
                item["vibration"],
        ]   # Numeric features
        features.append(feat)
        labels.append(item["label"])                    # Supervised label

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)           # Ensure labels are integers
    # print(f"RAW MEAN: {features.mean(axis=0)}")
    # print(f"RAW STD:  {features.std(axis=0)}")


    if NORMALIZE_FEATURES:
        print(f"[Client {CLIENT_ID}] Normalization ON (GLOBAL stats)")
        features = (features - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)   # We are using standard normalization for same scale features

        print(f"[Client {CLIENT_ID}] Feature mean after: {features.mean(axis=0)}")
        print(f"[Client {CLIENT_ID}] Feature std  after: {features.std(axis=0)}")
    else:
        print(f"[Client {CLIENT_ID}] Normalization OFF")


    x = torch.tensor(features, dtype=torch.float32)     # My input tensor
    y = torch.tensor(labels, dtype=torch.long)          # My label tensor
    dataset = TensorDataset(x, y)                        # TensorDataset lets PyTorch treat my data as a dataset
    
    split_idx = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split_idx, len(dataset) - split_idx]
    )
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"[Client] Number of training samples: {len(train_dataset)}")  # Debug: Training samples
    print(f"[Client] Number of validation samples: {len(val_dataset)}")  # Debug: Validation samples
    return trainloader, valloader

# Net Function
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

class Net(nn.Module):   # Fivr layer fully connected neural
    def __init__(self, input_dim=4, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Increased number of neurons
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Parameters Function
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_parameters(net) -> List[np.ndarray]:    # I need this as a bridge between pytorch and flwr
    """ Converts the models parameters from PyTorch tensors → NumPy arrays. """
    params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    print(f"[Client] Sending parameters to server: {[p.shape for p in params]}")  # Debug: Parameter shapes
    return params

def set_parameters(net, parameters: List[np.ndarray]):
    """ Converts the received NumPy arrays from the server → PyTorch tensors. This lets your local model update its weights before continuing training. """
    print(f"[Client] Received parameters from server: {[p.shape for p in parameters]}")  # Debug: Parameter shapes

    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Training and Testing function
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def train_local(net, trainloader, valloader, epochs, lr, weight_decay, patience, device):
    """Train the network on the training set with early stopping."""
    net.to(device)
    class_weights = torch.tensor([1.0, 9.0]).to(device) # Because of the weighted data (10% intrusions) else it predicts olways normal
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    no_improve = 0
    avg_loss = 0.0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(trainloader)

        # Validation for early stopping
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in valloader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_loss += criterion(net(x_val), y_val).item()
        val_loss /= max(1, len(valloader))
        net.train()
        print(f"[Client] Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            print(f"[Client] Validation loss improved to {best_loss:.4f}")
        else:
            no_improve += 1
            print(f"[Client] No improvement for {no_improve} epoch(s).")
        if no_improve >= patience:
            print(f"[Client] Early stopping at epoch {epoch + 1}.")
            break

    return avg_loss

def test(net, testloader, device):
    """Evaluate the network on the validation set."""
    net.to(device)
    net.eval()
    class_weights = torch.tensor([1.0, 9.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            total_loss += criterion(outputs, y).item()
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    print(f"[Client] Evaluation completed. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy

# Flwr Client
# Client App
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print(f"[Client {self.partition_id}] FlowerClient initialized with partition_id: {self.partition_id}")  # Debug: Initialization
        print(f"[Client {self.partition_id}] Model structure: {self.net}")  # Debug: Model structure
        print(f"[Client {self.partition_id}] Training loader size: {len(self.trainloader.dataset)}")  # Debug: Training dataset size
        print(f"[Client {self.partition_id}] Validation loader size: {len(self.valloader.dataset)}")  # Debug: Validation dataset size
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_parameters(self):
        print(f"[Client {self.partition_id}] get_parameters called")  # Debug: Method call
        params = get_parameters(self.net)
        print(f"[Client {self.partition_id}] Parameters sent to server: {[p.shape for p in params]}")  # Debug: Parameter shapes
        return params
    
    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit() called with config: {config}")
        set_parameters(self.net, parameters)

        avg_loss = train_local(
            net=self.net,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=config.get("local_epochs", 10),
            lr=config.get("lr", 1e-4),
            weight_decay=1e-4,
            patience=config.get("patience", 3),
            device=DEVICE
        )

        updated_params = get_parameters(self.net)
        return updated_params, len(self.trainloader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate() called")
        set_parameters(self.net, parameters)

        loss, accuracy = test(self.net, self.valloader, DEVICE)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
    

# Main Programm
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    consumer_app = KafkaConsumerApp(kafka_topic=KAFKA_TOPIC, kafka_server=KAFKA_SERVER)
    threading.Thread(target=consumer_app.consume_data, daemon=True).start()

    # Wait for Kafka messages to arrive
    print("⏳ Waiting for Kafka data...")
    max_wait = 1200  # seconds
    start_time = time.time()

    TARGET = 20000   # It would be better if i mention the changes if we wait for all the messages
    while len(consumer_app.data_buffer) < TARGET:   # wait for at lease 20000 messages
        if time.time() - start_time > max_wait:
            print(f"⚠️ Timeout: Only {len(consumer_app.data_buffer)} messages collected. \nProceeding anyway...")
            break
        print(f"Waiting... collected {len(consumer_app.data_buffer)} messages")
        time.sleep(1)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f"✅ Collected {len(consumer_app.data_buffer)} messages from Kafka, starting training...")  # Debug: Number of messages collected
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    trainloader, valloader = load_datasets_from_kafka(consumer_app.data_buffer)
    net = Net().to(DEVICE)
    
    # Debug: Verify the model and data loaders
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f"[Client] Model initialized: {net}")  # Debug: Model structure
    print(f"[Client] Training loader size: {len(trainloader.dataset)}")  # Debug: Training dataset size
    print(f"[Client] Validation loader size: {len(valloader.dataset)}")  # Debug: Validation dataset size
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    client_instance = FlowerClient(
        partition_id=CLIENT_ID,
        net=net,
        trainloader=trainloader,
        valloader=valloader
        )

    # Debug: Verify the client instance
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f"[Client] FlowerClient initialized with partition_id: {client_instance.partition_id}")  # Debug: Client initialization
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # Start the Flower client
    FLOWER_SERVER = os.environ.get("FLOWER_SERVER", "server:5000")
    MAX_RETRIES = 20
    RETRY_INTERVAL = 5
    
    def wait_for_server(host, port, retries=MAX_RETRIES, interval=RETRY_INTERVAL):
        for i in range(retries):
            try:
                with socket.create_connection((host, port), timeout=2):
                    print(f"[Client {CLIENT_ID}] Server is ready!")
                    return True
            except (OSError, ConnectionRefusedError):
                print(f"[Client {CLIENT_ID}] Server not ready, retrying ({i+1}/{retries})...")
                time.sleep(interval)
        print(f"[Client {CLIENT_ID}] Server did not become ready after {retries} attempts.")
        return False
    
    host, port = FLOWER_SERVER.split(":")
    port = int(port)
    if wait_for_server(host, port):
        flwr.client.start_client(server_address=FLOWER_SERVER, client=client_instance.to_client())
    else:
        print(f"[Client {CLIENT_ID}] Exiting, cannot connect to server.")
