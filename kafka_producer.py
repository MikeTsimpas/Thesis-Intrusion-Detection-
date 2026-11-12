from confluent_kafka import Producer
import logging
import random
import json
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Kafka Configuration
KAFKA_SERVER = os.environ.get("KAFKA_SERVER", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "sensor_data_drone1")
DRONE_ID = os.environ.get("DRONE_ID", "drone-1")

# Initialize Kafka Producer
producer = Producer({
    'bootstrap.servers': KAFKA_SERVER,
    'acks': 'all',  # Ensure full acknowledgment
    'batch.num.messages': 10000, # -
    'linger.ms': 0.1,
    'compression.codec': 'gzip',
})

def generate_sensor_data(anomaly=False):
    """Generate normal or anomalous sensor data."""
    if anomaly:
        return {
            "timestamp": int(time.time()),  # Add a timestamp
            "temperature": round(random.uniform(40.0, 50.0), 2),  # Extreme high temperature (50.0, 100.0)
            "humidity": round(random.uniform(90.0, 150.0), 2),       # Abnormally low humidity (0.0, 10.0)
            "pressure": round(random.uniform(600.0, 800.0), 2),    # Abnormal pressure drop (500.0, 700.0)
            "vibration": round(random.uniform(7.0, 12.0), 2),      # High vibration/possible mechanical failure (8.0, 15.0)
            "label": 1,                                            # Force an anomaly flag was: "engine_check": True / "label": 1
            # "idle_time": random.randint(0, 600)  
        }
    else:
        return {
            "timestamp": int(time.time()),  # Add a timestamp
            "temperature": round(random.uniform(15.0, 35.0), 2),
            "humidity": round(random.uniform(30.0, 80.0), 2),
            "pressure": round(random.uniform(900.0, 1100.0), 2),
            "vibration": round(random.uniform(0.0, 5.0), 2),
            "label": 0,
            # "idle_time": random.randint(0, 600)
        }

def delivery_report(err, msg):
    """Callback for delivery reports."""
    if err is not None:
        logging.error(f"Message delivery failed: {err}")
#    else:
#        logging.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def produce_messages(num_messages=20000, num_intrusions=2000):   #-0, -
    """Produce sensor data messages with exactly num_intrusions randomly distributed."""
    intrusion_indices = set(random.sample(range(num_messages), num_intrusions))
    logging.info(f"Intrusion indices selected: {sorted(list(intrusion_indices))[:10]}...")

    count = 0
    intrusion_count = 0

    while count < num_messages:
        anomaly = count in intrusion_indices
        data = generate_sensor_data(anomaly=anomaly)
        data["drone_id"] = DRONE_ID

        if anomaly:
            intrusion_count += 1
#            logging.info(f"Producing anomaly at index {count}: {data}")  # Log anomaly data

        try:
            producer.produce(KAFKA_TOPIC, json.dumps(data), callback=delivery_report)
            producer.poll(0)  # Ensure message is processed
        except Exception as kafka_error:
            logging.error(f"Kafka produce error: {kafka_error}")
            break  
        count += 1

        if count % 5000 == 0:
            logging.info(f"Produced {count} messages...")
            producer.flush()  # Flush every 5000 messages

        time.sleep(0.001)  # Avoid overloading Kafka

    producer.flush()
    logging.info(f"Finished producing {num_messages} messages.")
    logging.info(f"Total intrusions produced: {intrusion_count}")


if __name__ == "__main__":
    logging.info(f"Producing messages to topic: {KAFKA_TOPIC} (DRONE_ID={DRONE_ID})")
    try:
        produce_messages(num_messages=20000, num_intrusions=2000) #-0, -
    except Exception as e:
        logging.error(f"Error in producer: {e}")
