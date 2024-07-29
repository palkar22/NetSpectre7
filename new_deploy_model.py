from sys import argv
from os.path import isfile
import numpy as np
import onnxruntime as ort
from pathlib import Path
from scapy.all import sniff, IP, TCP, UDP
import multiprocessing as mp
from new_prioritizer import priority_setter

def load_quantized_model(model_path, ep='ipu'):
    providers = ['VitisAIExecutionProvider']
    cache_dir = Path(__file__).parent.resolve()
    provider_options = [{
        'config_file': 'vaip_config.json',
        'cacheDir': str(cache_dir),
        'cacheKey': 'modelcachekey'
    }]
    return ort.InferenceSession(model_path, providers=providers, provider_options=provider_options)

def extract_features(src_ip, src_port, dst_port):
    features = np.array([int(x) for x in src_ip.split('.')] + [src_port, dst_port])
    if len(features) < 10:
        features = np.pad(features, (0, 10 - len(features)))
    elif len(features) > 10:
        features = features[:10]
    return features.reshape(1, 10, 1).repeat(10, axis=2).astype(np.float32)

def get_priority(prediction):
    class_to_priority = {0: "Games", 1: "Real Time", 2: "Streaming", 3: "Normal", 4: "Web download", 5: "App download"}
    return class_to_priority[np.argmax(prediction[0])]

def packet_callback(packet, packet_queue):
    if IP in packet:
        src_ip = packet[IP].src
        if TCP in packet:
            src_port, dst_port = packet[TCP].sport, packet[TCP].dport
        elif UDP in packet:
            src_port, dst_port = packet[UDP].sport, packet[UDP].dport
        else:
            return
        packet_queue.put((src_ip, src_port, dst_port))

def process_packets(session, packet_queue, prediction_queue):
    while True:
        src_ip, src_port, dst_port = packet_queue.get()
        features = extract_features(src_ip, src_port, dst_port)
        prediction = session.run(None, {'input': features})
        priority = get_priority(prediction)
        prediction_queue.put({"src_ip": src_ip, "src_port": src_port, "dst_port": dst_port, "priority": priority})

def main(model_path):
    if model_path is None:
        while not isfile(model_path := input("Enter the quantized model (.onnx) file name: ")):
            pass

    session = load_quantized_model(model_path)
    packet_queue = mp.Queue()
    prediction_queue = mp.Queue()

    packet_process = mp.Process(target=process_packets, args=(session, packet_queue, prediction_queue))
    packet_process.start()

    priority_process = mp.Process(target=priority_setter, args=(prediction_queue,))
    priority_process.start()

    sniff(prn=lambda x: packet_callback(x, packet_queue), store=0)

if __name__ == "__main__":
    main(argv[1] if len(argv) > 1 else None)