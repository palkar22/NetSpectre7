from sys import argv
from os.path import isfile
import numpy as np
import onnxruntime as ort
from pathlib import Path
from scapy.all import sniff, IP, TCP, UDP
import queue
import threading

def load_quantized_model(model_path, ep='ipu'):
    providers = ['VitisAIExecutionProvider']
    cache_dir = Path(__file__).parent.resolve()
    provider_options = [{
        'config_file': 'vaip_config.json',
        'cacheDir': str(cache_dir),
        'cacheKey': 'modelcachekey'
    }]

    session = ort.InferenceSession(model_path, providers=providers,
                                   provider_options=provider_options)
    
    return session

def extract_features(src_ip, src_port, dst_port):
    return np.array([int(x) for x in src_ip.split('.')] + [src_port, dst_port]).reshape(1, -1).astype(np.float32)

def get_priority(prediction):
    class_to_priority = {
        0: "Games",
        1: "Real Time",
        2: "Streaming",
        3: "Normal",
        4: "Web download",
        5: "App download"
    }
    predicted_class = np.argmax(prediction[0])
    return class_to_priority[predicted_class]

def packet_callback(packet, packet_queue):
    if IP in packet:
        src_ip = packet[IP].src
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return
        
        packet_queue.put((src_ip, src_port, dst_port))

def process_packets(session, packet_queue):
    while True:
        src_ip, src_port, dst_port = packet_queue.get()
        features = extract_features(src_ip, src_port, dst_port)
        prediction = session.run(None, {'input': features})
        priority = get_priority(prediction)
        print(f"Packet from {src_ip}:{src_port} to port {dst_port} - Priority: {priority}")

def main(model_path):
    if model_path == None:
        while True:
            model_path = input("Enter the quantized model (.onnx) file name: ")
            if isfile(model_path):
                break
            else:
                print("File not found. Retry")


    session = load_quantized_model(model_path)

    packet_queue = queue.Queue()

    # Start the packet processing thread
    processing_thread = threading.Thread(target=process_packets, args=(session, packet_queue))
    processing_thread.daemon = True
    processing_thread.start()

    # Start packet capture
    print("Starting packet capture...")
    sniff(prn=lambda x: packet_callback(x, packet_queue), store=0)

if __name__ == "__main__":
    file = argv[1] if len(argv) > 1 else None
    main(file)