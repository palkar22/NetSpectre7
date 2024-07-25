import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path

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
    # Convert IP and ports to features
    # This is a placeholder - implement based on your model's requirements
    return np.array([int(x) for x in src_ip.split('.')] + [src_port, dst_port]).reshape(1, -1).astype(np.float32)

def get_priority(prediction):
    class_to_priority = {
        0: 0,  # Games
        1: 1,  # Real Time
        2: 2,  # Streaming
        3: 3,  # Normal
        4: 4,  # Web download
        5: 5   # App download
    }
    predicted_class = np.argmax(prediction[0])
    return class_to_priority[predicted_class]

def main(src_ip, src_port, dst_port):
    model_path = "packet_prioritizer_quantized_npu.onnx"
    session = load_quantized_model(model_path)

    features = extract_features(src_ip, src_port, dst_port)
    prediction = session.run(None, {'input': features})
    priority = get_priority(prediction)

    print(priority, end='')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python npu_model.py <src_ip> <src_port> <dst_port>")
        sys.exit(1)
    
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))