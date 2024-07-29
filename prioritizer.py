import win32process
import win32con
import win32api
import psutil
import time
import threading
import queue
import json
from scapy.all import sniff, IP, TCP, UDP
import winreg

# Global queue to store model predictions
prediction_queue = queue.Queue()

# Function to continuously read model predictions (simulated here)
def read_model_predictions():
    while True:
        # In a real scenario, you'd read from the NPU output
        # For simulation, we'll generate random predictions
        prediction = {
            "src_ip": "192.168.1.100",
            "src_port": 12345,
            "dst_port": 80,
            "priority": "Games"
        }
        prediction_queue.put(prediction)
        time.sleep(0.1)  # Adjust based on your model's inference speed

# Function to get the PID of the process using a specific port
def get_pid_by_port(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return conn.pid
    return None

# Function to set process priority
def set_process_priority(pid, priority):
    try:
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
        if priority == "Games" or priority == "Real Time":
            win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
        elif priority == "Streaming":
            win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
        elif priority == "Normal":
            win32process.SetPriorityClass(handle, win32process.NORMAL_PRIORITY_CLASS)
        elif priority == "Web download":
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        elif priority == "App download":
            win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
        print(f"Set priority of PID {pid} to {priority}")
    except Exception as e:
        print(f"Error setting priority for PID {pid}: {str(e)}")
    finally:
        if 'handle' in locals():
            win32api.CloseHandle(handle)

# Function to monitor and adjust application priorities
def monitor_and_adjust_priorities():
    while True:
        try:
            prediction = prediction_queue.get(timeout=1)
            pid = get_pid_by_port(prediction['src_port'])
            if pid:
                set_process_priority(pid, prediction['priority'])
        except queue.Empty:
            continue

# Function to get the WiFi interface name
def get_wifi_interface_name():
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\NetworkCards") as key:
            index = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, index)
                    with winreg.OpenKey(key, subkey_name) as subkey:
                        description = winreg.QueryValueEx(subkey, "Description")[0]
                        if "Wi-Fi" in description:
                            service_name = winreg.QueryValueEx(subkey, "ServiceName")[0]
                            return service_name
                    index += 1
                except WindowsError:
                    break
    except WindowsError:
        pass
    return None

# Function to capture packets and correlate with predictions
def capture_packets():
    wifi_interface = get_wifi_interface_name()
    if not wifi_interface:
        print("WiFi interface not found. Please check your network settings.")
        return

    def packet_callback(packet):
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
            
            # Check if this packet matches any recent predictions
            while not prediction_queue.empty():
                prediction = prediction_queue.get()
                if (prediction['src_ip'] == src_ip and 
                    prediction['src_port'] == src_port and 
                    prediction['dst_port'] == dst_port):
                    pid = get_pid_by_port(src_port)
                    if pid:
                        set_process_priority(pid, prediction['priority'])
                    break
                prediction_queue.put(prediction)  # Put it back if it doesn't match

    print(f"Starting packet capture on interface: {wifi_interface}")
    sniff(prn=packet_callback, store=0, iface=wifi_interface)

# Main function
def main():
    # Start the thread to read model predictions
    threading.Thread(target=read_model_predictions, daemon=True).start()

    # Start the thread to monitor and adjust priorities
    threading.Thread(target=monitor_and_adjust_priorities, daemon=True).start()

    # Start packet capture
    capture_packets()

if __name__ == "__main__":
    main()