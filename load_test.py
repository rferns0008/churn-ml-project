import requests
import concurrent.futures
import threading

url = "http://localhost:8000/predict"
payload = {
    "tenure": 12, 
    "monthly_charges": 50.0, 
    "total_charges": 600.0, 
    "contract_type": 1, 
    "internet_service": 1
}

# Create a flag that all threads can check
stop_flag = threading.Event()

def spam_requests():
    # Keep looping only while the stop flag is NOT set
    while not stop_flag.is_set():
        try:
            requests.post(url, json=payload, timeout=2)
        except requests.exceptions.RequestException:
            pass

print("Starting load test... Press Ctrl+C to stop.")
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for _ in range(20):
            executor.submit(spam_requests)
        
        # Keep the main thread alive to listen for Ctrl+C
        while True:
            stop_flag.wait(1) 
            
except KeyboardInterrupt:
    print("\nCaught Ctrl+C! Shutting down threads...")
    stop_flag.set() # This tells all 20 threads to exit their while loops