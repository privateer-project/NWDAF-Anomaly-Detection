from load_dataset import main_load_dataset
from load_models import main_load_models
from xAI_shap import main_shap
from xAI_lime import main_lime
import threading
import time
from multiprocessing import Process

#main_load_dataset.start_load_dataset("127.0.0.1","5000")
# main_load_models.start_load_model("127.0.0.1","5001")
# main_shap.start_xAI_shap("127.0.0.1","5002")
# main_lime.start_xAI_lime("127.0.0.1","5003")


if __name__ == '__main__':
    processes = [
        Process(target=main_load_dataset.start_load_dataset("127.0.0.1", 5000)),
        Process(target=main_load_models.start_load_model, args=("127.0.0.1", 5001)),
        Process(target=main_shap.start_xAI_shap, args=("127.0.0.1", 5002)),
        Process(target=main_lime.start_xAI_lime, args=("127.0.0.1", 5003)),
    ]

    # Start all processes concurrently
    for p in processes:
        p.start()

    print("servers running concurrently on ports 5000 to 5003...")

    # Keep the main process alive
    try:
        while True:
            time.sleep(1)  # Prevent CPU overuse
    except KeyboardInterrupt:
        print("Shutting down Flask servers...")
        for p in processes:
            p.terminate()
            p.join()



