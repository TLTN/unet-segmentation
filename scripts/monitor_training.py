import psutil
import time
import matplotlib.pyplot as plt
from collections import deque
import threading
import argparse

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    print("GPU monitoring not available (pynvml not installed)")


class TrainingMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.cpu_data = deque(maxlen=max_points)
        self.memory_data = deque(maxlen=max_points)
        self.gpu_data = deque(maxlen=max_points)
        self.gpu_memory_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)

        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_available = False

        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Started performance monitoring...")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Stopped performance monitoring.")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()

                # GPU
                gpu_percent = 0
                gpu_memory_percent = 0

                if self.gpu_available:
                    try:
                        gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        gpu_percent = gpu_info.gpu

                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_memory_percent = (memory_info.used / memory_info.total) * 100
                    except:
                        pass

                # Store data
                current_time = time.time()
                self.timestamps.append(current_time)
                self.cpu_data.append(cpu_percent)
                self.memory_data.append(memory.percent)
                self.gpu_data.append(gpu_percent)
                self.gpu_memory_data.append(gpu_memory_percent)

            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(1)  # Monitor every second

    def get_current_stats(self):
        """Get current system statistics"""
        if len(self.timestamps) == 0:
            return None

        return {
            'cpu': self.cpu_data[-1] if self.cpu_data else 0,
            'memory': self.memory_data[-1] if self.memory_data else 0,
            'gpu': self.gpu_data[-1] if self.gpu_data else 0,
            'gpu_memory': self.gpu_memory_data[-1] if self.gpu_memory_data else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Monitor training performance')
    parser.add_argument('--duration', type=int, default=60,
                        help='Monitoring duration in seconds')
    args = parser.parse_args()

    monitor = TrainingMonitor()
    monitor.start_monitoring()

    try:
        print(f"Monitoring for {args.duration} seconds...")
        for i in range(args.duration):
            time.sleep(1)
            stats = monitor.get_current_stats()
            if stats:
                print(f"\rCPU: {stats['cpu']:.1f}% | "
                      f"RAM: {stats['memory']:.1f}% | "
                      f"GPU: {stats['gpu']:.1f}% | "
                      f"GPU RAM: {stats['gpu_memory']:.1f}%", end='')

    except KeyboardInterrupt:
        print("\nMonitoring interrupted")

    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
