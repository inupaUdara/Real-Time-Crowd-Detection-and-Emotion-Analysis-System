# analytics.py
from collections import defaultdict, deque
from datetime import datetime
import json
import os
import threading

class AnalyticsManager:
    def __init__(self, save_dir="analytics", save_analytics=True):
        self.save_analytics = save_analytics
        self.data = defaultdict(lambda: deque(maxlen=1000))  # object_id -> list of timestamps
        self.zone_entry_times = defaultdict(dict)  # object_id -> zone_id -> entry time
        self.logs = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.lock = threading.Lock()

    def update_tracking_data(self, object_id, timestamp):
        with self.lock:
            self.data[object_id].append(timestamp)

    def record_zone_entry(self, object_id, zone_id, timestamp):
        with self.lock:
            if zone_id not in self.zone_entry_times[object_id]:
                self.zone_entry_times[object_id][zone_id] = timestamp

    def record_zone_exit(self, object_id, zone_id, timestamp):
        with self.lock:
            if zone_id in self.zone_entry_times[object_id]:
                entry_time = self.zone_entry_times[object_id].pop(zone_id)
                duration = (timestamp - entry_time).total_seconds()
                self.logs.append({
                    "object_id": object_id,
                    "zone_id": zone_id,
                    "entry_time": entry_time.isoformat(),
                    "exit_time": timestamp.isoformat(),
                    "duration": duration
                })

    def save(self):
        if self.save_analytics:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"analytics_{now}.json")
            with open(filename, "w") as f:
                json.dump(self.logs, f, indent=2)
            print(f"✅ Saved analytics log to {filename}")
        else:
            print("⚠️ Analytics saving is disabled.")
