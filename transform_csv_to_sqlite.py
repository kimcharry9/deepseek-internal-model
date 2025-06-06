import pandas as pd
import sqlite3
import glob
import time
from tqdm import tqdm
from datetime import datetime

# โหลดข้อมูลจาก CSV
#-- Load Datasets to Document Format --#
date_range = [str(i) for i in range(20250212, 20250219)]

file_paths = []
merged_data = []
for date in date_range:
    file_paths.extend(glob.glob(f"/data/deepseek-model/mockdata/{date}/*gsb-lake-prd-monitor02_8086_telegraf_disk.csv.gz"))

print("[1] Pull and Generate Datasets: Processing...")
with tqdm(file_paths, desc="Read CSV File...", unit="files") as raw_file:
    for file in raw_file:
        df = pd.read_csv(file, compression='gzip').dropna()
        df["time"] = pd.to_datetime(df['time'])  # แปลง timestamp

        with tqdm(df.iterrows(), total=len(df), desc=f"- Reading file '{file.split('/')[-1]}'...", unit="rows") as datasets:
            for _, row in datasets:
                doc = {
                    "date": row["time"].date().isoformat(),
                    "time": row["time"].time().isoformat(),
                    "hostname": row["hostname"],
                    "disk_path": row["path"],
                    "disk_used_percent": float(row["disk_used.percent"]),
                    "disk_total": int(row["disk_total.bytes"]),
                    "disk_used_bytes": int(row["disk_used.bytes"])
                }
                merged_data.append(doc)

df = pd.DataFrame(merged_data)
df.sort_values(by=["date", "time"], inplace=True)
print("[1] Pull and Generate Datasets: Completed!")

#df = pd.read_csv("/data/deepseek-model/mockdata/20250218/verify_data.csv")
#df['datetime'] = pd.to_datetime(df['time'])  # แปลงเป็น datetime object
#df['date'] = df['datetime'].dt.date.astype(str)
#df['time'] = df['datetime'].dt.time.astype(str)

# ลบ column เดิมถ้าไม่ใช้
#df.drop(columns=['datetime'], inplace=True)

print("[2] Export Datasets to SQLite Database: Processing...")
conn = sqlite3.connect("/data/sqlite_db/disk_usage.db")
df.to_sql("disk_usage", conn, if_exists="replace", index=False)

# ปิดการเชื่อมต่อ
conn.close()
print("[2] Export Datasets to SQLite Database: Completed!")
