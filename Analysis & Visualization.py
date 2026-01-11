import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sensor_data_processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])  

stats = df[["temperature_denoised", "air_humidity_denoised", "soil_moisture_denoised"]].describe()
print("Descriptive Statistics:")
print(stats)# 1. Descriptive Statistics
stats = df[["temperature_denoised", "air_humidity_denoised", "soil_moisture_denoised"]].describe()
print("Descriptive Statistics:")
print(stats)

# 2. Time Series Visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Temperature trend
axes[0].plot(df["timestamp"], df["temperature_denoised"], color="red")
axes[0].set_title("Temperature Trend (Denoised)")
axes[0].set_ylabel("Temperature (℃)")
axes[0].grid(True)

# Air humidity trend
axes[1].plot(df["timestamp"], df["air_humidity_denoised"], color="blue")
axes[1].set_title("Air Humidity Trend (Denoised)")
axes[1].set_ylabel("Air Humidity (%)")
axes[1].grid(True)

# Soil moisture trend
axes[2].plot(df["timestamp"], df["soil_moisture_denoised"], color="green")
axes[2].set_title("Soil Moisture Trend (Denoised)")
axes[2].set_ylabel("Soil Moisture (0-1)")
axes[2].set_xlabel("Time")
axes[2].grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("time_series_trends.png")
plt.show()
