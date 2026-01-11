import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

new_data_path = r"C:\Users\bangu\Desktop\winter_sensor_data.csv"

try:
    df = pd.read_csv(new_data_path)
    print(f"✅ Data loaded successfully from '{new_data_path}'")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Error: The file '{new_data_path}' was not found. Please check the file path and name.")

# 后续代码不变（省略，保持原功能）
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp"] = df["timestamp"].fillna(method='bfill').fillna(method='ffill')

required_original_columns = ["soil_moisture", "temperature", "air_humidity"]
missing_columns = [col for col in required_original_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"❌ Error: Missing required columns: {', '.join(missing_columns)}")

print("\nRaw data preview:")
print(df.head())

def moving_average_denoise(data, window_size=3):
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("❌ Window size must be a positive integer")
    return data.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')

window_size = 3
df["soil_moisture_denoised"] = moving_average_denoise(df["soil_moisture"], window_size)
df["temperature_denoised"] = moving_average_denoise(df["temperature"], window_size)
df["air_humidity_denoised"] = moving_average_denoise(df["air_humidity"], window_size)

scaler = MinMaxScaler(feature_range=(0, 1))
denoised_columns = ["temperature_denoised", "air_humidity_denoised", "soil_moisture_denoised"]
df[["temperature_norm", "air_humidity_norm", "soil_moisture_norm"]] = scaler.fit_transform(
    df[denoised_columns]
)

df.to_csv("sensor_data_processed.csv", index=False)
print("\n✅ Preprocessed data saved as 'sensor_data_processed.csv'")

print("\nPreprocessed data preview (soil moisture related):")
preview_columns = ["timestamp", "soil_moisture", "soil_moisture_denoised", "soil_moisture_norm"]
print(df[preview_columns].head())

plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["soil_moisture"], label="Original Soil Moisture", alpha=0.5, color="lightgray")
plt.plot(df["timestamp"], df["soil_moisture_denoised"], label="Denoised Soil Moisture", linewidth=2, color="darkgreen")
plt.title("Soil Moisture Denoising Effect (Moving Average Window=3)")
plt.xlabel("Time")
plt.ylabel("Soil Moisture (0-1)")
plt.legend(loc="best")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("denoising_effect.png", dpi=300, bbox_inches="tight")
plt.show()
