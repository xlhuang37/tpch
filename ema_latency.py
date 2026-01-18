import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('004_np_default_58_p_default_140_lam0.080808_0.080808_len3600_s123_round_raw.csv')

# Filter out DROPPED rows (convert to numeric, coercing errors to NaN, then drop NaN)
df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
df = df.dropna(subset=['latency_ms'])

# Sort by arrival time
df = df.sort_values('arrival_ms').reset_index(drop=True)

# Compute EMA with different spans
span = 20  # Adjust this for smoothing (higher = smoother)
df['ema_latency'] = df['latency_ms'].ewm(span=span, adjust=False).mean()

# Create the plot
plt.figure(figsize=(12, 6))

# Plot raw latency
plt.scatter(df['arrival_ms'] / 1000, df['latency_ms'], alpha=0.3, s=10, label='Raw Latency', color='#7f8c8d')

# Plot EMA
plt.plot(df['arrival_ms'] / 1000, df['ema_latency'], linewidth=2, label=f'EMA (span={span})', color='#e74c3c')

plt.xlabel('Arrival Time (s)', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.title('Query Latency with Exponential Moving Average', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig('ema_latency.png', dpi=150)
plt.show()

print(f"Plot saved to ema_latency.png")
print(f"\nLatency Statistics:")
print(f"  Mean:   {df['latency_ms'].mean():.2f} ms")
print(f"  Median: {df['latency_ms'].median():.2f} ms")
print(f"  Std:    {df['latency_ms'].std():.2f} ms")
