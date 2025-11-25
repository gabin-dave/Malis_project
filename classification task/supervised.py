import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

classes = ["benign", "ransomware", "trojan", "worm", "spyware", "adware"]

# total samples (you can change this)
n_samples_total = 1800
n_per_class = n_samples_total // len(classes)

# ----------------------------
# Helper functions
# ----------------------------

def clip_int(x, low, high):
    return np.clip(np.round(x).astype(int), low, high)

def clip_float(x, low, high):
    return np.clip(x.astype(float), low, high)

def bernoulli(p, size):
    return np.random.binomial(1, p, size=size)

# ----------------------------
# Sampling for each class
# ----------------------------

def generate_class_data(label, n):
    """
    Generate synthetic data for one class.
    Returns a dict of columns.
    """

    if label == "benign":
        file_size_kb = np.random.lognormal(mean=7.5, sigma=0.5, size=n)  # ~1k-10k KB
        num_imports = clip_int(np.random.normal(120, 30, n), 20, 300)
        num_sections = clip_int(np.random.normal(5, 1, n), 3, 10)
        avg_section_entropy = clip_float(np.random.normal(5.0, 0.7, n), 2.0, 7.5)
        num_strings = clip_int(np.random.normal(800, 200, n), 50, 3000)
        packed = bernoulli(0.1, n)
        uses_network_api = bernoulli(0.4, n)
        uses_crypto_api = bernoulli(0.1, n)
        uses_filesystem_api = bernoulli(0.5, n)

        avg_cpu_usage = clip_float(np.random.normal(10, 5, n), 0, 60)
        num_files_created = clip_int(np.random.poisson(2, n), 0, 50)
        num_registry_keys_modified = clip_int(np.random.poisson(1, n), 0, 20)
        num_outbound_connections = clip_int(np.random.poisson(2, n), 0, 30)
        num_unique_ips = clip_int(np.random.poisson(1, n), 0, 10)
        persistence_mechanism = bernoulli(0.1, n)
        num_processes_spawned = clip_int(np.random.poisson(1, n), 0, 20)

    elif label == "ransomware":
        file_size_kb = np.random.lognormal(mean=7.3, sigma=0.6, size=n)
        num_imports = clip_int(np.random.normal(90, 25, n), 10, 250)
        num_sections = clip_int(np.random.normal(5, 1, n), 3, 10)
        avg_section_entropy = clip_float(np.random.normal(6.5, 0.5, n), 3.5, 8.0)
        num_strings = clip_int(np.random.normal(400, 150, n), 20, 2000)
        packed = bernoulli(0.7, n)
        uses_network_api = bernoulli(0.5, n)
        uses_crypto_api = bernoulli(0.9, n)
        uses_filesystem_api = bernoulli(0.9, n)

        avg_cpu_usage = clip_float(np.random.normal(50, 15, n), 5, 100)
        num_files_created = clip_int(np.random.poisson(200, n), 10, 1000)
        num_registry_keys_modified = clip_int(np.random.poisson(15, n), 0, 100)
        num_outbound_connections = clip_int(np.random.poisson(8, n), 0, 100)
        num_unique_ips = clip_int(np.random.poisson(4, n), 0, 40)
        persistence_mechanism = bernoulli(0.8, n)
        num_processes_spawned = clip_int(np.random.poisson(6, n), 0, 50)

    elif label == "trojan":
        file_size_kb = np.random.lognormal(mean=7.0, sigma=0.7, size=n)
        num_imports = clip_int(np.random.normal(110, 35, n), 15, 300)
        num_sections = clip_int(np.random.normal(5, 1, n), 3, 10)
        avg_section_entropy = clip_float(np.random.normal(6.0, 0.6, n), 3.0, 8.0)
        num_strings = clip_int(np.random.normal(600, 250, n), 20, 2500)
        packed = bernoulli(0.5, n)
        uses_network_api = bernoulli(0.8, n)
        uses_crypto_api = bernoulli(0.3, n)
        uses_filesystem_api = bernoulli(0.8, n)

        avg_cpu_usage = clip_float(np.random.normal(35, 12, n), 5, 95)
        num_files_created = clip_int(np.random.poisson(40, n), 0, 300)
        num_registry_keys_modified = clip_int(np.random.poisson(10, n), 0, 80)
        num_outbound_connections = clip_int(np.random.poisson(25, n), 0, 200)
        num_unique_ips = clip_int(np.random.poisson(10, n), 0, 60)
        persistence_mechanism = bernoulli(0.8, n)
        num_processes_spawned = clip_int(np.random.poisson(8, n), 0, 60)

    elif label == "worm":
        file_size_kb = np.random.lognormal(mean=6.5, sigma=0.8, size=n)
        num_imports = clip_int(np.random.normal(80, 25, n), 10, 250)
        num_sections = clip_int(np.random.normal(4.5, 1, n), 3, 10)
        avg_section_entropy = clip_float(np.random.normal(6.3, 0.7, n), 3.0, 8.0)
        num_strings = clip_int(np.random.normal(500, 200, n), 20, 2500)
        packed = bernoulli(0.6, n)
        uses_network_api = bernoulli(0.95, n)
        uses_crypto_api = bernoulli(0.2, n)
        uses_filesystem_api = bernoulli(0.7, n)

        avg_cpu_usage = clip_float(np.random.normal(45, 15, n), 5, 100)
        num_files_created = clip_int(np.random.poisson(60, n), 0, 400)
        num_registry_keys_modified = clip_int(np.random.poisson(8, n), 0, 60)
        num_outbound_connections = clip_int(np.random.poisson(60, n), 5, 400)
        num_unique_ips = clip_int(np.random.poisson(25, n), 1, 200)
        persistence_mechanism = bernoulli(0.7, n)
        num_processes_spawned = clip_int(np.random.poisson(15, n), 0, 80)

    elif label == "spyware":
        file_size_kb = np.random.lognormal(mean=7.0, sigma=0.6, size=n)
        num_imports = clip_int(np.random.normal(100, 30, n), 15, 300)
        num_sections = clip_int(np.random.normal(5, 1, n), 3, 10)
        avg_section_entropy = clip_float(np.random.normal(6.0, 0.6, n), 3.0, 8.0)
        num_strings = clip_int(np.random.normal(700, 250, n), 20, 3000)
        packed = bernoulli(0.5, n)
        uses_network_api = bernoulli(0.85, n)
        uses_crypto_api = bernoulli(0.4, n)
        uses_filesystem_api = bernoulli(0.8, n)

        avg_cpu_usage = clip_float(np.random.normal(30, 10, n), 5, 90)
        num_files_created = clip_int(np.random.poisson(30, n), 0, 300)
        num_registry_keys_modified = clip_int(np.random.poisson(20, n), 0, 150)
        num_outbound_connections = clip_int(np.random.poisson(20, n), 0, 200)
        num_unique_ips = clip_int(np.random.poisson(8, n), 0, 80)
        persistence_mechanism = bernoulli(0.9, n)
        num_processes_spawned = clip_int(np.random.poisson(7, n), 0, 50)

    elif label == "adware":
        file_size_kb = np.random.lognormal(mean=6.8, sigma=0.7, size=n)
        num_imports = clip_int(np.random.normal(90, 25, n), 10, 250)
        num_sections = clip_int(np.random.normal(5, 1, n), 3, 10)
        avg_section_entropy = clip_float(np.random.normal(5.8, 0.7, n), 3.0, 8.0)
        num_strings = clip_int(np.random.normal(650, 220, n), 20, 2800)
        packed = bernoulli(0.4, n)
        uses_network_api = bernoulli(0.9, n)
        uses_crypto_api = bernoulli(0.2, n)
        uses_filesystem_api = bernoulli(0.6, n)

        avg_cpu_usage = clip_float(np.random.normal(40, 15, n), 5, 100)
        num_files_created = clip_int(np.random.poisson(20, n), 0, 200)
        num_registry_keys_modified = clip_int(np.random.poisson(6, n), 0, 60)
        num_outbound_connections = clip_int(np.random.poisson(35, n), 1, 300)
        num_unique_ips = clip_int(np.random.poisson(12, n), 0, 100)
        persistence_mechanism = bernoulli(0.6, n)
        num_processes_spawned = clip_int(np.random.poisson(5, n), 0, 40)

    else:
        raise ValueError(f"Unknown label: {label}")

    data = {
        "file_size_kb": file_size_kb,
        "num_imports": num_imports,
        "num_sections": num_sections,
        "avg_section_entropy": avg_section_entropy,
        "num_strings": num_strings,
        "packed": packed,
        "uses_network_api": uses_network_api,
        "uses_crypto_api": uses_crypto_api,
        "uses_filesystem_api": uses_filesystem_api,
        "avg_cpu_usage": avg_cpu_usage,
        "num_files_created": num_files_created,
        "num_registry_keys_modified": num_registry_keys_modified,
        "num_outbound_connections": num_outbound_connections,
        "num_unique_ips": num_unique_ips,
        "persistence_mechanism": persistence_mechanism,
        "num_processes_spawned": num_processes_spawned,
        "label": [label] * n,
    }

    return data

# ----------------------------
# Generate full dataset
# ----------------------------

all_rows = []

for label in classes:
    data = generate_class_data(label, n_per_class)
    df_label = pd.DataFrame(data)
    all_rows.append(df_label)

df = pd.concat(all_rows, ignore_index=True)


# ----------------------------
# Add some noise to numeric features
# ----------------------------

numeric_cols = [
    "file_size_kb",
    "num_imports",
    "num_sections",
    "avg_section_entropy",
    "num_strings",
    "avg_cpu_usage",
    "num_files_created",
    "num_registry_keys_modified",
    "num_outbound_connections",
    "num_unique_ips",
    "num_processes_spawned",
]

for col in numeric_cols:
    std = df[col].std()
    if std > 0:
        noise = np.random.normal(loc=0.0, scale=0.05 * std, size=len(df))
        df[col] = df[col] + noise

# Re-clip to keep values in realistic ranges after noise
df["file_size_kb"] = clip_float(df["file_size_kb"], 10, 100000)
df["num_imports"] = clip_int(df["num_imports"], 0, 500)
df["num_sections"] = clip_int(df["num_sections"], 1, 12)
df["avg_section_entropy"] = clip_float(df["avg_section_entropy"], 0.0, 8.0)
df["num_strings"] = clip_int(df["num_strings"], 0, 10000)
df["avg_cpu_usage"] = clip_float(df["avg_cpu_usage"], 0, 100)
df["num_files_created"] = clip_int(df["num_files_created"], 0, 2000)
df["num_registry_keys_modified"] = clip_int(df["num_registry_keys_modified"], 0, 1000)
df["num_outbound_connections"] = clip_int(df["num_outbound_connections"], 0, 2000)
df["num_unique_ips"] = clip_int(df["num_unique_ips"], 0, 500)
df["num_processes_spawned"] = clip_int(df["num_processes_spawned"], 0, 500)

# Shuffle rows
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# ----------------------------
# Save to CSV
# ----------------------------

df.to_csv("malware_classification.csv", index=False)
print("Saved malware_classification.csv with shape:", df.shape)
print(df.head())
