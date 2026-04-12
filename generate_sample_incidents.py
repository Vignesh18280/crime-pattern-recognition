import pandas as pd
import numpy as np
import os

def generate_sample_incident(incident_type, num_rows=100, num_attack_rows=10):
    """
    Generates a sample network incident log with mixed normal and attack traffic.
    """
    np.random.seed(42 if incident_type == 'dos' else 43)

    # Base features (simplified from UNSW-NB15)
    # We'll use spkts, sbytes, dur as core features to show anomaly detection
    features = ['spkts', 'sbytes', 'dur'] + [f'feature_{i}' for i in range(29)] # Total 32 features

    # Generate normal traffic
    normal_data = pd.DataFrame({
        'spkts': np.random.randint(1, 10, num_rows),
        'sbytes': np.random.randint(64, 1024, num_rows),
        'dur': np.random.uniform(0.1, 5.0, num_rows),
    })
    for col in features[3:]:
        normal_data[col] = np.random.uniform(0, 1, num_rows)

    # Generate attack traffic based on type
    attack_data = pd.DataFrame()
    if incident_type == 'dos':
        attack_data = pd.DataFrame({
            'spkts': np.random.randint(400, 600, num_attack_rows),
            'sbytes': np.random.randint(40000, 60000, num_attack_rows),
            'dur': np.random.uniform(0.001, 0.01, num_attack_rows),
        })
    elif incident_type == 'fuzzers':
        attack_data = pd.DataFrame({
            'spkts': np.random.randint(100, 200, num_attack_rows),
            'sbytes': np.random.randint(5000, 10000, num_attack_rows),
            'dur': np.random.uniform(0.05, 0.1, num_attack_rows),
        })
    else:
        raise ValueError("Unknown incident type")

    for col in features[3:]: # Fill other features for attack data
        attack_data[col] = np.random.uniform(0, 1, num_attack_rows) * (1.5 if incident_type == 'fuzzers' else 0.5)

    # Mix normal and attack traffic
    combined_data = pd.concat([normal_data, attack_data]).sample(frac=1).reset_index(drop=True)
    return combined_data

if __name__ == "__main__":
    output_dir = "data/sample_incidents"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating sample incident files in {output_dir}...")

    # Generate DoS incident files
    incident_dos_a = generate_sample_incident('dos')
    incident_dos_a.to_csv(os.path.join(output_dir, 'incident_dos_a.csv'), index=False)
    print("Generated incident_dos_a.csv")

    incident_dos_b = generate_sample_incident('dos')
    incident_dos_b.to_csv(os.path.join(output_dir, 'incident_dos_b.csv'), index=False)
    print("Generated incident_dos_b.csv")

    # Generate Fuzzers incident file
    incident_fuzzers_a = generate_sample_incident('fuzzers')
    incident_fuzzers_a.to_csv(os.path.join(output_dir, 'incident_fuzzers_a.csv'), index=False)
    print("Generated incident_fuzzers_a.csv")

    print("\nSample files generated. Use these for drag-and-drop comparison in the UI.")
