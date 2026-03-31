"""
demo_generator.py — AnomalyIQ Synthetic Fraud Dataset Generator

Generates 2000 rows with ~8% fraud rate, random_state=42.
Columns: amount, hour, velocity_1hr, distance_from_home,
         foreign_transaction, high_risk_merchant, deviation_score, is_fraud
"""

import numpy as np
import pandas as pd


def generate_demo_dataset(random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic fraud detection dataset.

    Distribution details:
    - ~8% fraud rate (≈160 fraudulent, ≈1840 normal)
    - Fraud: high amounts, late night hours, high velocity, foreign transactions
    - Normal: low amounts, daytime hours, low velocity, domestic transactions
    """
    rng = np.random.default_rng(random_state)

    n_total = 2000
    n_fraud = int(n_total * 0.08)  # ~160 fraud rows
    n_normal = n_total - n_fraud   # ~1840 normal rows

    # Create a pool of fake users to allow natural clustering for the graph engine
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon", "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy", "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper"]
    
    # 500 normal identities
    normal_names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(500)]
    normal_ccs = [f"45{rng.integers(1000, 9999)}1111{rng.integers(1000, 9999)}" for _ in range(500)]
    
    # 20 fraud identities (forces high concentration/hub formation for graph engine tests)
    fraud_names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(20)]
    fraud_ccs = [f"51{rng.integers(1000, 9999)}4444{rng.integers(1000, 9999)}" for _ in range(20)]

    # ── Fraud rows ──────────────────────────────────────────────────────────
    fraud_idx = rng.integers(0, 20, size=n_fraud)
    fraud_amount = rng.exponential(scale=400, size=n_fraud).clip(10, 10000)
    fraud_hour = rng.integers(1, 6, size=n_fraud)  # 1–5 AM
    fraud_velocity = rng.poisson(lam=6, size=n_fraud).clip(0, 30)
    fraud_distance = rng.exponential(scale=150, size=n_fraud).clip(0, 500)
    fraud_foreign = rng.binomial(1, 0.65, size=n_fraud)
    fraud_highRisk = rng.binomial(1, 0.70, size=n_fraud)
    fraud_deviation = rng.beta(7, 2, size=n_fraud)  # skewed high

    # ── Normal rows ─────────────────────────────────────────────────────────
    normal_idx = rng.integers(0, 500, size=n_normal)
    normal_amount = rng.exponential(scale=80, size=n_normal).clip(1, 3000)
    normal_hour = rng.integers(8, 23, size=n_normal)  # 8–22h
    normal_velocity = rng.poisson(lam=1.5, size=n_normal).clip(0, 15)
    normal_distance = rng.exponential(scale=20, size=n_normal).clip(0, 200)
    normal_foreign = rng.binomial(1, 0.05, size=n_normal)
    normal_highRisk = rng.binomial(1, 0.10, size=n_normal)
    normal_deviation = rng.beta(2, 8, size=n_normal)  # skewed low

    # ── Combine ─────────────────────────────────────────────────────────────
    df_fraud = pd.DataFrame({
        "cc_num": [fraud_ccs[i] for i in fraud_idx],
        "name": [fraud_names[i] for i in fraud_idx],
        "amount": fraud_amount,
        "hour": fraud_hour,
        "velocity_1hr": fraud_velocity,
        "distance_from_home": fraud_distance,
        "foreign_transaction": fraud_foreign,
        "high_risk_merchant": fraud_highRisk,
        "deviation_score": fraud_deviation,
        "is_fraud": 1
    })

    df_normal = pd.DataFrame({
        "cc_num": [normal_ccs[i] for i in normal_idx],
        "name": [normal_names[i] for i in normal_idx],
        "amount": normal_amount,
        "hour": normal_hour,
        "velocity_1hr": normal_velocity,
        "distance_from_home": normal_distance,
        "foreign_transaction": normal_foreign,
        "high_risk_merchant": normal_highRisk,
        "deviation_score": normal_deviation,
        "is_fraud": 0
    })

    df = pd.concat([df_fraud, df_normal], ignore_index=True)

    # Shuffle deterministically
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Round numeric columns for cleanliness
    df["amount"] = df["amount"].round(2)
    df["distance_from_home"] = df["distance_from_home"].round(2)
    df["deviation_score"] = df["deviation_score"].round(4)

    return df


def get_demo_csv_bytes(random_state: int = 42) -> bytes:
    """Return the demo dataset as CSV bytes for upload simulation."""
    df = generate_demo_dataset(random_state=random_state)
    return df.to_csv(index=False).encode("utf-8")


if __name__ == "__main__":
    df = generate_demo_dataset()
    print(f"Total rows: {len(df)}")
    print(f"Fraud rows: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Normal rows: {(df['is_fraud'] == 0).sum()}")
    print("\nSample fraud:")
    print(df[df["is_fraud"] == 1].head(3).to_string())
    print("\nSample normal:")
    print(df[df["is_fraud"] == 0].head(3).to_string())
