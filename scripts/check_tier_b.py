import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import load_dataset

print("Loading CIC-IDS2017...")
cic_df = load_dataset("cic-ids-2017", drop_non_features=True, enforce_provenance=True)
print(f"CIC shape: {cic_df.shape}")

print("Loading UNSW-NB15...")
unsw_df = load_dataset("unsw-nb15", drop_non_features=True, enforce_provenance=True)
print(f"UNSW shape: {unsw_df.shape}")

# Filter to benign rows
cic_benign = cic_df[cic_df["label"].str.lower() == "benign"]
unsw_benign = unsw_df[unsw_df["label"].str.lower() == "benign"]
print(f"CIC Benign: {len(cic_benign):,}, UNSW Benign: {len(unsw_benign):,}")

# Tier B features after harmonization:
# total_fwd_bytes: CIC 'Total Length of Fwd Packets' (bytes), UNSW 'sbytes' (bytes)
# total_bwd_bytes: CIC 'Total Length of Bwd Packets' (bytes), UNSW 'dbytes' (bytes)
# fwd_packet_length_mean: CIC 'Fwd Packet Length Mean' (bytes), UNSW 'smean' (bytes)
# bwd_packet_length_mean: CIC 'Bwd Packet Length Mean' (bytes), UNSW 'dmean' (bytes)
# flow_packet_rate: CIC 'Flow Packets/s' (pkts/s), UNSW 'rate' (pkts/s)

tier_b_map = {
    "total_fwd_bytes": ("Total Length of Fwd Packets", "sbytes"),
    "total_bwd_bytes": ("Total Length of Bwd Packets", "dbytes"),
    "fwd_packet_length_mean": ("Fwd Packet Length Mean", "smean"),
    "bwd_packet_length_mean": ("Bwd Packet Length Mean", "dmean"),
    "flow_packet_rate": ("Flow Packets/s", "rate"),
}

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
print("\n--- BENIGN QUANTILES COMPARISON (Tier B) ---")
for canon, (cic_col, unsw_col) in tier_b_map.items():
    c_s = pd.to_numeric(cic_benign[cic_col], errors="coerce").dropna()
    u_s = pd.to_numeric(unsw_benign[unsw_col], errors="coerce").dropna()
    c_q = np.quantile(c_s, quantiles)
    u_q = np.quantile(u_s, quantiles)
    print(f"\nFeature: {canon} (CIC: '{cic_col}' vs UNSW: '{unsw_col}')")
    print(f"  Quantile | {'CIC-IDS2017':>14} | {'UNSW-NB15':>14} | Ratio (UNSW/CIC)")
    for q, cv, uv in zip(quantiles, c_q, u_q):
        ratio_str = f"{uv/cv:12.3f}" if cv > 0 else "         inf"
        print(f"  {q*100:6.1f}%  | {cv:14.2f} | {uv:14.2f} | {ratio_str}")

# Zero-byte flows
print("\n--- ZERO-BYTE FLOW FRACTIONS ---")
c_fwd_0 = (cic_df["Total Length of Fwd Packets"] == 0).mean()
c_bwd_0 = (cic_df["Total Length of Bwd Packets"] == 0).mean()
u_fwd_0 = (unsw_df["sbytes"] == 0).mean()
u_bwd_0 = (unsw_df["dbytes"] == 0).mean()
print(f"CIC zero total_fwd_bytes: {c_fwd_0:.4%}")
print(f"CIC zero total_bwd_bytes: {c_bwd_0:.4%}")
print(f"UNSW zero sbytes:         {u_fwd_0:.4%}")
print(f"UNSW zero dbytes:         {u_bwd_0:.4%}")

# Minimum bytes per packet (non-zero flows)
print("\n--- MINIMUM BYTES PER PACKET ---")
c_nz_fwd = cic_df[(cic_df["Total Length of Fwd Packets"] > 0) & (cic_df["Total Fwd Packets"] > 0)]
c_min_bpp_fwd = (c_nz_fwd["Total Length of Fwd Packets"] / c_nz_fwd["Total Fwd Packets"]).min()
u_nz_fwd = unsw_df[(unsw_df["sbytes"] > 0) & (unsw_df["spkts"] > 0)]
u_min_bpp_fwd = (u_nz_fwd["sbytes"] / u_nz_fwd["spkts"]).min()

c_min_len = cic_df["Fwd Packet Length Min"].dropna().min()
u_min_len = unsw_df["smean"].dropna().min()

print(f"CIC min fwd bytes/pkt:  {c_min_bpp_fwd:.2f} (Fwd Packet Length Min: {c_min_len:.2f})")
print(f"UNSW min fwd bytes/pkt: {u_min_bpp_fwd:.2f} (smean min: {u_min_len:.2f})")

# UNSW rate formula check
print("\n--- UNSW RATE DEFINITION CHECK ---")
unsw_nz_dur = unsw_df[unsw_df["dur"] > 0].copy()
spkts = unsw_nz_dur["spkts"]
dpkts = unsw_nz_dur["dpkts"]
dur = unsw_nz_dur["dur"]
rate_actual = unsw_nz_dur["rate"]

r1 = (spkts + dpkts) / dur
r2 = (spkts + dpkts - 1) / dur

err1 = np.abs(r1 - rate_actual)
err2 = np.abs(r2 - rate_actual)

close1 = np.isclose(r1, rate_actual, rtol=1e-3, atol=1e-2).mean()
close2 = np.isclose(r2, rate_actual, rtol=1e-3, atol=1e-2).mean()

print(f"UNSW rows with dur > 0: {len(unsw_nz_dur):,}")
print(f"Formula 1: (spkts + dpkts) / dur     -> Mean Abs Err: {err1.mean():.4f}, Max Err: {err1.max():.4f}, Close: {close1:.2%}")
print(f"Formula 2: (spkts + dpkts - 1) / dur -> Mean Abs Err: {err2.mean():.4f}, Max Err: {err2.max():.4f}, Close: {close2:.2%}")

