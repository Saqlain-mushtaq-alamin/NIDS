# Virtual Lab Attack Playbook & Traffic Capture Plan

This playbook defines repeatable attack scenarios and traffic generation procedures executed from the Kali Linux VM against target victim VMs on the isolated virtual network.

---

## 1. Target Infrastructure

| VM Name | Role | IP Address | OS | Services |
|---------|------|------------|----|----------|
| `kali-attacker` | Attacker | 10.0.0.5 | Kali Linux 2024.x | hping3, nmap, hydra, metasploit, slowloris |
| `ubuntu-victim` | Primary Target | 10.0.0.10 | Ubuntu Server 22.04 | Apache (HTTP 80), OpenSSH (22), VSFTPD (21) |
| `win-victim` | Secondary Target | 10.0.0.15 | Windows 10 Enterprise | RDP (3389), SMB (445), HTTP (8080) |
| `monitor-node` | Traffic Capture | 10.0.0.2 | Ubuntu Server 22.04 | Zeek 6.x, Suricata 7.x, Promiscuous Mode |

---

## 2. Attack Execution Matrix

| Sequence | Attack Category | Command / Script | Target Service | Expected Label |
|----------|-----------------|------------------|----------------|----------------|
| 1 | Baseline Normal | `curl`, `wget`, `ab -n 1000 -c 10` | HTTP / SSH | `Benign` |
| 2 | DoS / SYN Flood | `hping3 -S -p 80 --flood 10.0.0.10` | Apache HTTP | `DoS` |
| 3 | Port Scan | `nmap -sS -sV -p 1-65535 10.0.0.10` | All Ports | `PortScan` |
| 4 | SSH Brute Force | `hydra -l admin -P wordlist.txt 10.0.0.10 ssh` | OpenSSH | `BruteForce` |
| 5 | Web Attack (SQLi/XSS) | `sqlmap -u "http://10.0.0.10/vulnerable.php?id=1" --batch` | Apache / PHP | `WebAttack` |
| 6 | C2 / Beaconing | Metasploit reverse TCP handler + HTTP beacon script | Reverse Shell | `Botnet` |

---

## 3. Session Capture Procedure

1. Start monitor interface in promiscuous mode:
   ```bash
   sudo ip link set dev eth0 promisc on
   ```
2. Launch Zeek capture:
   ```bash
   sudo zeek -i eth0 LogAscii::use_json=T
   ```
3. Execute scheduled playbook scenarios recording start/end timestamps in `session_labels.csv`.
4. Export PCAP and Zeek log files to `data/lab_captures/`.

---

## 4. Capture Sessions

Phase 2 requires at least three independent capture sessions on different days to
introduce natural temporal variation (plan section 2.1.10). The authoritative
machine-readable record is `lab/capture_logs/session_labels.csv`; the tables
below mirror it.

### Session 1 — 2026-09-01, 10:00–11:00 UTC (60 min)
| Time Window | Activity | Tool | Target | Label |
|---|---|---|---|---|
| 10:00–10:15 | Normal browsing, SSH, FTP | `curl`, `wget`, `ssh`, `ftp` | 10.0.0.10 | Benign |
| 10:15–10:20 | SYN Flood | `hping3 -S -p 80 --flood` | 10.0.0.10 | DoS |
| 10:20–10:30 | SYN Stealth Scan | `nmap -sS -p 1-65535` | 10.0.0.10 | PortScan |
| 10:30–10:40 | SSH Brute Force | `hydra -l admin -P rockyou.txt` | 10.0.0.10 | BruteForce |
| 10:40–10:50 | SQL Injection | `sqlmap --batch` | 10.0.0.10 | WebAttack |
| 10:50–11:00 | Reverse Shell + Beaconing | `metasploit` | 10.0.0.15 | Botnet |

### Session 2 — 2026-09-03, 14:00–15:00 UTC (60 min)
| Time Window | Activity | Tool | Target | Label |
|---|---|---|---|---|
| 14:00–14:15 | Load generation + browsing | `ab -n 1000 -c 10`, `curl` | 10.0.0.10 | Benign |
| 14:15–14:20 | Slowloris | `slowloris` | 10.0.0.10 | DoS |
| 14:20–14:30 | Service / OS Detection | `nmap -sV -O` | 10.0.0.10 | PortScan |
| 14:30–14:40 | FTP Brute Force | `hydra ... ftp` | 10.0.0.10 | BruteForce |
| 14:40–14:50 | XSS + Directory Traversal | `curl` | 10.0.0.10 | WebAttack |
| 14:50–15:00 | HTTPS Beacon Callbacks | `curl` loop | 10.0.0.15 | Botnet |

### Session 3 — 2026-09-05, 09:00–10:00 UTC (60 min)
| Time Window | Activity | Tool | Target | Label |
|---|---|---|---|---|
| 09:00–09:15 | SSH sessions + FTP transfers | `ssh`, `ftp` | 10.0.0.10 | Benign |
| 09:15–09:20 | UDP Flood + SYN-ACK Flood | `hping3` | 10.0.0.10 | DoS |
| 09:20–09:30 | Connect Scan (TCP full) | `nmap -sT -p 1-65535` | 10.0.0.10 | PortScan |
| 09:30–09:40 | Credential Stuffing | `hydra` (SSH + FTP) | 10.0.0.10 | BruteForce |
| 09:40–09:50 | Time / Boolean SQLi | `sqlmap --time-based` | 10.0.0.10 | WebAttack |
| 09:50–10:00 | Staged Reverse Shell | `metasploit` keepalive | 10.0.0.15 | Botnet |

---

## 5. Capture Status

> [!IMPORTANT]
> The capture artefacts currently present in `data/lab_captures/` were **simulated**
> by `src/data/lab_simulator.py`, not recorded from this VirtualBox topology. See
> `data/lab_captures/PROVENANCE.md`. The commands, targets and time windows above
> are the specification the simulator implements; running them on real VMs and
> replacing the simulated files converts the lab domain from a validated stand-in
> into genuinely self-captured traffic.

