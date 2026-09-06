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
