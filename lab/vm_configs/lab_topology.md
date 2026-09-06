# Virtual Lab Network Topology & Configuration

## Architecture Diagram

```
+-------------------------------------------------------------------------+
|                  VirtualBox Internal Network (nids_intnet)              |
|                                                                         |
|  +-------------------+   +--------------------+   +------------------+  |
|  |   kali-attacker   |   |   ubuntu-victim    |   |    win-victim    |  |
|  |    (10.0.0.5)     |   |    (10.0.0.10)     |   |    (10.0.0.15)   |  |
|  +---------+---------+   +---------+----------+   +--------+---------+  |
|            |                       |                       |            |
|            +-----------------------+-----------------------+            |
|                                    |                                    |
|                        +-----------+-----------+                        |
|                        |     monitor-node      |                        |
|                        |      (10.0.0.2)       |                        |
|                        | Zeek / Suricata PCAP  |                        |
|                        +-----------------------+                        |
+-------------------------------------------------------------------------+
```

## Network Settings
- **Subnet:** 10.0.0.0/24
- **Adapter Type:** Intel PRO/1000 MT Desktop (82540EM)
- **Promiscuous Mode:** Allow All (on monitor node)
- **DHCP:** Disabled (Static IP configuration on all endpoints)
