# Provenance of this directory

**These captures were SIMULATED, not recorded from a physical lab.**

They were produced by `src/data/lab_simulator.py` from the behavioural model in
`src/data/traffic_model.py`, using the time windows documented in
`lab/capture_logs/session_labels.csv`.

The PCAP files are valid libpcap files and the `conn.log` files are valid Zeek
JSON-lines logs, so the extraction code path is exercised for real. The *packets
themselves* are synthetic.

To replace them with real captures:

1. Run the playbook in `lab/attack_scripts/playbook.md` against your lab.
2. Capture with `zeek -i <iface> LogAscii::use_json=T` (or save a pcap with
   `tcpdump -i <iface> -w capture.pcap`).
3. Drop the resulting `*.pcap` / `conn.log` files in this directory.
4. Re-run `python scripts/generate_lab_capture.py --extract-only`.
5. Delete this file.
