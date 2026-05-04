# Hardware Topology Detection — Summary

## Overview

Detect hardware topology on single node: compute (GPUs, CPU, memory), communication (RDMA, NICs, GPUDirect), storage (disks, shared filesystems). Generates YAML report, terminal summary, and memory entries for other skills to consume.

## Output

- **YAML file**: Machine-readable topology report at `{output_path}` (default: `./topo_report.yaml`)
- **Terminal summary**: Human-readable overview grouped by Compute/Communication/Storage/Recommendations
- **Memory entries**: Four keys for other skills to consume:
  - `topo_compute`: GPU count, model, memory, interconnect type, NVLink bandwidth
  - `topo_comm`: RDMA availability, NIC type/count/rate, GPUDirect RDMA
  - `topo_storage`: Local storage type/size, shared storage, sequential write speed
  - `node_topology`: One-line summary combining all above

## What It Detects

### Compute Detection
- GPU inventory: count, model, memory, compute capability, driver version
- GPU interconnect: NVSwitch (full mesh), NVLink-P2P (partial), or PCIe-only
- NVLink bandwidth: links per GPU, bandwidth per link
- CPU topology: model, sockets, cores, threads, NUMA nodes
- GPU-NUMA affinity: which GPUs belong to which NUMA node
- System memory: total and available

### Communication Detection
- RDMA devices: InfiniBand or RoCE
- NIC details: rate (HDR/EDR/FDR/NDR), state, NUMA affinity
- NIC-GPU affinity: which NICs are closest to which GPUs (PCIe distance)
- GPUDirect RDMA: whether `nv_peer_mem` or `nvidia_peermem` kernel module is loaded
- NCCL version and environment variables

### Storage Detection
- Block devices: NVMe SSD, SATA/SAS SSD, HDD
- Mount points: filesystem type, size, usage percentage
- Shared storage: NFS, Lustre, GPFS, Ceph, BeeGFS detection
- IO benchmark (optional, requires user confirmation): sequential write speed, random read IOPS

## Execution Rules

- Run each detection step in order. If a command fails or is unavailable, note it as "unavailable" and continue.
- Do NOT install any packages. Only use tools already present on the system.
- IO benchmark writes temporary files — ask user for confirmation before running.
- Parse all command outputs yourself. Do NOT ask user to interpret raw output.
- After all steps, assemble YAML report, print summary, and write memory entries.

## Key Detection Commands

### GPU Topology
```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
nvidia-smi topo -m  # interconnect matrix + NIC-GPU affinity
nvidia-smi nvlink -s  # NVLink bandwidth
```

### RDMA Detection
```bash
ls /sys/class/infiniband/  # RDMA devices
cat /sys/class/infiniband/<device>/ports/1/rate  # NIC rate
lsmod | grep -E 'nv_peer_mem|nvidia_peermem'  # GPUDirect RDMA
```

### Storage Detection
```bash
lsblk -d -o NAME,SIZE,TYPE,ROTA,TRAN,MODEL  # block devices
df -hT  # mount points
mount | grep -iE 'type (nfs|lustre|gpfs|ceph|beegfs)'  # shared storage
```

### IO Benchmark (optional)
```bash
dd if=/dev/zero of=/tmp/topo_bench_tmp bs=1M count=1024 oflag=direct  # seq write
fio --name=topo_randread --rw=randread --bs=4k --size=256M ...  # random read (if fio available)
```

## Recommendations Generated

Based on detected topology, the skill generates recommendations:

### Compute Recommendations
- NVSwitch full mesh → "TP up to <gpu_count> is efficient within this node"
- NVLink-P2P partial → "TP groups should follow NVLink connectivity"
- PCIe-only → "TP across GPUs will be slow — prefer PP or DP"
- High GPU memory (≥80GB) → "Large model shards fit per GPU — may reduce TP/PP degree needed"

### Communication Recommendations
- Multiple IB NICs → "Aggregate bandwidth: <count> × <rate> = <total>"
- GPUDirect RDMA available → "Enable NCCL_NET_GDR_LEVEL=5 for lowest latency"
- NIC-GPU affinity detected → "Bind NCCL to NICs closest to each GPU group"
- No RDMA → "Inter-node communication will use TCP, expect lower bandwidth"

### Storage Recommendations
- NVMe local storage → "Fast local storage available for checkpoint staging"
- Shared storage detected → "Shared filesystem at <mount> — suitable for dataset and checkpoint storage"
- High IO bandwidth → "IO bandwidth sufficient for data loading"
- Low IO bandwidth or HDD → "Consider caching datasets to local NVMe"
- Disk usage > 80% → "Warning: <mount> is <pct>% full — ensure sufficient space for checkpoints"

## Multi-Node Topology Verification

For multi-node clusters:

1. **Run on each node separately**: Compare reports to verify homogeneous hardware. Flag heterogeneous clusters to user.

2. **Cross-node communication test**: Run NCCL tests to verify inter-node bandwidth:
   ```bash
   # Using nccl-tests (if available)
   mpirun -np <total_gpus> --hostfile <hostfile> \
     -x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=0 \
     all_reduce_perf -b 8 -e 2G -f 2 -g 1
   
   # Or using PyTorch's built-in NCCL
   torchrun --nproc_per_node=<gpus> --nnodes=2 \
     --master_addr=<node0_ip> --master_port=29500 \
     <allreduce_benchmark_script>
   ```

3. **Record cross-node bandwidth** in memory:
   - key: `topo_cross_node`, type: `finding`
   - Content: `nodes=<N> allreduce_bw_gbps=<B> nccl_version=<V>`

## When to Load Full Skill

Load the full `topo-detect` skill when:
- Running topology detection for the first time on a node
- Debugging multi-node communication issues
- Setting up NCCL environment variables based on topology
- Understanding detailed NIC-GPU affinity for optimal NCCL binding
- Running cross-node NCCL bandwidth tests
- Interpreting `nvidia-smi topo -m` output

This summary covers the essentials. For detailed detection steps, YAML structure, memory format, and multi-node verification procedures, load the full skill.
