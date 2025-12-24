#!/bin/bash

unset SLURM_JOB_ID
unset SLURM_JOBID

echo "################################################################################"
echo "### LOGIN NODE RAW DATA"
echo "################################################################################"
echo "--- HOSTNAME ---"
hostname
echo -e "\n--- KERNEL ---"
uname -a
echo -e "\n--- CPU FULL DETAILS (lscpu) ---"
lscpu
echo -e "\n--- CPU CACHE INDEX (Detailed) ---"
ls -F /sys/devices/system/cpu/cpu0/cache/
echo -e "\n--- NUMA TOPOLOGY (numactl) ---"
numactl -H
echo -e "\n--- MEMORY INFO (/proc/meminfo) ---"
cat /proc/meminfo
echo -e "\n--- PCI DEVICES (Verbose for NVIDIA/Mellanox) ---"
lspci -vv -d 10de:
lspci -vv -d 15b3:
echo -e "\n--- INFINIBAND STATUS ---"
ibstat || echo "ibstat not found"
ibv_devinfo || echo "ibv_devinfo not found"


echo -e "\n\n"
echo "################################################################################"
echo "### COMPUTE NODE RAW DATA (Partition: cpu)"
echo "################################################################################"

srun --partition=cpu --nodes=1 --ntasks=1 --time=00:05:00 --exclusive bash -c "
echo '--- HOSTNAME ---'
hostname
echo -e '\n--- CPU FULL DETAILS ---'
lscpu
echo -e '\n--- CORE DISTRIBUTION (lscpu -e) ---'
lscpu -e
echo -e '\n--- NUMA TOPOLOGY ---'
numactl -H
echo -e '\n--- MEMORY INFO ---'
cat /proc/meminfo
echo -e '\n--- INFINIBAND DEVICE DETAILS ---'
ibv_devinfo -v
echo -e '\n--- PCI BUS TREE (lspci -t) ---'
lspci -t
"


echo -e "\n\n"
echo "################################################################################"
echo "### GPU NODE RAW DATA (Partition: gpu)"
echo "################################################################################"

srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:05:00 --exclusive bash -c "
echo '--- HOSTNAME ---'
hostname
echo -e '\n--- HOST CPU DETAILS ---'
lscpu
echo -e '\n--- HOST MEMORY ---'
cat /proc/meminfo | grep Mem
echo -e '\n--- NVIDIA-SMI FULL QUERY (ALL DETAILS) ---'
nvidia-smi -q
echo -e '\n--- GPU TOPOLOGY MAP ---'
nvidia-smi topo -m
echo -e '\n--- INFINIBAND DEVICE DETAILS ---'
ibv_devinfo -v
"

