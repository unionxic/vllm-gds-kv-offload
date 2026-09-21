#!/bin/bash
# PCIe·NIC 상태 기록(읽기 전용). usage: pcie_probe.sh <outdir>
O=${1:?outdir}; mkdir -p "$O"; F=$O/pcie_probe_$(hostname).txt
{
echo "### $(hostname) $(date -Is) kernel $(uname -r)"; echo "cmdline: $(cat /proc/cmdline)"
echo "### lspci -tv"; lspci -tv
for d in $(lspci -D | grep -iE 'nvidia|mellanox|non-volatile|root port|PCI bridge' | awk '{print $1}'); do
  echo "### $d $(lspci -s $d | cut -d' ' -f2-)"; sudo lspci -s $d -vvv | grep -E 'DevCap:|DevCtl:|MaxPayload|MaxReadReq|LnkCap:|LnkSta:|LnkCtl:|ACSCap|ACSCtl|CESta|UESta|Region'
done
echo "### nvidia-smi -q -d PCIE"; nvidia-smi -q -d PCIE
echo "### nvidia-smi topo -m"; nvidia-smi topo -m
echo "### IOMMU groups: $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)"; dmesg 2>/dev/null | grep -iE 'iommu|DMAR' | head -20
echo "### RDMA"; ibv_devinfo -v 2>/dev/null | grep -E 'hca_id|fw_ver|state|active_mtu|max_mtu|active_width|active_speed|link_layer|max_qp:|max_mr_size|max_sge:'
IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}'); echo "### NIC $IF"; ethtool $IF | grep -E 'Speed|Duplex'; ip link show $IF | grep -oE 'mtu [0-9]+'; ethtool -a $IF; ethtool -c $IF | grep -E 'Adaptive|rx-usecs:|tx-usecs:'; ethtool -g $IF | grep -A4 Current; ethtool -i $IF | grep -E 'driver|version|firmware'
echo "### PFC/QoS"; sudo mlnx_qos -i $IF 2>&1 | head -40
echo "### ECN"; for f in /sys/class/net/$IF/ecn/roce_np/enable/* /sys/class/net/$IF/ecn/roce_rp/enable/*; do [ -f $f ] && echo "$f=$(cat $f)"; done
echo "### ethtool -S nonzero (pause/discard/err/oob)"; ethtool -S $IF | grep -E 'pause|discard|out_of_buffer|ecn|cnp|drop|err' | grep -vE ': 0$'
echo "### mlx5 hw_counters"; for h in /sys/class/infiniband/mlx5_*; do p=$h/ports/1/hw_counters; [ -d $p ] || continue; echo "-- $(basename $h)"; for f in $p/*; do v=$(cat $f 2>/dev/null); [ "$v" != 0 ] && echo "$(basename $f)=$v"; done; done
echo "### NVMe-oF"; for c in /sys/class/nvme/nvme*; do t=$(cat $c/transport 2>/dev/null); [ "$t" = rdma ] && echo "$(basename $c): $(cat $c/subsysnqn) queue_count=$(cat $c/queue_count) sqsize=$(cat $c/sqsize)"; done
for p in /sys/kernel/config/nvmet/ports/*; do [ -d $p ] && echo "nvmet port $(basename $p): $(sudo cat $p/addr_traddr):$(sudo cat $p/addr_trsvcid) inline_data_size=$(sudo cat $p/param_inline_data_size)"; done
} > $F 2>&1; echo "wrote $F"
