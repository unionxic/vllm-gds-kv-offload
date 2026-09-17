#!/usr/bin/env bash
# 메모리 워치독: OS가 죽기 전에 run_combo_66b 프로세스를 먼저 죽인다.
#   트리거: MemAvailable < 2GiB, 또는 MemAvailable < 4GiB 이면서 PSI memory full avg10 > 30%
#   (참고: pinned 호스트 티어는 meminfo에 Shmem/Mapped/Cached로 잡히고 MemAvailable에서 제외됨 → avail은 믿을 만한 지표.
#    h0.85는 pinned 106GiB 확보 완료 시점에 avail 1.6GB → 프로세스 오버헤드 ~16GiB가 들어갈 자리가 없어 KILL이 정당했음)
#   샘플링 1초(h0.85에서 avail이 초당 ≈3GiB씩 떨어져 2초면 임계 아래 6GiB 오버슈트). 부수 기록: 10초마다 meminfo 스냅샷(pinned 호스트 티어가 Cached로 잡히는지 진단용)
# usage: memguard.sh <tag> <logfile>
tag=$1; log=$2; i=0
mi(){ awk -v k="$1" '$1==k":"{print int($2/1024)}' /proc/meminfo; }   # MiB
while true; do
  avail=$(mi MemAvailable); full=$(awk '/^full/{split($2,a,"=");print a[2]}' /proc/pressure/memory)
  some=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory)
  if (( i % 10 == 0 )); then
    echo "$(date +%H:%M:%S) $tag avail=${avail}M free=$(mi MemFree)M cached=$(mi Cached)M anon=$(mi AnonPages)M mapped=$(mi Mapped)M shmem=$(mi Shmem)M unevict=$(mi Unevictable)M slab=$(mi Slab)M psi_some=$some psi_full=$full" >> "$log"
  fi
  # avail<2GiB 단독, 또는 avail<4GiB이면서 psi_full>30(스래싱). 체크포인트 읽기 중 페이지캐시 회수만으로 psi_full 10%가 넘어 오탐(14:50)
  if (( avail < 2048 )) || { (( avail < 4096 )) && awk -v f="$full" 'BEGIN{exit !(f>30)}'; }; then
    pids=$(pgrep -f '^python run_combo_66b')
    echo "$(date +%H:%M:%S) $tag KILL avail=${avail}M psi_full=$full pids=$pids" >> "$log"
    [ -n "$pids" ] && kill -9 $pids
    sleep 5; continue
  fi
  i=$((i+1)); sleep 1
done
