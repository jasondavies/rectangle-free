#!/usr/bin/env bash
set -euo pipefail

campaign_dir=${RECT7_WORKDIR:-./twocolour_7x7_work}
workers=${RECT7_WORKERS:-$(nproc)}
buckets=${RECT7_BUCKETS:-16}

if ((workers < 1 || buckets < 1 || buckets > 999)); then
    echo "RECT7_WORKERS and RECT7_BUCKETS must be positive (buckets <= 999)" >&2
    exit 2
fi

mkdir -p "$campaign_dir/buckets" "$campaign_dir/results" "$campaign_dir/logs"
make binary_orbit_augment twocolour_7x7_solve

full_orbits="$campaign_dir/7x7.orbits"
if [[ ! -s "$full_orbits" ]]; then
    ./binary_orbit_augment build 7 "$full_orbits" \
        >"$campaign_dir/logs/augment.log" 2>&1
fi

last_bucket=$(printf '%03d' $((buckets - 1)))
if [[ ! -s "$campaign_dir/buckets/orbits.b000.orbits" ||
      ! -s "$campaign_dir/buckets/orbits.b${last_bucket}.orbits" ]]; then
    ./binary_orbit_augment partition "$full_orbits" "$buckets" \
        "$campaign_dir/buckets/orbits" \
        >"$campaign_dir/logs/partition.log" 2>&1
fi

active=0
for ((bucket = 0; bucket < buckets; bucket++)); do
    tag=$(printf '%03d' "$bucket")
    result="$campaign_dir/results/result.${tag}.txt"
    if [[ -s "$result" ]] && grep -q '^end$' "$result"; then
        continue
    fi
    ./twocolour_7x7_solve solve "$result" \
        "$campaign_dir/buckets/orbits.b${tag}.orbits" \
        >"$campaign_dir/logs/solve.${tag}.log" 2>&1 &
    active=$((active + 1))
    if ((active >= workers)); then
        wait -n
        active=$((active - 1))
    fi
done
wait

results=("$campaign_dir"/results/result.*.txt)
if ((${#results[@]} != buckets)); then
    echo "Expected $buckets result files, found ${#results[@]}" >&2
    exit 1
fi
./twocolour_7x7_solve aggregate "${results[@]}"
