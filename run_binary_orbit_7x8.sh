#!/usr/bin/env bash
set -euo pipefail

parent=${RECT78_PARENT:-/tmp/rect7x7-full.orbits}
workdir=${RECT78_WORKDIR:-/tmp/rect7x8-build}
workers=${RECT78_WORKERS:-8}
buckets=${RECT78_BUCKETS:-32}
parent_count=33642660

if ((workers < 1 || buckets < 1 || buckets > 999)); then
    echo "RECT78_WORKERS and RECT78_BUCKETS must be positive (buckets <= 999)" >&2
    exit 2
fi
if [[ ! -s "$parent" ]]; then
    echo "Missing checked 7x7 parent corpus: $parent" >&2
    exit 1
fi

mkdir -p "$workdir/local" "$workdir/reduced" "$workdir/logs" "$workdir/done"
make LTO=0 binary_orbit_augment_7x8

pids=()
for ((shard = 0; shard < workers; shard++)); do
    tag=$(printf '%03d' "$shard")
    if [[ -f "$workdir/done/extend.$tag" ]]; then
        continue
    fi
    start=$((parent_count * shard / workers))
    end=$((parent_count * (shard + 1) / workers))
    (
        ./binary_orbit_augment_7x8 extend "$parent" "$start" "$end" "$buckets" \
            "$workdir/local/s$tag" >"$workdir/logs/extend.$tag.log" 2>&1
        touch "$workdir/done/extend.$tag"
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
active=0
for ((bucket = 0; bucket < buckets; bucket++)); do
    tag=$(printf '%03d' "$bucket")
    if [[ -f "$workdir/done/reduce.$tag" ]]; then
        continue
    fi
    inputs=()
    for ((shard = 0; shard < workers; shard++)); do
        shard_tag=$(printf '%03d' "$shard")
        inputs+=("$workdir/local/s$shard_tag.b$tag")
    done
    (
        ./binary_orbit_augment_7x8 reduce 8 "$workdir/reduced/b$tag.orbits" \
            "${inputs[@]}" >"$workdir/logs/reduce.$tag.log" 2>&1
        touch "$workdir/done/reduce.$tag"
    ) &
    pids+=("$!")
    active=$((active + 1))
    if ((active >= workers)); then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
        active=$((active - 1))
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

reduced=()
for ((bucket = 0; bucket < buckets; bucket++)); do
    tag=$(printf '%03d' "$bucket")
    reduced+=("$workdir/reduced/b$tag.orbits")
done
output="$workdir/rect7x8-full.orbits"
./binary_orbit_augment_7x8 combine "$output" "${reduced[@]}" \
    >"$workdir/logs/combine.log" 2>&1
./binary_orbit_augment_7x8 check "$output" | tee "$workdir/logs/check.log"
sha256sum "$output" | tee "$workdir/logs/sha256.log"
