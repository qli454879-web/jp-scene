#!/bin/bash
# AI 加工 + MP3 生成 同时跑
cd "$(dirname "$0")"
echo "=== 并行启动 ==="
BATCH_ENRICH_MAX=200000 python3 batch_enrich.py &
PID_ENRICH=$!
MP3_CONCURRENCY=25 python3 generate_mp3.py &
PID_MP3=$!
wait $PID_ENRICH
wait $PID_MP3
echo "=== 全部完成 ==="
