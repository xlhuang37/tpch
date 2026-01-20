RUN_ID="bench.$(date +%Y%m%d-%H%M%S)"   # e.g. bench.20250924-231530
echo "$RUN_ID"

../ClickHouse/build-debug/programs/clickhouse-format -n --oneline < ./sql/np_default_60.sql > ./sql/np_default_60_processed.sql
../ClickHouse/build-debug/programs/clickhouse-benchmark < ./sql/np_default_60_processed.sql -c 4