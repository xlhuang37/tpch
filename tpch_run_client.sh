RUN_ID="bench.$(date +%Y%m%d-%H%M%S)"   # e.g. bench.20250924-231530
echo "$RUN_ID"

../ClickHouse/build-debug/programs/clickhouse-format -n --oneline < ./sql/tpch.sql > ./sql/tpch_processed.sql
../ClickHouse/build-debug/programs/clickhouse-client < ./sql/tpch_processed.sql