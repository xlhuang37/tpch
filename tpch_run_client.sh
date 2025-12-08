RUN_ID="bench.$(date +%Y%m%d-%H%M%S)"   # e.g. bench.20250924-231530
echo "$RUN_ID"

../ClickHouse/build/programs/clickhouse-format -n --oneline < ./sql/tpch.sql > ./sql/tpch_processed.sql
../ClickHouse/build/programs/clickhouse-client < ./sql/tpch_processed.sql