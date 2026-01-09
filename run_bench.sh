../ClickHouse/build-debug/programs/clickhouse-format -n --oneline < ./sql/tpch.sql > ./sql/tpch_processed.sql
../ClickHouse/build-debug/programs/clickhouse-client < ./sql/tpch_processed.sql