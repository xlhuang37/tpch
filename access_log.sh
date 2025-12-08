../ClickHouse/build/programs/clickhouse-format --oneline < ./sql/access_log.sql > ./sql/access_log_processed.sql
../ClickHouse/ClickHouse/build/programs/clickhouse-client < ./access_log_processed.sql