../ClickHouse/build/programs/clickhouse-format -n --oneline < ./sql/table_create.sql > ./sql/table_create_processed.sql
../ClickHouse/build/programs/clickhouse-client < table_create_processed.sql