../ClickHouse/build/programs/clickhouse-format -n --oneline < ./templates/table_create.sql > ./templates/table_create_processed.sql
../ClickHouse/build/programs/clickhouse-client < ./templates/table_create_processed.sql

