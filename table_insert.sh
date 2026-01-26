../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO nation FORMAT CSV" < ./tpch-kit/dbgen/nation.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO region FORMAT CSV" < ./tpch-kit/dbgen/region.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO part FORMAT CSV" < ./tpch-kit/dbgen/part.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO supplier FORMAT CSV" < ./tpch-kit/dbgen/supplier.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO partsupp FORMAT CSV" < ./tpch-kit/dbgen/partsupp.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO customer FORMAT CSV" < ./tpch-kit/dbgen/customer.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO orders FORMAT CSV" < ./tpch-kit/dbgen/orders.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO lineitem FORMAT CSV" < ./tpch-kit/dbgen/lineitem.tbl
