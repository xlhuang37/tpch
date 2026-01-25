echo "deleting existing tables"
rm -rf ../runtime

if [ "$1" == "-s" ]; then
    # 1. Check if $2 is empty
    if [ -z "$2" ]; then
        echo "Error: -s requires a numeric value."
        exit 1
    fi

    # 2. Check if $2 is a number (using a Regular Expression)
    if [[ ! "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: '$2' is not a valid number."
        exit 1
    fi

    size=$2
    echo "Setting size to $size"
else 
    echo "Error: Invalid argument. Use -s <size> to specify the size of the database."
    exit 1
fi


git submodule update --init --recursive

../ClickHouse/build/programs/clickhouse-format -n --oneline < ./templates/table_create.sql > ./templates/table_create_processed.sql
../ClickHouse/build/programs/clickhouse-client < ./templates/table_create_processed.sql

cd ./tpch-kit/dbgen
make
./dbgen -s 

../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO nation FORMAT CSV" < ./tpch-kit/dbgen/nation.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO region FORMAT CSV" < ./tpch-kit/dbgen/region.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO part FORMAT CSV" < ./tpch-kit/dbgen/part.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO supplier FORMAT CSV" < ./tpch-kit/dbgen/supplier.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO partsupp FORMAT CSV" < ./tpch-kit/dbgen/partsupp.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO customer FORMAT CSV" < ./tpch-kit/dbgen/customer.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO orders FORMAT CSV" < ./tpch-kit/dbgen/orders.tbl
../ClickHouse/build/programs/clickhouse-client --format_csv_delimiter '|' --query "INSERT INTO lineitem FORMAT CSV" < ./tpch-kit/dbgen/lineitem.tbl
