INSERT INTO nation SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/nation.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO region SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/region.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO part SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/part.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO supplier SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/supplier.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO partsupp SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/partsupp.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO customer SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/customer.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO orders SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/orders.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO lineitem SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/lineitem.tbl.gz', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

