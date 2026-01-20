INSERT INTO nation SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/nation.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO region SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/region.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO part SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/part.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO supplier SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/supplier.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO partsupp SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/partsupp.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO customer SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/customer.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO orders SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/orders.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

INSERT INTO lineitem SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1 SELECT * FROM s3('https://clickhouse-datasets.s3.amazonaws.com/h/1/lineitem.tbl', NOSIGN, CSV) SETTINGS format_csv_delimiter = '|', input_format_defaults_for_omitted_fields = 1, input_format_csv_empty_as_default = 1;

