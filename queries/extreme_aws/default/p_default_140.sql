SELECT
    l_returnflag,
    l_linestatus,
    -- divide by 64 to keep output magnitudes similar to original Q1
    sum(l_quantity) / 64                           AS sum_qty,
    sum(l_extendedprice) / 64                      AS sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) / 64   AS sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) / 64 AS sum_charge,
    avg(l_quantity)                                AS avg_qty,
    avg(l_extendedprice)                           AS avg_price,
    avg(l_discount)                                AS avg_disc,
    count(*) / 64                                  AS count_order
FROM
(
    SELECT
        l_returnflag,
        l_linestatus,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
) AS l
CROSS JOIN
(
    SELECT number
    FROM system.numbers
    LIMIT 4
) AS dup
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;