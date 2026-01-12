SELECT
    l_returnflag,
    l_linestatus,
    avg(running_charge) AS avg_running_charge,
    count(*) AS cnt
FROM
(
    SELECT
        l_returnflag,
        l_linestatus,
        l_orderkey,
        sum(l_extendedprice * (1 - l_discount)) OVER
        (
            PARTITION BY l_orderkey
            ORDER BY l_linenumber
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_charge
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY 
) AS t
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus
SETTINGS
	workload='SpeedUpTwo';