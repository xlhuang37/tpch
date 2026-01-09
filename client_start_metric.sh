../ClickHouse/build-debug/programs/clickhouse-client --param_rounding=5 --param_seconds=30 --query "SELECT

    CAST(toStartOfInterval(event_time, toIntervalSecond({rounding:UInt32})), 'INT') AS t,

    avg(ProfileEvent_OSCPUVirtualTimeMicroseconds) / 1000000

FROM system.metric_log

WHERE (event_date >= toDate(now() - {seconds:UInt32})) AND (event_time >= (now() - {seconds:UInt32}))

GROUP BY t

ORDER BY t ASC WITH FILL STEP {rounding:UInt32}"
