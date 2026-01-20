SELECT
  *
FROM system.query_log
WHERE startsWith(query_id, '20250925-032151')
  AND type = 'QueryFinish';