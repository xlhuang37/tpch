CREATE TABLE nation (
    n_nationkey  Int32,
    n_name       String,
    n_regionkey  Int32,
    n_comment    String)
ORDER BY (n_nationkey);

CREATE TABLE region (
    r_regionkey  Int32,
    r_name       String,
    r_comment    String)
ORDER BY (r_regionkey);

CREATE TABLE part (
    p_partkey     Int32,
    p_name        String,
    p_mfgr        String,
    p_brand       String,
    p_type        String,
    p_size        Int32,
    p_container   String,
    p_retailprice Decimal(15,2),
    p_comment     String)
ORDER BY (p_partkey);

CREATE TABLE supplier (
    s_suppkey     Int32,
    s_name        String,
    s_address     String,
    s_nationkey   Int32,
    s_phone       String,
    s_acctbal     Decimal(15,2),
    s_comment     String)
ORDER BY (s_suppkey);

CREATE TABLE partsupp (
    ps_partkey     Int32,
    ps_suppkey     Int32,
    ps_availqty    Int32,
    ps_supplycost  Decimal(15,2),
    ps_comment     String)
ORDER BY (ps_partkey, ps_suppkey);

CREATE TABLE customer (
    c_custkey     Int32,
    c_name        String,
    c_address     String,
    c_nationkey   Int32,
    c_phone       String,
    c_acctbal     Decimal(15,2),
    c_mktsegment  String,
    c_comment     String)
ORDER BY (c_custkey);

CREATE TABLE orders  (
    o_orderkey       Int32,
    o_custkey        Int32,
    o_orderstatus    String,
    o_totalprice     Decimal(15,2),
    o_orderdate      Date,
    o_orderpriority  String,
    o_clerk          String,
    o_shippriority   Int32,
    o_comment        String)
ORDER BY (o_orderkey);
-- The following is an alternative order key which is not compliant with the official TPC-H rules but recommended by sec. 4.5 in
-- "Quantifying TPC-H Choke Points and Their Optimizations":
-- ORDER BY (o_orderdate, o_orderkey);

CREATE TABLE lineitem (
    l_orderkey       Int32,
    l_partkey        Int32,
    l_suppkey        Int32,
    l_linenumber     Int32,
    l_quantity       Decimal(15,2),
    l_extendedprice  Decimal(15,2),
    l_discount       Decimal(15,2),
    l_tax            Decimal(15,2),
    l_returnflag     String,
    l_linestatus     String,
    l_shipdate       Date,
    l_commitdate     Date,
    l_receiptdate    Date,
    l_shipinstruct   String,
    l_shipmode       String,
    l_comment        String)
ORDER BY (l_orderkey, l_linenumber);