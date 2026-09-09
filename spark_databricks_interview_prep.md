# Spark / Databricks Data Engineer Interview Prep
### 20+ Questions per Module — Scenario, Trap, Coding, Output, Internals, Follow-ups
*Target level: 5 YOE, FAANG/Databricks-style Data Engineer interviews*

---

## 1. Spark Architecture

**Scenario / Complex**
1. Walk through exactly what happens from `df.groupBy("col").count()` to a result appearing on the driver. *(Expect: logical plan → Catalyst optimization → physical plan → DAG → stages → tasks → executors → shuffle → result collected to driver.)*
2. Your cluster has 10 executors with 4 cores each, but the Spark UI shows only 20 tasks running concurrently at peak instead of 40. What could explain this?
3. You submit a job in `client` mode from a notebook vs `cluster` mode via a job scheduler — what actually differs operationally (driver location, network implications, failure behavior)?
4. A job fails with "Driver OOM" but never touches large data via `collect()`. What architectural components on the driver could still consume large memory (broadcast joins, large plans, accumulators)?
5. How would the failure of the **driver** vs the failure of **one executor** differently affect a running application?

**Trap / Conceptual**
6. Is the `SparkContext` per-application or per-executor? *(Trap: one SparkContext per application on the driver; executors don't have one.)*
7. Does increasing `executor cores` always increase parallelism proportionally? *(Trap: bounded by number of partitions and shuffle partitions, plus memory contention.)*
8. Is `SparkSession` the same as `SparkContext`? *(SparkSession wraps SparkContext + SQLContext + HiveContext since Spark 2.x.)*
9. Does the Cluster Manager (YARN/K8s/Standalone) execute your Spark code? *(No — it only allocates resources; the driver schedules tasks.)*
10. True or false: more executors always means faster jobs. *(False — shuffle/network overhead, small-file/partition overhead, and skew can dominate.)*

**Coding / Edge Cases**
11. Given `spark.conf.get("spark.executor.instances")` returns nothing on Databricks — why? (Autoscaling clusters don't fix instance count.)
12. Write code to inspect the number of executors currently alive at runtime via `SparkContext`. Edge case: value can change mid-job under autoscaling.
13. What happens if you call `.collect()` on a DataFrame larger than driver memory? Expected behavior: driver OOM / job killed.
14. What's the effect of setting `spark.driver.memory` too low on a job that only does narrow transformations plus one `.show(20)`? (Usually fine — show doesn't require large driver memory.)

**Expected Output / Execution Behavior**
15. If you run the same job twice with identical data and cluster config, will stage IDs and job IDs be identical? (No — IDs increment per session/application, not deterministic across runs.)
16. Does `spark.sparkContext.applicationId` change if the same notebook is re-run in the same session? (No, same app ID until session restarts.)

**Internals & Performance**
17. How does executor memory get divided internally (execution vs storage vs overhead vs user memory)?
18. Why does a cluster with fewer, larger executors sometimes outperform many small executors for shuffle-heavy jobs?
19. How does the driver's Catalyst-optimized physical plan get shipped to executors — what's actually serialized and sent?

**Follow-ups an interviewer may ask**
20. "You said stages run on executors — who decides how many tasks per stage?" (Task Scheduler, based on partition count.)
21. "If autoscaling adds executors mid-job, do running tasks get rebalanced?" (No — only future stages/tasks benefit.)
22. "What's the tradeoff between `client` and `cluster` deploy modes for production pipelines?"

---

## 2. DAG and Execution

**Scenario / Complex**
1. You have `df.filter(...).join(other_df).groupBy(...).count()`. Sketch the DAG and mark exactly where stage boundaries occur.
2. A job has 3 actions in sequence (`count()`, `show()`, `write.parquet()`) on the same DataFrame without caching. How many separate DAGs/jobs are triggered, and what's the performance implication?
3. Spark UI shows a job with 1 job → 4 stages, but you only wrote one wide transformation. Why 4 stages, not 2?
4. Your job has a stage stuck at "still running" with 1 task remaining for 40 minutes while others finished in 2 minutes — what's your diagnostic path using the DAG/stage view?
5. Explain how Spark decides task count for a stage reading from a Parquet table with 1000 small files vs one with 4 huge files.

**Trap / Conceptual**
6. Does every wide transformation create a new stage, or only actions? (Wide transformations create *stage boundaries* within a job; actions trigger *jobs*.)
7. True/false: a `map()` followed by a `filter()` always creates two separate tasks. (False — narrow transformations get pipelined into a single task via whole-stage codegen.)
8. Does calling `.explain()` execute the job? (No — it only shows the plan, no action triggered.)
9. Is the DAG built lazily or eagerly? Is it rebuilt for every action? (Lazily; a new DAG is generated per action unless cached.)
10. What causes Spark to create a new stage? *(Answer: primarily shuffle dependencies — wide dependencies — not merely presence of transformations/UDFs.)*

**Coding / Edge Cases**
11. Given:
```python
df1 = df.filter(df.x > 10)
df2 = df1.select("a", "b")
df2.count()
df2.show()
```
How many jobs total, and is `df1`'s filter recomputed for `show()`? (2 jobs, filter recomputed both times — no caching.)
12. Edge case: what happens to the DAG if an exception occurs mid-transformation chain before any action is called? (Nothing executes; error surfaces only if it's a plan-analysis error, e.g., unresolved column, since analysis happens eagerly on unresolved plans in newer Spark versions.)
13. Write code to visualize a job's stages programmatically via `SparkListener` (conceptually) or via `df.explain()` + Spark UI.

**Expected Output / Execution Behavior**
14. Does `df.rdd.getNumPartitions()` trigger a job? (No, it's metadata, no execution.)
15. If a stage has 200 tasks but the cluster has only 50 cores, how are tasks scheduled? (Queued in waves of 50 until all complete.)

**Internals & Performance**
16. What is the relationship between DAG Scheduler and Task Scheduler? (DAG Scheduler splits job into stages of tasks based on dependencies; Task Scheduler assigns tasks to executors and handles retries.)
17. Why can a single slow task delay an entire stage (and downstream stages)? (Stage completion barrier — next stage can't start until current stage's shuffle output is fully written.)
18. How does DAG-level pipelining reduce the need for materializing intermediate data between narrow transformations?

**Follow-ups**
19. "How would AQE change this DAG at runtime?" (Coalescing shuffle partitions, replanning joins after seeing actual stats.)
20. "If you cache `df1`, how does the DAG change for the second action?" (Second job's DAG starts from the cached stage, skipping recomputation.)
21. "Can two stages run in parallel?" (Yes, if they're independent branches feeding into a later join, until dependency requires sync.)

---

## 3. Lazy Evaluation

**Scenario / Complex**
1. You build a chain of 15 transformations and then call one `.count()`. An intermediate transformation references a column that doesn't exist. When does the error surface — at definition time or at `.count()`? (Typically at definition/analysis time in modern Spark due to eager analysis, not at action time — a classic trap.)
2. A teammate says "lazy evaluation means nothing happens until you call `.show()`, so it's safe to put expensive Python logic inside `.filter()` lambdas without perf concerns." Debunk this.
3. You read a huge Parquet dataset, apply 5 filters, then call `.count()` three times in a row without caching. What's the actual cost, and how would you fix it?
4. Explain why `df.filter(cond1).filter(cond2)` might produce a different physical plan than `df.filter(cond1 & cond2)` even though logically equivalent — are they actually different after optimization?

**Trap / Conceptual**
5. Does `df.printSchema()` trigger a Spark job? (No — schema is known from the logical plan / catalog metadata.)
6. Does `.cache()` trigger execution? (No, it's still lazy — marks the DataFrame for caching; materializes on first action.)
7. Is reading a CSV file with `inferSchema=True` lazy? (Trap: No — schema inference requires reading data, so it partially executes eagerly, unlike Parquet whose schema is embedded in metadata.)
8. True/false: transformations always return a new DataFrame without mutating original data. (True — Spark DataFrames are immutable.)
9. If you define transformations but never call an action, does anything get sent to executors at all? (No — nothing runs on executors; only the driver builds the plan.)

**Coding / Edge Cases**
10. 
```python
df = spark.read.parquet("path")
df = df.withColumn("y", df.x / 0)
df.show()
```
Does this throw at line 2 or at `.show()`? (Division errors are runtime/data-dependent — thrown at `.show()` when the task executes, unless it's a static analysis catch like divide-by-literal-zero constant folding producing null/Infinity depending on type — for doubles, division by zero yields `Infinity`/`NaN`, not exception.)
11. Edge case: what happens when a lazy transformation references an external Python variable that changes value before the action runs? (Value is captured based on closure semantics/serialization; explain the risk of relying on mutable external state.)

**Expected Output**
12. Does calling `df.columns` force execution? (No.)
13. Does `df.count()` after several transformations reflect the transformations, or run the original untransformed source? (Reflects transformations — full lineage executes.)

**Internals & Performance**
14. Why does Spark use lazy evaluation? *(Optimization, combining/fusing transformations, avoiding unnecessary computation, enabling Catalyst to see the whole plan before running anything, better execution planning.)*
15. How does lazy evaluation enable predicate pushdown that wouldn't be possible with eager execution?
16. How does lineage (built via lazy evaluation) support fault tolerance?

**Follow-ups**
17. "If lazy evaluation is so beneficial, why does eager execution exist in Pandas at all — what's the tradeoff?" (Debuggability/interactivity vs whole-plan optimization at scale.)
18. "How would you force partial materialization mid-pipeline to debug an issue without a full action?" (`.limit(n).collect()`, checkpointing, or caching an intermediate stage.)
19. "Does lazy evaluation change with Structured Streaming?" (Streaming still builds a logical plan, but execution is trigger-driven per micro-batch.)
20. "Give an example where lazy evaluation caused a *bug* in production, not just a performance issue." (e.g., side-effecting code, like writing to a DB inside a `map`, executing more or fewer times than expected due to re-execution/retries.)

---

## 4. Transformations (Narrow vs Wide)

**Scenario / Complex**
1. Is `df.select("a", explode("arr_col"))` narrow or wide? (Narrow — explode operates row-locally producing multiple output rows per input row, no shuffle needed.)
2. Is `df.repartition(200)` narrow or wide? Is `df.coalesce(50)` narrow or wide? (repartition = wide/full shuffle; coalesce = narrow, avoids shuffle by merging existing partitions.)
3. Is a `join` always wide? Give a scenario where a join is narrow. (Broadcast join avoids shuffling the large side — effectively narrow for that side.)
4. `df.groupBy("key").agg(...)` followed immediately by `.filter(...)` on the aggregated result — where's the stage boundary, before or after the filter?
5. You replace `.distinct()` with `.dropDuplicates(["id"])` — does this change whether a shuffle happens? (No — both require a shuffle to group identical/keyed rows together, though dropDuplicates on a subset can sometimes be cheaper depending on plan.)

**Trap / Conceptual**
6. Is `orderBy` narrow or wide? (Wide — requires global ordering via range partitioning/shuffle.)
7. True/false: `withColumn` always causes a shuffle if the new column depends on an aggregate. (True if it uses a window function or subquery aggregate without partitioning info causing a full shuffle — depends on implementation.)
8. Does `union()` require a shuffle? (No — narrow, just concatenates partitions unless followed by operations needing shuffle.)
9. Is `sample()` narrow or wide? (Narrow — operates per-partition.)
10. Does adding more narrow transformations to a pipeline increase the number of stages? (No — they get pipelined/fused into existing stages via whole-stage codegen.)

**Coding / Edge Cases**
11. 
```python
df2 = df.repartition(10, "key").groupBy("key").count()
```
Does the subsequent `groupBy` trigger another shuffle, or reuse partitioning from `repartition`? (Typically Spark can be smart via `df.repartition(n, col)` + matching shuffle partitions but historically still triggers its own shuffle unless plan detects compatible partitioning — worth checking `explain()` in practice.)
12. Edge case: `df.coalesce(1000)` when df currently has 10 partitions — what actually happens? (Coalesce can't increase partitions without a shuffle; it silently no-ops or stays at ≤10 partitions.)

**Expected Output**
13. `df.explain()` output shows an `Exchange` node — what does that tell you? (A shuffle boundary/wide dependency is present.)
14. Given a plan with no `Exchange` nodes at all, how many stages will the job have? (One stage.)

**Internals & Performance**
15. Why do wide transformations require data redistribution across executors while narrow ones don't? (Because output partitioning depends on a key that may not align with existing partitioning — records with the same key can live on any partition.)
16. How does understanding narrow vs wide help you optimize a slow pipeline? (Minimizing/reordering wide ops, filtering before wide ops, using broadcast to avoid a shuffle.)
17. Why is reordering `filter` before a `join` (predicate pushdown) valuable specifically for wide operations?

**Follow-ups**
18. "Can Catalyst reorder narrow and wide operations automatically?" (Yes, to some extent — predicate/projection pushdown, but not all reordering is safe or automatic.)
19. "If two consecutive wide transformations use the same partitioning key, does Spark shuffle twice?" (Potentially avoidable/reduced with AQE and plan awareness, but historically often shuffles both — a key case for `explain()` verification.)
20. "How would data skew differently affect a narrow vs wide transformation stage?" (Narrow — no impact from key skew since no key-based redistribution; wide — skew concentrates load on specific tasks.)

---

## 5. Shuffle

**Scenario / Complex**
1. Your `groupBy` job runs fine at 10M rows but times out at 1B rows with excessive disk spill. Walk through your diagnosis using shuffle read/write metrics in Spark UI.
2. Two jobs process the same data volume; Job A has `spark.sql.shuffle.partitions=200`, Job B has it at `4`. Job B is dramatically slower. Why? (Too few partitions → huge partitions per task → spill/OOM/serialization overhead, poor parallelism.)
3. You see huge "shuffle write" but tiny "shuffle read" in the Spark UI for a stage. What could cause this mismatch? (Could indicate downstream stage failed/wasn't reached yet, or heavy filtering after shuffle read reduced apparent read size — needs careful interpretation, e.g., read metrics reported per completed task.)
4. Explain shuffle from network, disk, and CPU (serialization) perspectives simultaneously — which resource typically bottlenecks first for wide-and-skewed joins?
5. A job shuffles 500GB of data across a 20-node cluster. Sketch the map-side and reduce-side steps and where shuffle files physically live.

**Trap / Conceptual**
6. Does more shuffle partitions always reduce spill? (Not necessarily — too many creates scheduling/overhead and tiny-task inefficiency; must balance.)
7. True/false: shuffle always writes to HDFS/cloud storage. (False — typically writes to local executor disk, not distributed storage, unless using external shuffle service configurations.)
8. Is shuffle spill the same as shuffle write? (No — shuffle write is the normal shuffle output; spill is writing intermediate data to disk due to memory pressure during a sort/aggregation, an additional cost.)
9. Does caching a DataFrame before a `groupBy` eliminate the shuffle? (No — caching avoids recomputation of prior stages, not the shuffle needed for the aggregation itself.)

**Coding / Edge Cases**
10. How would you inspect shuffle partition sizes to detect skew programmatically? (`df.rdd.glom().map(len).collect()` post-shuffle, or Spark UI per-task metrics.)
11. Edge case: setting `spark.sql.shuffle.partitions` to 1 for a `groupBy` on 100GB — what breaks? (Single task must hold ~100GB → executor OOM/severe spill, essentially serializes the job.)

**Expected Output**
12. In `df.explain("formatted")`, what physical operator indicates shuffle write/read boundaries? (`Exchange`.)
13. Does shuffle read size ever exceed shuffle write size for the same stage boundary? (Generally shouldn't in simple cases — read should roughly equal write across all tasks, barring compression/serialization differences reported.)

**Internals & Performance**
14. Explain map-side vs reduce-side shuffle work precisely — what happens on the writing side vs receiving side?
15. How does serialization format (Java vs Kryo) affect shuffle performance? (Kryo is more compact/faster, reducing shuffle write/read and network time.)
16. What's the role of the external shuffle service, and why does it matter for dynamic allocation? (Allows executors to be removed without losing shuffle data other executors still need to fetch.)
17. How does data skew manifest specifically in shuffle read metrics? (One task reading vastly more shuffle data than others.)

**Follow-ups**
18. "How would AQE help mitigate the shuffle skew you just described?" (Skew join optimization splits skewed partitions into multiple tasks.)
19. "What's the cost of shuffle in terms of network and how would you reduce it architecturally (not just via config)?" (Broadcast joins, pre-aggregation/combiners, reducing data volume via filter/select pushdown, salting.)
20. "How does shuffle interact with dynamic resource allocation removing idle executors?" (Executors holding shuffle files can't be removed unless external shuffle service is enabled.)

---

## 6. Spark Partitions

**Scenario / Complex**
1. You write `df.repartition(10)` then `.write.partitionBy("date")`. How many actual output files could you end up with, and why might it be far more than 10? (Each of the 10 in-memory partitions can contain multiple distinct `date` values, producing up to 10 × distinct_dates files.)
2. Your input has 100,000 tiny files (~10KB each) on S3. What partitioning problems does this cause and how do you fix it? (Excess task overhead/small-file problem; fix via `coalesce`, larger file writes upstream, or `maxPartitionBytes` tuning, or compaction/OPTIMIZE.)
3. Explain the difference in outcome between `df.repartition(10)` and `df.repartition(10, "customer_id")`. (First is round-robin/random redistribution; second hash-partitions by key, so same key always lands in the same partition.)
4. Your table has 2000 Spark partitions in memory but you call `.coalesce(1)` before writing. What happens to parallelism during that final write stage? (Forced down to 1 task — writing becomes single-threaded/bottleneck, defeating prior parallelism.)
5. Why might `spark.sql.shuffle.partitions=200` (the default) be wrong for both a 10GB job and a 10TB job simultaneously running on the same cluster config?

**Trap / Conceptual**
6. Is `repartition()` a narrow or wide operation, and does it guarantee equal-sized partitions? (Wide/full shuffle; round-robin repartition roughly balances size, but hash-based repartition by skewed key does not.)
7. Does `coalesce()` guarantee reduced partitions exactly to N? (Trap: it can only decrease, and actual count may be ≤N depending on data/parallelism; also doesn't rebalance data — can create skewed partitions if source was skewed.)
8. Is Spark partitioning the same as Hive/table partitioning (`partitionBy` on write)? (No — Spark partitions are in-memory/execution-time splits; table partitioning is a physical directory/file-organization concept on storage.)
9. True/false: increasing `spark.sql.shuffle.partitions` always improves performance. (False — diminishing returns and overhead beyond a point.)

**Coding / Edge Cases**
10. 
```python
df.repartition(10).write.partitionBy("date").parquet(path)
```
vs
```python
df.repartition(10, "date").write.partitionBy("date").parquet(path)
```
Which produces fewer output files per date partition, and why?
11. Edge case: calling `.coalesce(100)` on a DataFrame that only has 10 partitions. (No-op — can't increase via coalesce; stays at 10.)

**Expected Output**
12. `df.rdd.getNumPartitions()` after a plain `.filter()` with no repartition — same as source or different? (Same as source — narrow transformation preserves partition count.)
13. After a `.groupBy().count()` with default config, how many output partitions does the result have? (Equal to `spark.sql.shuffle.partitions`, default 200, unless AQE coalesces them.)

**Internals & Performance**
14. How does `spark.sql.files.maxPartitionBytes` influence initial partition count when reading files? (Controls max bytes per input partition/split during scan planning.)
15. Why does partition sizing matter for both too-large and too-small partitions (memory pressure vs task scheduling overhead)?
16. How do input partitions differ from shuffle partitions differ from output partitions conceptually?

**Follow-ups**
17. "How would AQE's coalescing feature change your understanding of `spark.sql.shuffle.partitions` as a fixed setting?" (AQE can dynamically merge small post-shuffle partitions at runtime, reducing the need for precise manual tuning.)
18. "If you're writing to Delta and want ~128MB files, how do partition counts before `.write` relate to that goal?" (Tune repartition/coalesce count so `total_size / num_partitions ≈ target file size`, or use Delta's `OPTIMIZE`.)
19. "What's the danger of `repartition(1)` before a `.write()` on a huge dataset?" (Single task/file — no parallelism, huge file, OOM risk, slow write.)
20. "How do partition and bucketing differ, and when would you use bucketing instead of just partitioning?"

---

## 7. Catalyst Optimizer

**Scenario / Complex**
1. Walk through the full Catalyst pipeline for `SELECT name FROM t WHERE age > 30` from unresolved logical plan to execution.
2. You write `df.filter(df.age > 30).select("name")` vs `df.select("name").filter(df.age > 30)` (assume `age` still accessible). Do these produce different physical plans after optimization? (No — Catalyst normalizes via predicate/projection pushdown to the same optimized plan in most cases.)
3. A query has a filter on a partitioned column plus a filter on a non-partitioned column. Explain how Catalyst/physical planning uses each differently (partition pruning vs predicate pushdown at file level).
4. Why might a UDF in your `.filter()` prevent Catalyst from doing predicate pushdown that a native function would allow? (Catalyst can't introspect opaque UDF logic to push it into the scan/reorder safely.)
5. You have a self-join with a filter after the join. Show how Catalyst might push part of that filter down into one side before the join (if column availability allows).

**Trap / Conceptual**
6. Does Catalyst optimize DataFrame code but not Spark SQL string queries, or both? (Both — same underlying Catalyst/plan pipeline regardless of DataFrame API or SQL text.)
7. True/false: Catalyst can optimize away a UDF entirely if unused. (True for unused/dead columns via projection pruning, but it can't optimize *inside* an actually-used UDF's logic.)
8. Does predicate pushdown work the same on all file formats? (No — works fully for columnar formats like Parquet/ORC with statistics; limited/none for CSV/JSON since no embedded stats.)
9. Is "constant folding" a runtime or plan-time optimization? (Plan-time — evaluated during logical plan optimization before execution.)

**Coding / Edge Cases**
10. Run `df.filter("1 = 1").filter(df.age > 30).explain()` — what happens to the trivially true `1=1` filter? (Eliminated via boolean simplification/constant folding.)
11. Edge case: `df.select("*").filter(col("a") == col("a"))` — does Catalyst simplify this to always-true and potentially drop the filter node? (Often yes, depending on nullability assumptions — worth noting null-safety nuance: if `a` can be null, `null == null` is `null`, not `true`, so Spark must retain the check unless it can prove non-nullability.)

**Expected Output**
12. Compare `df.explain()` output before/after adding an unnecessary `.select(df.columns)` — does the plan differ after optimization? (Typically optimized away — same optimized plan.)
13. In `.explain("formatted")`, which section shows the *optimized* logical plan vs the physical plan?

**Internals & Performance**
14. List and explain 4 key Catalyst optimizations relevant to a typical ETL job (predicate pushdown, projection pruning, constant folding, join reordering/optimization).
15. How does partition pruning differ from predicate pushdown, and where in the pipeline does each occur?
16. Why do native Spark SQL functions benefit more from Catalyst than Python UDFs?

**Follow-ups**
17. "If Catalyst already optimizes your plan, why do we still need to manually tune joins/broadcast hints?" (Cost-based decisions rely on statistics that may be stale/missing; hints override imperfect estimates.)
18. "How does Catalyst interact with AQE — is AQE part of Catalyst or separate?" (AQE re-invokes planning/optimization at runtime using actual stats — an extension layered on the same optimizer framework.)
19. "Can you force Catalyst to *not* optimize a query, and why would you ever want to?" (Rare — mostly for debugging; not a normal production need.)
20. "Explain a real production case where understanding Catalyst helped you fix a slow query." (Expect candidate to describe pushdown/pruning/UDF replacement anecdote.)


---

## 8. Physical Execution

**Scenario / Complex**
1. For `SELECT department, COUNT(*) FROM employees GROUP BY department`, walk through the physical operators (`FileScan → Project → Exchange → HashAggregate`) and explain why there are typically *two* `HashAggregate` nodes (partial + final).
2. You run `.explain("formatted")` and see `SortMergeJoin` where you expected `BroadcastHashJoin`. What would you check first? (Table size vs `spark.sql.autoBroadcastJoinThreshold`, stats availability, explicit broadcast hint missing, AQE not yet kicked in.)
3. A physical plan shows two `Exchange` nodes back-to-back with no operator between them. Is that expected, or a red flag for wasted shuffle? (Usually a red flag worth investigating — possibly separate wide ops that could be consolidated.)
4. Explain the difference between `HashAggregate` and `SortAggregate` and when Spark would pick one over the other.
5. You see `WholeStageCodegen` wrapping multiple operators in the physical plan. What does that tell you about how those operators execute together?

**Trap / Conceptual**
6. Does the physical plan always match the logical plan structurally? (No — physical planning can restructure significantly, e.g., partial aggregation splitting.)
7. True/false: `Project` in the physical plan always means columns are being added. (False — it also represents column selection/pruning, and can be optimized away entirely if a no-op.)
8. Is `FileScan` always the first operator? (Not necessarily if multiple sources are joined/unioned — multiple `FileScan` leaves possible.)
9. Does `.explain()` show you what the AQE-adjusted runtime plan will actually be? (No — default `explain()` shows the initial plan; use `explain(mode="formatted")` post-execution or `explain(extended=True)`/adaptive re-planning artifacts, or check Spark UI SQL tab for the *actual* executed plan with AQE.)

**Coding / Edge Cases**
10. Run `df.groupBy("dept").count().explain("formatted")` — identify partial vs final aggregation nodes in the output.
11. Edge case: a `GROUP BY` on a column with extremely high cardinality (e.g., UUID) — does Spark still do partial aggregation, and is it actually helpful here? (Partial aggregation happens but yields little benefit since almost no combining occurs — mostly overhead.)

**Expected Output**
12. What does an `Exchange hashpartitioning(dept#12, 200)` line in the physical plan literally mean? (Shuffling data into 200 partitions hash-partitioned by `dept`.)
13. If you see `BroadcastExchange` in the plan, what does that imply about task distribution for that side of the join? (That side's full data is sent to every executor rather than shuffled/partitioned.)

**Internals & Performance**
14. Why does Spark do partial aggregation before the shuffle in a `GROUP BY`? (Reduces the amount of data shuffled — combiner-like optimization.)
15. How do physical operator choices change based on available statistics (row count, size, column stats)? Where do these stats come from (`ANALYZE TABLE`, Delta stats, file metadata)?
16. Why is reading `.explain("formatted")` output considered "extremely valuable in interviews" and in real debugging?

**Follow-ups**
17. "How would you confirm which join strategy Spark *actually* used at runtime versus what the initial plan predicted?" (Spark UI SQL tab, post-execution plan with AQE metrics.)
18. "If stats are stale (table changed but not re-analyzed), what physical-plan mistakes could result?" (Wrong join strategy choice, e.g., not broadcasting a table that's now small, or broadcasting one that's grown too large → OOM.)
19. "What's the risk of blindly forcing `broadcast()` hints everywhere?" (Driver/executor OOM if the "small" table isn't actually small, or stale assumption after data growth.)

---

## 9. Whole-Stage Code Generation

**Scenario / Complex**
1. You have a pipeline using only built-in Spark SQL functions (filter, select, arithmetic) — Spark UI shows a single `WholeStageCodegen` block wrapping several operators. Explain what's happening physically at the JVM level.
2. You insert a single Python UDF in the middle of an otherwise all-native pipeline. How does this break/interrupt whole-stage codegen? (UDF execution requires falling out of generated JVM code into Python process boundary — breaking the fused codegen pipeline into separate stages/segments.)
3. Two functionally-equivalent pipelines — one built with native `when()/col()` expressions, one with an equivalent Python UDF — show very different codegen behavior in the physical plan. Explain why and what performance difference to expect.

**Trap / Conceptual**
4. Does whole-stage codegen eliminate shuffles? (No — it only optimizes the CPU-bound, per-row processing within pipelined narrow operators; shuffle boundaries remain.)
5. True/false: every physical operator benefits from whole-stage codegen. (False — some operators, e.g., certain joins/sorts, aren't fully codegen-compatible and fall back to Volcano-style iterator execution.)
6. Is codegen generating Python bytecode or JVM bytecode? (JVM bytecode — compiled and JIT-executed, not Python.)

**Coding / Edge Cases**
7. In `.explain("formatted")` or the Spark UI's SQL tab, how do you identify which operators got fused under a single `WholeStageCodegen` node vs which broke out separately?
8. Edge case: extremely wide (many columns) DataFrames can sometimes hit codegen size limits ("Generated method too long"). What happens then? (Spark falls back to non-codegen/interpreted execution for that stage — a known JVM 64KB method-size limitation edge case.)

**Expected Output**
9. Given a plan with `WholeStageCodegen (1)` wrapping `Filter` and `Project`, and a separate `Exchange` node outside it — is `Exchange` part of the same codegen block? (No — shuffle boundaries are never part of codegen fusion.)

**Internals & Performance**
10. Why can native Spark functions outperform Python UDFs specifically because of whole-stage codegen (not just serialization)? (Native ops get compiled into a single tight JVM loop with no per-row abstraction overhead; UDFs force interpreted per-row calls plus, for Python, cross-process serialization.)
11. How does Tungsten's binary row format relate to what whole-stage codegen operates on? (Codegen operates directly on Tungsten's compact binary representation, avoiding JVM object overhead per row.)
12. What's the performance difference class (order of magnitude) typically observed between codegen-optimized native pipelines and interpreted/UDF-broken pipelines on large datasets?

**Follow-ups**
13. "If codegen is so much faster, why doesn't every single operator support it?" (Complexity of generating correct code for all operator variants; some fall back for correctness/maintainability.)
14. "How would you detect in production that codegen got disabled or bypassed for a critical stage?" (Compare expected vs actual stage CPU time; inspect physical plan for missing `WholeStageCodegen` wrapping around expected operators; check for UDF usage or codegen fallback warnings in logs.)
15. "Does DataFrame/SQL API always benefit from codegen more than RDD API? Why?" (Yes — RDDs operate on JVM objects without the structured schema Catalyst/Tungsten need to generate optimized code.)

---

## 10. Tungsten

**Scenario / Complex**
1. Two jobs process identical data — one using RDDs of Python objects, one using DataFrames. Explain, from a Tungsten perspective, why the DataFrame version uses less memory and CPU.
2. Your team debates moving legacy RDD-based jobs to DataFrame API purely for "modern syntax." What's the deeper Tungsten-related performance argument you'd make beyond syntax?

**Trap / Conceptual**
3. Is Tungsten a separate system from Spark SQL, or an execution engine component? (An execution engine optimization layer *within* Spark, not standalone.)
4. True/false: Tungsten's benefits apply equally to RDD and DataFrame APIs. (False — schema-aware binary format and codegen benefits are mainly realized through the structured DataFrame/SQL API.)
5. Does Tungsten eliminate garbage collection entirely? (No — reduces GC pressure significantly via off-heap/binary memory management, doesn't eliminate it entirely.)

**Coding / Edge Cases**
6. No direct "coding" here — instead: how would you empirically demonstrate Tungsten's benefit to a teammate using a benchmark? (Compare RDD `.map()` heavy job vs equivalent DataFrame job on same data/cluster, measure task GC time and duration in Spark UI.)

**Expected Output**
7. What would you look for in the Spark UI's "GC Time" column to spot a job *not* benefiting from Tungsten's memory management (e.g., heavy RDD/UDF usage)? (High GC time relative to task duration.)

**Internals & Performance**
8. List Tungsten's core focus areas (memory management, binary data representation, CPU efficiency, cache efficiency, code generation, reduced object overhead) and briefly explain each in one line.
9. Why did Spark introduce Tungsten? *(To move beyond JVM object overhead and GC pressure toward CPU/cache-efficient, off-heap binary processing — closing the gap with hand-optimized code.)*
10. How does Tungsten's binary row format improve CPU cache locality compared to JVM object graphs? (Compact contiguous memory layout vs scattered pointer-chasing JVM objects.)
11. How does off-heap memory management under Tungsten relate to `spark.memory.offHeap.enabled`?

**Follow-ups**
12. "Does Tungsten help Python UDF performance?" (No — UDF execution leaves the JVM/Tungsten-optimized path entirely.)
13. "How does Tungsten relate to Project Catalyst — are they the same project?" (Complementary but distinct: Catalyst optimizes the logical/physical *plan*; Tungsten optimizes low-level *execution* — memory/CPU.)
14. "If you inherited a legacy RDD pipeline, what's your migration argument to leadership grounded in Tungsten benefits?" (Reduced GC, better CPU cache use, codegen eligibility, likely 2-10x speedups depending on workload.)

---

## 11. Serialization

**Scenario / Complex**
1. A job with a Python UDF applied to 1 billion rows is significantly slower than an equivalent job using only native functions. Trace the exact serialization path causing this (JVM → Python → JVM per row/batch).
2. You switch `spark.serializer` from the default Java serializer to Kryo for RDD-heavy jobs and see a meaningful shuffle time improvement. Explain mechanistically why.
3. A closure captures a large lookup dictionary (50MB) from the driver. What serialization cost does this incur, and per what unit (per task? per executor?)? (Serialized and shipped potentially per task depending on caching of the closure — significant overhead; better solved via broadcast variable.)

**Trap / Conceptual**
4. Is Kryo used by default in Spark? (No — Java serialization is default; Kryo must be explicitly configured, though DataFrame/SQL internal encoders bypass generic serializers anyway via Tungsten encoding.)
5. True/false: DataFrame/Dataset operations rely on the same generic Java/Kryo serialization as RDDs. (False, mostly — structured APIs use Tungsten's specialized encoders for known schemas, far more efficient than generic object serialization.)
6. Does using Kryo serialization improve Python UDF performance? (No — Kryo is a JVM-side serializer; Python UDF overhead is a separate JVM↔Python boundary issue, unaffected by Kryo.)

**Coding / Edge Cases**
7. How do you register a custom class with Kryo to avoid the default (slower) reflection-based serialization fallback? (`spark.kryo.registrator` / `registerKryoClasses`.)
8. Edge case: a class that isn't Kryo-registered and isn't serializable at all — what error do you get and where (driver-side submission or executor-side task failure)? (Typically a `NotSerializableException` surfacing during task execution/serialization on the executor or during closure cleaning on the driver.)

**Expected Output**
9. If you inspect a Spark UI task's metrics, is serialization time reported separately from computation time? (Yes — "Task Deserialization Time" and result serialization are shown as distinct metrics.)

**Internals & Performance**
10. Compare Java serialization vs Kryo serialization on size, speed, and ease of use tradeoffs.
11. What exactly gets serialized when a Python UDF runs — describe the JVM → Python → JVM round trip in detail (pickling, socket/pipe transfer, py4j or Arrow-based transport).
12. Why is serialization overhead a first-class performance concern in distributed systems generally, not just Spark?

**Follow-ups**
13. "If DataFrames use Tungsten encoders instead of generic serialization, why does `spark.serializer=kryo` still matter in practice?" (Still relevant for RDD-based code, shuffle of certain internal structures, caching, and some non-DataFrame paths.)
14. "How would switching a UDF to a Pandas UDF change the serialization story?" (Batch-based Arrow serialization instead of row-by-row pickling — dramatically fewer round trips and more efficient columnar transfer.)
15. "What's a real scenario where closure serialization caused you a production issue?" (e.g., accidentally capturing a non-serializable driver-side object like a DB connection inside a `map` closure.)

---

## 12. Python UDF / Pandas UDF / Arrow

**Scenario / Complex**
1. A data engineer replaces a native `when/otherwise` expression with a Python UDF "for readability." Quantify (qualitatively) the expected performance regression and explain the root causes (serialization + loss of codegen + loss of Catalyst optimization insight).
2. You need to apply a complex ML model's `.predict()` per row. Compare three approaches: row-at-a-time Python UDF, Pandas UDF (vectorized), and a native Spark ML pipeline — tradeoffs of each.
3. A Pandas UDF works fine on small data but throws OOM on executors at scale. Why might vectorized Pandas UDFs still cause OOM despite being "faster"? (Each batch materializes as a pandas Series/DataFrame in executor memory — large batch sizes or wide data can still blow memory; batch size tuning matters.)
4. Your job barely improves after switching row UDFs to Pandas UDFs. What else might still be limiting throughput (e.g., Arrow not enabled, small `arrow.maxRecordsPerBatch`, or the UDF logic itself being the bottleneck, not the transport)?

**Trap / Conceptual**
5. Are Pandas UDFs guaranteed to be faster than Python UDFs in all cases? (Not guaranteed — for very lightweight per-row logic on small data, the vectorization overhead might not pay off; typically faster at scale though.)
6. True/false: Pandas UDFs completely eliminate the JVM↔Python boundary cost. (False — they reduce the *number* of round trips via batching with Arrow, not eliminate the boundary itself.)
7. Does enabling Arrow (`spark.sql.execution.arrow.pyspark.enabled`) automatically speed up all Python UDFs? (No — it mainly benefits `toPandas()`/`createDataFrame()` conversions and Pandas UDFs specifically, not plain row-at-a-time Python UDFs.)
8. Is a Scala UDF as slow as a Python UDF? (No — Scala UDFs run natively in the JVM with no cross-process serialization, though they still bypass some codegen/Catalyst-level optimization compared to fully native expressions.)

**Coding / Edge Cases**
9. Write a simple Pandas UDF (`@pandas_udf`) that doubles a numeric column, and contrast it with the row-based `udf()` equivalent.
10. Edge case: a Pandas UDF function returns a Series of a different length than the input batch. What happens? (Error — Pandas UDFs (Series-to-Series) must return the same length as input.)
11. Edge case: null handling — how do Python UDFs vs Pandas UDFs differ in default null propagation behavior? (Both generally require explicit null handling in your function logic; nulls become `None`/`NaN` and must be guarded against.)

**Expected Output**
12. In Spark UI, how would you identify that a stage is bottlenecked by Python UDF execution rather than JVM computation? (Look for `ArrowEvalPython` / `BatchEvalPython` operators in the physical plan and high executor CPU time attributed to Python worker processes.)

**Internals & Performance**
13. Why is Python UDF slower? *(Per-row serialization/deserialization across the JVM↔Python boundary, no codegen participation, no Catalyst-level optimization into the row, process-switch overhead.)*
14. What is Arrow, precisely, and why does it enable fast data exchange between JVM and Python? (A columnar in-memory format shared without per-row (de)serialization — zero/low-copy transfer of batches.)
15. Why are Pandas UDFs faster? (Batch/vectorized processing via Arrow reduces round-trip count and leverages vectorized pandas/numpy operations instead of per-row Python interpreter overhead.)
16. When would you use a Pandas UDF? (Complex per-row/per-group logic not expressible in native Spark SQL functions, especially where vectorized numpy/pandas/ML libraries can be leveraged — e.g., custom stats, ML inference.)
17. Why are built-in functions still preferred over Pandas UDFs when equivalent logic exists natively? (Full Catalyst optimization, whole-stage codegen eligibility, no cross-process overhead at all.)

**Follow-ups**
18. "If Pandas UDFs use Arrow, why not just always use Pandas UDFs instead of native functions by default?" (Native functions still outperform due to codegen/Catalyst integration; Pandas UDFs are a fallback for expressiveness, not a default choice.)
19. "How would you tune `spark.sql.execution.arrow.maxRecordsPerBatch` and why does it matter?" (Balances memory per batch (larger batches lower overhead but higher per-task memory) vs number of round trips.)
20. "Explain Grouped Map / Cogrouped Pandas UDFs — when would you reach for those over simple scalar Pandas UDFs?" (Group-wise custom transformations, e.g., per-group time-series modeling, that can't be expressed as simple column-wise scalar ops.)

---

## 13. Memory Management

**Scenario / Complex**
1. A `groupBy` aggregation that worked fine at 100GB starts spilling heavily to disk at 500GB on the same cluster. Walk through the unified memory model to explain what's happening and how you'd remediate (more executors, more memory per executor, better partitioning, pre-filtering).
2. An executor dies with OOM during a large shuffle-heavy join, but driver memory usage looks fine. Where specifically in the memory model did it likely fail (execution memory during shuffle/sort/aggregation buffers)?
3. You increase `spark.executor.memory` but the job still OOMs at the same point. What other memory-related settings/behaviors might still be the bottleneck (e.g., off-heap overhead, too few partitions concentrating data per task, memory fraction settings, or driver-side broadcast size)?
4. Explain unified memory management: how do execution and storage memory dynamically share space, and what happens when both compete under pressure? (Execution can evict storage/cached blocks when needed; storage cannot forcibly evict active execution memory — execution has priority under contention, subject to configured minimum storage guarantee.)

**Trap / Conceptual**
5. Does increasing `spark.executor.memory` always fix OOM errors? (Not necessarily — could be a data skew/partitioning problem that no amount of extra memory per executor fully solves for the single overloaded task.)
6. True/false: cached data and shuffle/aggregation buffers compete for the exact same memory pool. (True under Spark's unified memory model — both draw from the same unified region, dynamically balanced.)
7. Does driver OOM only happen from `.collect()`? (No — also from large broadcast joins, accumulators, or building huge query plans.)
8. Is "spill to disk" always a bug/misconfiguration? (Not necessarily — it's a safety mechanism; occasional modest spill is fine, but *heavy* spill indicates a real tuning problem.)

**Coding / Edge Cases**
9. How would you use Spark UI's "Executors" tab to distinguish an execution-memory-driven OOM from a storage-memory-driven OOM?
10. Edge case: caching a DataFrame that's larger than total cluster memory using `MEMORY_ONLY` — what happens to the fraction that doesn't fit? (Dropped/recomputed on demand — partitions that don't fit are simply not cached, recomputed via lineage when needed, potentially causing repeated expensive recomputation.)

**Expected Output**
11. In Spark UI's Stages tab, which columns specifically indicate memory pressure (Spill (Memory), Spill (Disk))? What does non-zero "Spill (Disk)" always imply? (That memory-resident intermediate data exceeded available execution memory and had to spill to disk.)

**Internals & Performance**
12. Explain execution memory vs storage memory vs "user memory" vs reserved memory within an executor's heap.
13. Why does `groupBy()` sometimes spill to disk? *(Aggregation/shuffle requires more memory than available, so Spark spills intermediate data to disk to avoid OOM.)*
14. How does `spark.memory.fraction` and `spark.memory.storageFraction` influence this unified pool split?
15. What's the practical impact of enabling off-heap memory (`spark.memory.offHeap.enabled`) on GC behavior and OOM risk?

**Follow-ups**
16. "If you can't add more cluster resources, what are your top 3 levers to fix an OOM/spill problem?" (Increase shuffle partitions to shrink per-task data, fix skew via salting/AQE, filter/select earlier to reduce data volume, avoid unnecessary caching.)
17. "How does memory pressure differ between a wide `join` OOM and a `groupBy` OOM in terms of root cause?" (Join memory pressure often from building hash tables (broadcast or shuffle hash join) vs aggregation memory pressure from sort/hash aggregation buffers — both ultimately execution memory, but different operator-level causes.)
18. "How would AQE's skew handling reduce OOM risk without adding cluster resources?" (Splits skewed partitions into smaller sub-tasks, reducing peak memory per task.)

---

## 14. Cache and Persist

**Scenario / Complex**
1. You cache a DataFrame with `df.cache()`, run one action, then modify the DataFrame's lineage upstream (e.g., re-reading source with new filters) — does the cached data reflect the new lineage automatically, or is it stale? (Cache is tied to the DataFrame lineage object at cache time — a new DataFrame with different lineage won't reuse the old cache; the old cached DataFrame is simply a separate object.)
2. Your notebook caches 5 different large DataFrames across a session "just in case," and jobs start slowing down and spilling. Diagnose why caching *hurt* here. (Cached data competes with execution memory, causing eviction/spill for actual computation — over-caching backfires.)
3. You call `df.cache()` but never call an action before writing results downstream directly from `df`. Was the cache useful at all? (No — caching only helps if the same materialized data is reused across multiple actions; a single downstream consumption gains nothing extra from caching.)
4. Explain a real pipeline scenario where `persist(StorageLevel.MEMORY_AND_DISK)` is clearly better than `cache()` (which defaults to `MEMORY_ONLY` for DataFrames using `MEMORY_AND_DISK` in recent Spark? — clarify: DataFrame `.cache()` actually defaults to `MEMORY_AND_DISK` in modern Spark, unlike RDD's `MEMORY_ONLY` default — a good trap to raise).

**Trap / Conceptual**
5. Does calling `cache()` immediately execute the DataFrame? *(No — caching is lazy; an action materializes the cached data.)*
6. True/false: `.cache()` and `.persist()` are functionally different methods. (False, essentially — `.cache()` is shorthand for `.persist()` with a default storage level.)
7. Does uncached data get recomputed from scratch every single time it's referenced in a new action? (Yes, unless cached/persisted or checkpointed — recomputed via lineage each time.)
8. Is caching always beneficial for iterative algorithms? (Generally yes for reused data, but can hurt if data doesn't fit memory causing eviction/spill overhead exceeding recomputation cost.)

**Coding / Edge Cases**
9. 
```python
df = spark.read.parquet(path)
df.cache()
df.count()
df2 = df.filter(df.x > 10)
df2.count()
```
Is `df2`'s underlying scan served from cache, or recomputed from source? (Served from `df`'s cached data since `df2` derives from the already-cached `df` lineage.)
10. Edge case: calling `.unpersist()` while a job is actively reading from the cached data — what happens? (Depends on timing/blocking parameter; can cause the currently running job to recompute from source if data is evicted mid-use, or block until safe depending on `blocking=True/False`.)

**Expected Output**
11. In Spark UI's "Storage" tab, what would you expect to see for a DataFrame cached with `MEMORY_AND_DISK` that partially spilled? (Some fraction of the cached size reported "In-memory" and some fraction "On Disk".)

**Internals & Performance**
12. What are the available storage levels (`MEMORY_ONLY`, `MEMORY_AND_DISK`, `MEMORY_ONLY_SER`, `DISK_ONLY`, replication variants) and when would each be appropriate?
13. When caching helps: describe a concrete case (iterative ML training, reused intermediate DataFrame across multiple downstream branches).
14. When caching hurts: describe a concrete case (single-use DataFrame, or data too large causing eviction/GC pressure).
15. How does cache eviction (LRU) interact with active jobs needing that same data?

**Follow-ups**
16. "How is `.cache()` different from `.checkpoint()` in terms of lineage and fault tolerance?" (Cache preserves lineage for recomputation on eviction/failure; checkpoint truncates lineage entirely and writes reliably to durable storage.)
17. "If you cache a DataFrame but a later filter is applied before every use, should you cache before or after the filter?" (After the filter — cache only the data actually reused downstream to save memory.)
18. "How would you decide whether a piece of your pipeline should be cached, checkpointed, or just recomputed?" (Based on reuse frequency, cost of recomputation, lineage length/fault-tolerance needs, and available memory.)
19. "What's the risk of forgetting to `.unpersist()` long-lived cached DataFrames in a long-running interactive cluster/notebook environment?" (Gradual memory exhaustion across many sessions/users sharing a cluster.)

---

## 15. Broadcast Variables

**Scenario / Complex**
1. You join a 2TB fact table with a 50MB dimension table, but the plan shows a `SortMergeJoin` instead of a `BroadcastHashJoin`. Diagnose why (stats missing/stale, size exceeds `autoBroadcastJoinThreshold`, one side involves a complex subquery Spark can't estimate).
2. You explicitly call `broadcast(small_df)` but the job OOMs on the driver during planning. What went wrong? (The "small" table wasn't actually small — collecting/broadcasting it exceeded driver or executor broadcast memory limits.)
3. Explain the full lifecycle of a broadcast join: driver collects small table → broadcasts to all executors → each executor performs local hash join against its partition of the large table.
4. When would a broadcast join *increase* total cluster memory usage compared to a shuffle join, even though it avoids a shuffle? (When many executors hold a full redundant copy simultaneously — memory multiplied by executor count vs distributed shuffle data.)

**Trap / Conceptual**
5. Does `spark.sql.autoBroadcastJoinThreshold` guarantee a broadcast join will happen below that size? (Not guaranteed — depends on accurate size estimation via stats; also disabled for certain join types like full outer joins on the broadcast side.)
6. True/false: broadcasting is always safer than a full shuffle join. (False — broadcasting a mis-estimated large table can OOM the driver/executors far worse than a well-partitioned shuffle join.)
7. Can you broadcast a DataFrame involved in a full outer join as the broadcast side? (No — broadcast joins aren't supported for full outer joins as the broadcasted/outer side in some configurations — must fall back to shuffle-based strategies.)
8. Does broadcasting eliminate shuffle for *both* sides of the join, or just one? (Just the large side avoids being shuffled/repartitioned by key; the small side is fully replicated instead.)

**Coding / Edge Cases**
9. 
```python
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), "id")
```
Edge case: what if `small_df` has grown beyond `autoBroadcastJoinThreshold` since you last checked — does the explicit `broadcast()` hint still force it? (Yes — explicit hints override the auto-threshold check, which is exactly the danger scenario for OOM.)
10. Edge case: broadcasting a DataFrame with skewed/duplicate keys causing multiplicative row explosion on join — how does that interact with broadcast vs shuffle strategy risk? (Row explosion happens regardless of join strategy; broadcast doesn't cause it but doesn't protect against it either.)

**Expected Output**
11. In `.explain()`, what does a `BroadcastExchange` + `BroadcastHashJoin` pairing in the physical plan confirm? (That Spark broadcast the smaller side and performed local hash joins per partition of the larger side.)

**Internals & Performance**
12. Learn: how do broadcast joins work — describe driver memory implications precisely (the small table is collected to the driver before being broadcast to executors, so driver memory is a real constraint, not just executor memory).
13. When is broadcast dangerous? (Underestimated table size, growth over time without threshold re-check, or complex nested subqueries where Spark's cost-based size estimate is unreliable.)
14. Auto broadcast threshold: how is it configured (`spark.sql.autoBroadcastJoinThreshold`, default 10MB) and what happens if you set it to `-1`? (Disables automatic broadcast join entirely.)

**Follow-ups**
15. "How would AQE change broadcast join decisions compared to static Catalyst planning?" (AQE can dynamically switch a shuffle join to a broadcast join at runtime after observing actual post-shuffle/post-filter size stats, not just initial estimates.)
16. "What's the driver memory risk specifically, separate from executor memory, when broadcasting?" (The small table must first be materialized/collected on the driver before broadcast — driver OOM risk if underestimated.)
17. "How would you safely test whether a table is safe to broadcast in production before hardcoding a hint?" (Check actual table size stats, monitor driver memory during a dry run, consider dynamic/AQE-based broadcast instead of hardcoded hints for tables that change size over time.)

---

## 16. Join Internals

**Scenario / Complex**
1. You're joining two multi-billion-row tables, neither small enough to broadcast. Spark chooses `SortMergeJoin`. Explain the internal steps (both sides shuffled/hash-partitioned by join key, then sorted within partitions, then merged).
2. A `SortMergeJoin` on a skewed key causes one task to take 40x longer than others. How would AQE's skew join optimization fix this without code changes? (Splits the skewed partition into multiple smaller sub-partitions/tasks processed in parallel, then unions results.)
3. When would Spark choose a `ShuffleHashJoin` over `SortMergeJoin`, and why is it less commonly the default choice in modern Spark? (When one side is small enough to build an in-memory hash table post-shuffle but not small enough to broadcast; disabled by default/less preferred due to memory risk building a full hash table per partition vs sort-based approach being more memory-robust.)
4. You accidentally write a join without an equality condition (a Cartesian-style join). What operator appears, and what's the real-world danger? (`BroadcastNestedLoopJoin` or `CartesianProduct` — explosive row count, extremely expensive, often a sign of a missed join condition.)
5. Compare cost/performance of `BroadcastHashJoin`, `SortMergeJoin`, `ShuffleHashJoin`, and `CartesianProduct` for a large-large join scenario — rank them and explain.

**Trap / Conceptual**
6. Does Spark always pick the theoretically most efficient join strategy? (No — it depends on accurate statistics; stale/missing stats can lead to suboptimal choices.)
7. True/false: `SortMergeJoin` requires both sides to be shuffled even if already partitioned identically upstream. (False in ideal cases — if both sides already share compatible partitioning from an earlier operation, Spark/AQE can sometimes avoid re-shuffling, though this isn't guaranteed and depends on plan analysis.)
8. Is a broadcast nested loop join always bad? (Not always — for genuinely tiny tables with non-equi join conditions, it may be the only/reasonable option, but it's dangerous at scale.)
9. Does join order matter for correctness? For performance? (Not for correctness in inner joins generally; absolutely for performance — join reordering is a key cost-based optimization.)

**Coding / Edge Cases**
10. 
```python
big.join(small, big.id == small.id, "left")
```
vs explicitly hinting `big.join(broadcast(small), ...)` — under what condition would Spark have chosen broadcast automatically anyway, making the hint redundant? (If `small`'s size is already below `autoBroadcastJoinThreshold` and stats are accurate.)
11. Edge case: joining on a column with different data types on each side (e.g., `int` vs `string` id) — what happens to the join, silently or with an error? (Spark generally requires implicit/explicit cast compatibility; can silently cast types leading to unexpected matches/mismatches — a common data quality trap.)

**Expected Output**
12. `.explain()` shows `SortMergeJoin` with `Sort` nodes on both input branches before the join — why is sorting necessary for this strategy specifically? (Merge-join algorithm requires both sides sorted by join key to merge efficiently in a single pass.)

**Internals & Performance**
13. Precisely describe how `BroadcastHashJoin` avoids shuffle for the large side (small side broadcast fully; large side probed locally per partition against the broadcast hash table).
14. Why is `SortMergeJoin` generally more memory-robust than `ShuffleHashJoin` for large-large joins? (Sort-based external merge doesn't require holding the entire partition's hash table in memory at once, better handling memory pressure via spill-friendly sorting.)
15. How does data skew specifically degrade `SortMergeJoin` performance (one partition's sort/merge dominates runtime)?

**Follow-ups**
16. "If both tables are huge and skewed, what real levers do you have (salting, AQE skew join, pre-aggregation, broadcasting a filtered subset)?"
17. "How would you detect which join strategy Spark actually used in a production job after the fact?" (Spark UI SQL tab / physical plan inspection post-execution, event logs.)
18. "Explain the difference between an *equi-join* and *non-equi-join* and how that constrains available join strategies." (Non-equi joins can't use hash-based strategies like broadcast/shuffle hash join; typically fall back to nested loop or sort-merge with range conditions.)
19. "How does join reordering (Catalyst cost-based optimization) decide the order for a 4-table join query?" (Uses table/row-count statistics to minimize intermediate result sizes, when CBO is enabled and stats are available.)
20. "What's your production checklist before deploying a large join to avoid surprises (stats freshness, skew check, broadcast size validation, explain plan review)?"

---

## 17. Data Skew

**Scenario / Complex**
1. Spark UI shows: Task 1-3 finish in ~10 seconds, Task 4 runs for 45 minutes on the same stage. Walk through your full diagnostic and remediation process end-to-end.
2. `customer_id=123` accounts for 80% of rows due to a bot/system account polluting real customer data. How would salting solve this for a `groupBy` aggregation, step by step? (Add a random suffix to the skewed key to split it into N sub-keys, aggregate at sub-key level, then aggregate the partial results back together by original key.)
3. You enable AQE skew join handling but still see a lagging task. What could still be wrong (skew threshold config too high, skew in a non-join operation like `groupBy` which AQE skew *join* handling doesn't address, or skew is upstream of a narrow op)?
4. Two tables both have skew on the *same* join key. Does salting need to be applied to both sides symmetrically, and why? (Yes — the salted key must match on both sides for correct join results; typically explode the smaller/broadcast side across all salt values while salting the large side with a single random value per row.)
5. How would you distinguish "skew" from simply "one partition legitimately has more data because the business data itself is uneven" — are these different problems requiring different fixes?

**Trap / Conceptual**
6. Does increasing `spark.sql.shuffle.partitions` fix skew? (No — more partitions doesn't split an individual overloaded key's rows across partitions; the skewed key's rows still hash to the same partition(s).)
7. True/false: skew only affects `groupBy`/aggregation, not joins. (False — skew significantly affects joins too, arguably worse since it's compounded across both sides.)
8. Does caching help with skew? (No — caching doesn't redistribute data; skew is about key distribution during shuffle, unrelated to caching.)
9. Is skew always caused by a single dominant key, or can it be more diffuse? (Can be diffuse — several moderately-large keys combined, not just one extreme outlier; detection approach differs slightly.)

**Coding / Edge Cases**
10. How would you detect skew programmatically before running the full job (e.g., `df.groupBy("key").count().orderBy(desc("count")).show()`)? Edge case: this detection query itself can be expensive/skewed on huge data — how would you sample first?
11. Write pseudocode for salting a skewed join: add `salt = rand() % N` to the large table's key, explode the small table N times with each salt value, then join on `(key, salt)`.

**Expected Output**
12. In the Spark UI Stage detail page, which specific visualization/metric most directly reveals skew (task duration distribution / max vs median task time, and "Shuffle Read Size" per task)?

**Internals & Performance**
13. How to detect skew: task duration/shuffle-read imbalance in Spark UI, `.groupBy(key).count()` distribution checks, or event log analysis.
14. Why does skew happen? (Natural data imbalance — power-law distributions, null/default keys absorbing many rows, join keys with poor cardinality.)
15. How does AQE's skew join optimization work internally (splitting an oversized partition into multiple tasks based on a size threshold, then handling the join correctly by duplicating the corresponding smaller-side partition)?
16. What repartitioning strategies help beyond salting (e.g., composite/derived keys, isolating and broadcasting the skewed key subset separately from the rest)?

**Follow-ups**
17. "If salting fixes the skew, what's the added cost/complexity it introduces (need to re-aggregate, more shuffle partitions, more complex code)?"
18. "How would you handle skew in a streaming (not batch) context differently?" (State store partitioning skew is harder to salt dynamically; may need key redesign or state store tuning.)
19. "What's a real production skew incident you've handled, and what was your root cause vs your fix?" (Expect a concrete story.)
20. "Would you rather fix skew at the data/schema design level or always patch it at the Spark job level — what's your philosophy?" (Ideally fix at the source/schema level for a lasting fix; job-level patches like salting are tactical workarounds.)

---

## 18. Adaptive Query Execution (AQE)

**Scenario / Complex**
1. A job's initial physical plan shows `SortMergeJoin`, but the Spark UI's *actual executed* SQL plan shows `BroadcastHashJoin`. Explain exactly how/why AQE made this switch mid-execution. (After the first shuffle stage completes, AQE observes the actual materialized size of one side and, if now below the broadcast threshold, replans the join strategy for the remaining execution.)
2. You have `spark.sql.shuffle.partitions=2000` hardcoded from years ago for a now much-smaller dataset. How does AQE's shuffle-partition-coalescing feature help without you changing that config? (AQE merges many small post-shuffle partitions into fewer right-sized partitions at runtime based on actual data size.)
3. Explain, end to end, how AQE handles a skewed `SortMergeJoin` — from detection to the corrective action taken at runtime.
4. A job behaves differently (fewer stages, different join strategy) between two runs on the exact same data and code — what non-deterministic AQE-driven behavior could explain this, and is it something to worry about? (AQE's plan can adapt based on actual runtime stats gathered per-run, e.g., due to slight data or cluster state differences; generally not a correctness concern, just an efficiency adaptation.)

**Trap / Conceptual**
5. Is AQE enabled by default in modern Spark (3.x+)? (Yes, by default since Spark 3.2 in most distributions — a common trap if a candidate assumes it's opt-in.)
6. True/false: AQE can change join strategy but never changes the actual output correctness/semantics. (True — AQE affects performance/execution strategy only, never correctness of results.)
7. Does AQE re-run completed stages when it adapts the plan? (No — it adapts *future* stages, not those already executed.)
8. Does AQE eliminate the need to ever manually tune `spark.sql.shuffle.partitions`? (Reduces the need significantly, but doesn't eliminate all manual tuning needs, especially for non-shuffle-related tuning or edge cases AQE doesn't cover.)

**Coding / Edge Cases**
9. How would you confirm AQE actually altered a plan for a specific job (compare `explain()` pre-execution vs the Spark UI's SQL tab post-execution plan)?
10. Edge case: AQE's skew join handling has size/ratio thresholds (`spark.sql.adaptive.skewJoin.skewedPartitionFactor`/`skewedPartitionThresholdInBytes`) — what happens if your skew is real but below these thresholds? (AQE won't intervene — you'd still see a lagging task and need manual mitigation like salting.)

**Expected Output**
11. Where in the Spark UI would you find evidence that AQE coalesced shuffle partitions (e.g., fewer actual tasks in a stage than the configured `shuffle.partitions` value)?

**Internals & Performance**
12. Understand the AQE loop precisely: initial plan → execute part of query → collect runtime statistics → modify plan → continue execution — what triggers each replanning point (typically at shuffle/materialization boundaries between stages)?
13. Important AQE features: coalescing shuffle partitions, dynamic join strategy switching, skew join optimization, use of runtime statistics — explain each in 1-2 sentences.
14. Why can Spark's actual execution plan differ from the initial plan? *(AQE is a major reason — replanning after observing real intermediate data statistics rather than relying solely on pre-execution estimates.)*

**Follow-ups**
15. "If AQE requires materializing shuffle stages to gather stats, does that mean it can't help within a single unbroken stage (i.e., before any shuffle boundary)?" (Correct — AQE adapts at shuffle/stage boundaries; it doesn't optimize inside a fully pipelined single stage that has no materialization point.)
16. "How would you tune AQE's advisory partition size or skew thresholds for your specific workload?" (`spark.sql.adaptive.advisoryPartitionSizeInBytes`, skew factor/threshold configs — tuned based on observed task size distributions.)
17. "Does AQE help with narrow-transformation-only pipelines (no shuffle at all)?" (No — without a shuffle/materialization boundary, there's no runtime stats collection point for AQE to act on.)
18. "What's a case where AQE's dynamic join switching could still get it wrong?" (If post-shuffle partition stats still don't accurately reflect true broadcastability due to compression/estimation quirks, or thresholds set inappropriately.)

---

## 19. Query Optimization (Systematic Debugging)

**Scenario / Complex**
1. Given only "the job used to take 20 minutes, now takes 3 hours" with no code changes, walk through your full systematic diagnostic flow using the Spark UI (Jobs → Stages → identify slow stage → shuffle/skew/task-distribution/input-output checks → physical plan review).
2. A job's total task count looks normal, and no task is a dramatic outlier, yet the whole job is just uniformly slow. What classes of problems does this suggest (versus the classic single-skewed-task pattern)? (Under-provisioned cluster, inefficient/UDF-heavy transformations, small-file overhead, excessive I/O, or genuinely increased data volume.)
3. You identify a wide transformation causing a large shuffle that seems unavoidable given the business logic. What are your options to reduce its *cost* even if you can't eliminate the shuffle itself (better key distribution, pre-aggregation before shuffle, right-sizing shuffle partitions, using more efficient serialization)?
4. Data volume grew 10x year-over-year but nobody revisited cluster sizing or partition counts. What's your methodical approach to re-right-sizing the whole pipeline (not just increasing everything blindly)?

**Trap / Conceptual**
5. Is "add more executors" always the correct first response to a slow job? (No — often masks a root cause like skew or inefficient code that will resurface at the next scale threshold.)
6. True/false: a job with zero failed tasks and zero spill is automatically "well optimized." (False — it could still be inefficient in resource usage, e.g. using Python UDFs unnecessarily, or over-partitioned causing scheduling overhead, without technically failing or spilling.)
7. Does looking only at total job duration tell you where the bottleneck is? (No — you must drill into per-stage and per-task metrics.)

**Coding / Edge Cases**
8. How would you programmatically extract stage-level metrics (shuffle read/write, spill, task duration distribution) via the Spark REST API or event logs for automated regression detection across job runs?
9. Edge case: a job's `explain()` plan looks perfectly optimized (no UDFs, pushdown present, broadcast join used) but it's still slow — where do you look next? (Cluster resource contention/multi-tenancy, storage layer throttling (e.g., S3 request limits), skew not visible in static plan, GC pauses, network issues.)

**Expected Output**
10. Given a hypothetical Spark UI screenshot description (one stage with 200 tasks, most finishing in 5s, five tasks running 30+ minutes with high "Shuffle Read Size"), state your diagnosis and next 3 concrete actions. (Diagnosis: data skew on shuffle key; actions: enable/verify AQE skew handling, inspect key distribution, consider salting or isolating hot keys.)

**Internals & Performance**
11. Walk the full systematic flow: Slow Job → Spark UI → identify slow stage → check shuffle → check skew → check task distribution → check input/output → check physical plan → optimize.
12. List concrete optimization levers and when each applies: filter early, select only required columns, avoid unnecessary shuffles, broadcast small tables, handle skew, tune partitions, use appropriate file sizes, avoid Python UDFs, use Delta optimizations.
13. Why is "filter early" (predicate pushdown mindset even when writing code manually) one of the highest-leverage habits for large-scale pipelines?

**Follow-ups**
14. "How do you prioritize among 5 simultaneous candidate root causes when time-constrained in an incident?" (Start with cheapest-to-check, highest-likelihood culprits — Spark UI stage view for skew/spill first, then plan review.)
15. "How would you build regression testing/alerting so this kind of slowdown gets caught before it becomes a 3-hour production incident?" (Track historical stage duration/shuffle metrics over time, alert on anomalies, periodic `ANALYZE TABLE` stats refresh.)
16. "What's the difference between optimizing for latency (a single job) versus optimizing for cluster-wide throughput (many concurrent jobs)?" (Latency optimization might over-provision one job; throughput optimization must consider fair resource sharing/queueing across jobs.)
17. "Tell me about the single most impactful Spark optimization you've personally made in production." (Expect concrete story with before/after metrics.)

---

## 20. File Formats

**Scenario / Complex**
1. Your pipeline reads raw CSV daily, applies transformations, and writes back to CSV, taking hours. Propose a redesign using Parquet/Delta and explain concretely where the time savings come from (columnar pruning, predicate pushdown, compression, avoiding repeated schema inference).
2. A downstream consumer only needs 3 of 200 columns from a massive table. Compare the I/O cost of this query if the table is stored as CSV/JSON vs Parquet/ORC. (Columnar formats read only the needed column chunks; row-based formats must read entire rows/full files regardless of columns selected.)
3. You apply a filter on a Parquet file's column that has row-group level statistics (min/max). Explain precisely how predicate pushdown uses these stats to skip entire row groups without reading them.
4. Why might a poorly-compacted Parquet dataset (many tiny files) still perform poorly despite being a "good" columnar format? (Small-file overhead dominates regardless of format — file format choice doesn't fix file-size/count problems.)

**Trap / Conceptual**
5. Does predicate pushdown work on JSON files the same way it does on Parquet? (No — JSON/CSV lack embedded column statistics/row-group metadata, so pushdown benefits are far more limited, typically requiring a full read/parse.)
6. True/false: ORC and Parquet are functionally identical for Spark's purposes. (Mostly similar columnar benefits, but different ecosystems/optimizations/defaults — not "identical," worth knowing both exist and Parquet is Spark's more common default.)
7. Is schema inference required for Parquet the way it is for CSV? (No — Parquet embeds schema in its metadata footer; CSV requires either inference (an extra read pass) or explicit schema definition.)
8. Does compression always help performance, or can it hurt? (Generally net-positive for I/O-bound workloads, but adds CPU decompression cost — for CPU-bound compute-heavy jobs on fast storage, tradeoffs should be evaluated.)

**Coding / Edge Cases**
9. Reading a Parquet directory where files have slightly inconsistent schemas (extra column in some files) — what happens by default, and what config controls merge behavior? (`mergeSchema` option controls whether Spark reconciles differing schemas across files; without it, may error or use only the first file's schema depending on settings.)
10. Edge case: writing a DataFrame with a column containing commas as CSV without proper quoting config — what data corruption risk arises downstream? (Improper delimiter escaping corrupts row/column boundaries on read.)

**Expected Output**
11. Given `df.filter(col("date") == "2024-01-01")` on a Parquet table with min/max row-group stats, what would you expect `.explain()` / the scan metrics to show regarding "files/row-groups pruned"? (Evidence of pushed filters and reduced files-read/bytes-read compared to a full scan.)

**Internals & Performance**
12. Explain columnar storage's core performance advantage over row-based storage for analytical (read-heavy, column-selective) workloads.
13. How does compression interact with columnar layout to achieve better ratios than row-based formats (similar data types grouped together compress better)?
14. What are "file statistics" in Parquet/Delta and how do they enable both predicate pushdown and data skipping?

**Follow-ups**
15. "When would you actually still choose CSV/JSON over Parquet despite the performance downsides?" (Interop with external non-Spark systems, human-readability needs, small one-off datasets, ingestion from external sources you don't control.)
16. "How does Delta build on top of Parquet's benefits — what does it add?" (Transaction log for ACID/versioning, schema enforcement/evolution, time travel, data skipping via file-level stats, on top of Parquet's columnar physical storage.)
17. "What's your approach to choosing target file size when writing Parquet at scale?" (Target ~128MB-1GB per file typically, balancing task parallelism against small-file metadata overhead — tuned via repartition/coalesce or Delta `OPTIMIZE`.)

---

## 21. Delta Lake Internals

**Scenario / Complex**
1. If a Delta table contains Parquet files, how does Delta know which files belong to the current table version? *(Answer: the transaction log (`_delta_log`) records `add`/`remove` actions per commit, and the current snapshot is reconstructed by replaying the log (plus latest checkpoint) to determine the exact active file set.)*
2. You run `VACUUM` with a very short retention period on a table that has concurrent readers doing time travel to yesterday's version. What breaks? (Files needed for that older version get physically deleted, causing time-travel queries or long-running readers referencing removed files to fail.)
3. Two concurrent writers commit to the same Delta table at nearly the same time. Explain how optimistic concurrency control resolves this (both attempt to commit the next version number; one succeeds, the other detects conflict and retries/rebases if possible, or fails if truly conflicting).
4. Explain what `OPTIMIZE` + `ZORDER BY` actually does physically to the underlying files and why it improves query performance (compacts small files into larger ones and co-locates data with similar Z-order column values to improve data-skipping efi

ciency for multi-column filters).
5. A table has deletion vectors enabled. Explain how a `DELETE`/`UPDATE` operation differs physically from the traditional copy-on-write rewrite approach (marks rows as deleted via a deletion vector file rather than rewriting entire Parquet files — a copy-on-write vs merge-on-read style tradeoff).

**Trap / Conceptual**
6. Does `DELETE` on a Delta table without deletion vectors rewrite entire files, or just remove rows in place? (Rewrites entire affected Parquet files — Parquet files are immutable; Delta creates new files and marks old ones removed in the log.)
7. True/false: `_delta_log` files are Parquet. (False for the per-commit JSON files — periodic checkpoints *are* Parquet, but individual commits are JSON.)
8. Does time travel require re-reading all historical data each time, or just the log? (Requires reconstructing the file list for that version from the log/checkpoint, then reading only the Parquet files valid at that version — not literally replaying all history's data.)
9. Does `VACUUM` remove data referenced by the *current* table version? (No, by design/safety — it only removes files no longer referenced by the current version and older than the retention threshold.)

**Coding / Edge Cases**
10. Given a table history with commits 0-10, and you run `SELECT * FROM table VERSION AS OF 5`, what does Delta do internally to answer this query? (Reconstructs the exact active-file set as of version 5 from the transaction log/checkpoint, then reads those specific Parquet files.)
11. Edge case: running `OPTIMIZE` on a table with active concurrent writers — does it block writes, and how does optimistic concurrency handle the resulting file rewrite conflicting with a concurrent commit? (Delta's OCC mechanism/conflict detection handles this — OPTIMIZE and concurrent writes can generally coexist through commit conflict resolution, though specifics depend on isolation level and operation types.)

**Expected Output**
12. After several small `MERGE`/`UPDATE` operations without compaction, what would you expect `DESCRIBE DETAIL` or file-count metrics to show, and how would that affect subsequent read performance? (High file count relative to data size — small-file overhead degrading scan performance.)

**Internals & Performance**
13. `_delta_log`, JSON transaction files, checkpoints, `add`/`remove`/`metadata` actions, protocol, commit, snapshot, version — define each briefly and how they relate.
14. Why do periodic checkpoints (Parquet snapshots of the log) exist rather than always replaying every JSON commit from version 0? (Performance — avoids replaying potentially thousands of incremental JSON commits on every table read.)
15. How do deletion vectors improve `UPDATE`/`DELETE`/`MERGE` performance versus full file rewrites? (Avoids rewriting entire large files for small row-level changes — much cheaper for sparse modifications.)
16. Explain ACID guarantees in Delta specifically: how is atomicity achieved (single commit = single log entry, all-or-nothing), and how is isolation achieved (snapshot isolation via versioned reads)?

**Follow-ups**
17. "How does schema enforcement differ from schema evolution, and how does Delta decide which applies on a given write?" (Enforcement blocks writes violating existing schema by default; evolution (`mergeSchema`) explicitly allows compatible schema changes when opted in.)
18. "What's the interaction between `OPTIMIZE`/`ZORDER` and data skipping — why does ZORDER specifically help multi-column filter queries more than simple file compaction alone?" (Co-locating correlated column value ranges within files makes per-file min/max stats far more selective for filters spanning those columns.)
19. "How would you design a retention/VACUUM policy for a table that both supports time-travel audits and controls storage cost?" (Balance retention window against audit/compliance requirements and storage cost, ensuring no active readers/streams need older versions being vacuumed.)
20. "Explain how Delta's transaction log enables safe concurrent batch + streaming writes to the same table." (Structured Streaming can track progress via the log's versioned commits, and OCC ensures conflicting concurrent writes are detected/retried rather than corrupting data.)

---

## 22. Driver vs Executor Internals

**Scenario / Complex**
1. A job that only does `df.filter(...).select(...).write.parquet(...)` (no `.collect()`/`.show()`) still shows meaningful driver memory usage growth over time across many iterations in a loop. What driver-side state might be accumulating (query plan objects, accumulators, listener/event bus history, broadcast variable references not cleaned up)?
2. Where does UDF execution actually run — driver or executor — and why does that matter for debugging print statements inside a UDF? (Executor — `print()` inside a UDF won't show in the driver/notebook output; it appears in executor logs instead, a common debugging trap.)
3. You call `.collect()` on a DataFrame you believe is small, but it's actually large due to an upstream bug — trace exactly what happens at each architectural layer (executors compute all partitions, serialize the results, send them over the network to the driver, driver deserializes and stores in its heap) and where the failure will manifest.

**Trap / Conceptual**
4. Complete correctly — Build logical plan: Driver. Catalyst optimization: Driver. DAG scheduling: Driver. Task scheduling: Driver. Task execution: Executor. UDF execution: Executor. Data processing: Executor. `collect()` result: Executor → Driver. `show()` result: Executor → Driver.
5. Does `show(20)` bring back the entire DataFrame to the driver, or just what's needed? (Just the requested rows — Spark short-circuits/limits computation where possible, unlike `.collect()` which brings back everything.)
6. True/false: broadcast variable creation happens entirely on executors. (False — the data is first collected/prepared on the driver, then distributed to executors.)
7. Does the Catalyst optimizer run per-task on each executor, or once on the driver? (Once on the driver — the optimized physical plan is then shipped to executors for execution, not re-optimized per task.)

**Coding / Edge Cases**
8. 
```python
def my_udf(x):
    print(f"Processing {x}")  # where does this output go?
    return x * 2
```
Answer: executor stdout/logs, not the driver console — a classic debugging trap for notebook users expecting to see prints inline.
9. Edge case: an accumulator updated inside a transformation (not an action) — is its value guaranteed correct/consistent on the driver? (Accumulators updated inside transformations (not actions) can be applied more than once due to task retries/speculative execution, risking incorrect counts — a well-known accumulator pitfall.)

**Expected Output**
10. If you call `df.show()` vs `df.collect()` on the exact same DataFrame, does the underlying job/task execution differ in scope? (`show()` can potentially limit computation to fewer partitions/rows needed to satisfy the limit, especially with `.limit()`-like short-circuiting; `.collect()` always computes and returns all partitions fully.)

**Internals & Performance**
11. Why is understanding driver vs executor responsibility "an extremely common interview area" from a production-debugging standpoint (helps localize OOMs, slow prints, serialization errors, and misplaced business logic)?
12. What's the performance/scalability implication of putting heavy computation logic inside driver-side code (e.g., large Python loops before invoking Spark actions) versus expressing it as DataFrame transformations executed on executors?

**Follow-ups**
13. "If a job fails with a stack trace mentioning `PythonException` inside task execution, where did that fail — driver or executor?" (Executor — task-level Python UDF execution failure.)
14. "How would you design logging so you can actually see executor-side print/debug output during development?" (Use `logging`/executor logs via the cluster manager UI (YARN/K8s/Databricks driver+executor logs), or use accumulators/structured logging rather than relying on `print()`.)
15. "What's a concrete driver-memory bug you've debugged, and how did you trace it back to a driver-side operation?" (Expect a story — e.g., huge broadcast, `.collect()` misuse, accumulator misuse, or plan complexity from many unioned DataFrames.)

---

## 23. Fault Tolerance

**Scenario / Complex**
1. An executor is lost mid-shuffle due to a spot-instance preemption. Walk through exactly what Spark does to recover — does it restart the whole application, or just recompute lost partitions/tasks? *(Usually no full restart — Spark recomputes lost data/tasks based on lineage and retries failed tasks/stages as appropriate; if lost shuffle files were needed by downstream stages, those map tasks get re-run too.)*
2. A task fails repeatedly (not due to data, but due to a bad node) — how does Spark's blacklisting/task-retry mechanism handle this to avoid infinite retries on the same bad node? (Node blacklisting after repeated failures reschedules the task elsewhere; retry limits (`spark.task.maxFailures`) eventually fail the stage/job if exceeded.)
3. In a long streaming job with checkpointing enabled, the driver crashes and restarts. Explain how checkpointed state/offsets allow exactly-once (or at-least-once) recovery without reprocessing the entire stream from the beginning.
4. Explain the difference in blast radius between losing a single task, losing an entire executor, and losing the driver — and how recovery differs for each.

**Trap / Conceptual**
5. If an executor dies, does Spark restart the entire application? *(Usually no — Spark can recompute lost data/tasks based on lineage and retry failed tasks/stages as appropriate; only driver failure typically kills the whole application unless using a supervised/HA driver setup.)*
6. True/false: RDD lineage alone is enough to recover *shuffle* data lost when an executor dies. (Mostly true conceptually — lineage allows recomputing the map-side tasks that produced the lost shuffle files, though this can be expensive if it requires recomputing far upstream, which is exactly why checkpointing exists for long lineages.)
7. Does caching a DataFrame protect it from being lost if an executor holding cached partitions dies? (No — cached data on a lost executor is simply gone; Spark recomputes those specific partitions from lineage on demand, it doesn't magically survive the executor loss.)
8. Is task retry the same mechanism as stage retry? (Related but distinct — individual task failures trigger task-level retry; if failures indicate a systemic issue (e.g., lost shuffle map output), Spark may need to retry the entire preceding stage to regenerate that data.)

**Coding / Edge Cases**
9. How would you configure `spark.task.maxFailures` and what tradeoff does raising/lowering it represent? (Higher tolerates more transient failures before giving up (more resilient but slower to surface real bugs); lower fails fast but is less tolerant of transient infra issues.)
10. Edge case: a task has a non-idempotent side effect (e.g., writes to an external API) inside a `map`. If that task is retried due to failure, what real-world bug results? (Duplicate external side effects — a critical correctness trap tied to Spark's retry-based fault tolerance model.)

**Expected Output**
11. In Spark UI, how would you identify that a stage suffered an executor loss and recomputation, versus just normal slow tasks? (Look for stage retries, "Lost" executor entries in the Executors tab/event timeline, and duplicate stage attempt IDs.)

**Internals & Performance**
12. Explain how lineage underlies fault tolerance conceptually (each RDD/DataFrame knows how to recompute itself from its parent + transformation, forming a recovery graph without needing full data replication).
13. What's the performance cost tradeoff of relying purely on lineage-based recovery for very long transformation chains, and how does checkpointing address it?
14. How does shuffle data loss specifically get detected and triggers re-execution of the *producing* stage, not just the consuming one?

**Follow-ups**
15. "How does this differ in a Structured Streaming context with exactly-once sink guarantees?" (Requires idempotent/transactional sinks plus checkpointed offsets/state, not just lineage-based recompute.)
16. "What's your production strategy for handling non-idempotent side effects safely under Spark's retry model?" (Design idempotent writes (upserts keyed on unique IDs), use transactional sinks, or push side effects to a separate exactly-once-guaranteed stage.)
17. "Tell me about a time an executor/node failure caused a real production incident, and what changed afterward (e.g., adding checkpointing, idempotency, retry tuning)."

---

## 24. Checkpoint vs Cache

**Scenario / Complex**
1. An iterative ML/graph algorithm (e.g., PageRank-style) grows its DataFrame lineage by one more transformation every iteration for 100 iterations. Without checkpointing, what breaks eventually (driver stack overflow / extremely expensive plan optimization and potential full recomputation on failure due to enormous lineage)? Explain how periodic checkpointing solves this by truncating lineage.
2. You checkpoint a DataFrame mid-pipeline. Does the checkpoint operation still allow you to trace back to original lineage for debugging, or is that information gone? (Gone/truncated by design for reliable checkpoints — the checkpointed DataFrame becomes a new lineage root pointing at the reliably-stored data, not the original chain.)
3. Compare recovery behavior: if an executor holding *cached* data dies vs one holding data that was read from *checkpointed* storage — which requires recomputation and which doesn't? (Cached data lost on executor death requires recomputation from lineage; checkpointed data is durably stored (e.g., HDFS/cloud storage), so it doesn't need recomputation — it's simply re-read from reliable storage.)

**Trap / Conceptual**
4. Is checkpointing lazy like caching, or does it trigger immediate execution? (Checkpointing typically triggers an eager action/write to reliably persist the data, unlike pure `.cache()` marking which stays lazy until the next action — worth verifying exact semantics/version nuances, but conceptually checkpoint materializes data reliably right away in most implementations.)
5. True/false: checkpoint and cache serve the same underlying purpose (performance via reuse). (False — cache is primarily a performance/reuse optimization; checkpoint is primarily a reliability/lineage-truncation mechanism, though it incidentally also avoids recomputation.)
6. Does checkpointing use executor memory the way caching does? (No — checkpoint typically writes to durable external storage (e.g., HDFS/DBFS/cloud), not executor memory.)

**Coding / Edge Cases**
7. What's the difference between "reliable checkpointing" (`sparkContext.setCheckpointDir` + `.checkpoint()`) and "local checkpointing" (`.localCheckpoint()`), and what fault-tolerance tradeoff does local checkpointing make? (Local checkpointing truncates lineage but stores data on executor local disk/memory rather than durable distributed storage — faster but not fault-tolerant against executor loss, unlike reliable checkpointing.)
8. Edge case: calling `.checkpoint()` without first setting a checkpoint directory — what happens? (Error/exception — a checkpoint directory must be configured before checkpointing can be used.)

**Expected Output**
9. After checkpointing a DataFrame and running `.explain()`, does the plan still show the original long transformation chain, or a simple scan from the checkpoint location? (A simple scan from the checkpoint location — lineage is truncated.)

**Internals & Performance**
10. Cache: used primarily for reusing computed data (in-memory/disk, tied to executor lifetime, lineage preserved for recomputation on loss).
11. Checkpoint: used to truncate lineage and persist a reliable checkpoint (durable storage, new lineage root, doesn't depend on original source for recovery).
12. Why does this distinction become particularly important in iterative workloads and streaming (unbounded lineage growth risk without periodic truncation)?

**Follow-ups**
13. "Would you ever use both cache and checkpoint together on the same DataFrame, and why?" (Yes — cache for immediate reuse performance within the current session, checkpoint for durability/lineage truncation for long-running or fault-sensitive pipelines; commonly recommended to cache before checkpointing to avoid recomputing twice.)
14. "How does Structured Streaming rely on checkpointing differently than batch iterative jobs?" (Streaming checkpoints track offsets/state for exactly-once/recovery semantics across restarts, not just lineage truncation for a single long-running batch computation.)
15. "What's the storage cost/cleanup consideration for checkpoint directories over time in a long-running production system?" (Old checkpoint data can accumulate and needs lifecycle/cleanup management, unlike ephemeral cache that's naturally cleaned up with executor/session lifecycle.)

---

## 25. Spark Streaming Internals

**Scenario / Complex**
1. A Structured Streaming job reading from Kafka falls increasingly behind (growing consumer lag) over time even though throughput looks stable. What are your top diagnostic hypotheses (undersized cluster for peak load, expensive stateful operations growing state size, shuffle-heavy aggregations per micro-batch, watermark/state cleanup not keeping state bounded)?
2. Explain micro-batch processing end-to-end: how Structured Streaming turns a continuous Kafka topic into discrete DataFrame batches per trigger interval, each going through the same Catalyst/execution pipeline as a batch job.
3. You need sub-second latency but are using default micro-batch triggers. Explain the tradeoffs of continuous processing mode versus tuning micro-batch trigger intervals down. (Continuous processing offers lower latency but with reduced fault-tolerance guarantees and operator support versus mature/robust micro-batch mode.)
4. A late-arriving event (timestamp older than the current watermark) arrives. What happens to it under a `groupBy` windowed aggregation with a watermark set? (Dropped/excluded from the aggregation since its window has already been finalized/cleaned up per the watermark policy.)

**Trap / Conceptual**
5. Is Structured Streaming truly continuous/row-at-a-time by default? (No — default execution model is micro-batch; true continuous processing is a separate, more limited experimental mode.)
6. True/false: watermarks guarantee zero data loss for late events. (False — watermarks are a bounded tradeoff mechanism; events later than the watermark threshold are intentionally dropped to bound state size, a deliberate correctness/resource tradeoff.)
7. Does "exactly-once" in Structured Streaming apply automatically regardless of the sink? (No — exactly-once end-to-end requires idempotent/transactional sinks; Spark's internal processing can guarantee exactly-once semantics for its own state, but the sink must cooperate.)
8. Is checkpointing optional for stateful streaming aggregations? (Effectively mandatory for reliable recovery of state and offsets in production stateful streaming jobs.)

**Coding / Edge Cases**
9. Write the skeleton for a Structured Streaming job reading from Kafka, applying a watermark + windowed aggregation, and writing to a sink with checkpointing configured — highlight where trigger interval and output mode are specified.
10. Edge case: changing the watermark duration or aggregation logic between restarts of a stateful streaming job using the same checkpoint directory — what breaks? (Checkpoint/state schema incompatibility can cause the job to fail to restart cleanly or produce incorrect results — state format changes generally require careful migration or a fresh checkpoint.)

**Expected Output**
11. In the Structured Streaming UI/metrics, what specific metric would reveal growing Kafka consumer lag or backpressure (input rate vs processing rate, batch duration trend)?

**Internals & Performance**
12. Define and connect: trigger, checkpoint, offset, state store, watermark, event time vs processing time, late-arriving data, exactly-once semantics, output modes (append/update/complete).
13. Why does state store size directly impact micro-batch processing time and memory pressure over a long-running streaming job?
14. Explain the Kafka → Structured Streaming → micro-batch → state → sink pipeline at a component level.

**Follow-ups**
15. "How would you handle a schema change in the Kafka source without breaking a long-running streaming job?" (Schema evolution handling, versioned schemas/Avro+schema registry, careful backward-compatible changes.)
16. "What output modes are compatible with which types of aggregations, and why?" (e.g., `complete` mode required for certain unbounded aggregations without watermarks; `append` mode compatible with watermarked windowed aggregations once finalized; `update` mode for incremental updates.)
17. "How would you size/tune trigger interval for a workload balancing latency against cluster cost/throughput?"
18. "Explain how exactly-once semantics interact with a non-idempotent sink like a plain HTTP API call." (Exactly-once processing guarantees don't extend to non-idempotent side effects — duplicate delivery possible on retry unless the sink itself is made idempotent.)

---

## 26. Structured Streaming State

**Scenario / Complex**
1. A stateful `groupBy(user_id).agg(...)` streaming aggregation runs for months without bounded watermarking, and state store size grows unbounded, eventually causing memory/performance degradation. Diagnose and propose a fix (add appropriate watermark + windowing to bound state, or explicit state TTL/cleanup logic).
2. Explain, step by step, where Spark maintains state physically for a stateful streaming aggregation (state store, typically backed by RocksDB or in-memory HDFS-backed store, checkpointed periodically to durable storage per micro-batch).
3. A stream-stream join between two Kafka topics requires buffering unmatched rows from both sides. Explain how watermarks on both streams bound how long unmatched rows are retained in state before being dropped/expired.
4. You migrate a stateful streaming job's cluster to different hardware. What state-store-related consideration must you validate before restart (state store backend compatibility, checkpoint location accessibility, sufficient local disk/memory for RocksDB-backed state if used)?

**Trap / Conceptual**
5. Is streaming state stored only in memory, or can it spill/persist to disk? (Depends on state store provider — default in-memory HDFS-backed store keeps state in executor memory (checkpointed to durable storage), while RocksDB-backed state store can spill to local disk, better for large state.)
6. True/false: watermarks are required for all stateful operations. (False — some stateful ops (e.g., certain global aggregations in `complete` mode) don't require watermarks, though without them state grows unbounded — an important tradeoff to articulate.)
7. Does stream-stream join state cleanup happen automatically without watermarks defined on both sides? (No — without watermarks, state for unmatched join rows can grow indefinitely; watermarks are essential for bounding stream-stream join state.)

**Coding / Edge Cases**
8. Write pseudocode for a stateful `groupBy(user_id).window(...)` aggregation with an appropriate watermark, and explain what happens to state for a window once the watermark passes its end boundary. (State for that window is finalized and cleaned up/emitted, no longer accepting updates.)
9. Edge case: using `mapGroupsWithState`/`flatMapGroupsWithState` for custom stateful logic — what happens if you don't explicitly set a state timeout, and unbounded groups accumulate state forever? (State per group persists indefinitely without an explicit timeout, risking unbounded growth — must configure `GroupStateTimeout`.)

**Expected Output**
10. In the Structured Streaming query progress metrics (`lastProgress`), which field(s) reveal current state store size/row counts (`stateOperators` metrics including `numRowsTotal`, memory used)?

**Internals & Performance**
11. Where does Spark maintain state? Understand: stateful operators, state store, aggregations, stream-stream joins, watermarks, state cleanup — explain how they interconnect using the `groupBy(user_id) → state → updated aggregation` example from the source material.
12. Why is RocksDB-backed state store often preferred over the default in-memory store for large-state production workloads? (Better handling of large state via disk-backed storage with efficient access patterns, reduced JVM heap/GC pressure.)
13. How does state store checkpointing per micro-batch interact with fault tolerance/exactly-once semantics for stateful operators?

**Follow-ups**
14. "How would you monitor state store growth over time in production to catch an unbounded-state bug before it causes an outage?" (Track `stateOperators` metrics from streaming query progress, alert on growth trend anomalies.)
15. "What's the operational cost of choosing a very long watermark delay to accommodate very late data?" (Larger state retained longer, higher memory/storage cost and delayed finalization of results.)
16. "Explain a real bug you've seen from misconfigured or missing watermarks in a stateful streaming job."

---

## 27. Serialization and Closures

**Scenario / Complex**
1. Given:
```python
x = 100
def my_func(row):
    return row.value + x
df.rdd.map(my_func).collect()
```
Explain exactly what Spark must do to make `x` available on executors (closure capture, serialization of the closure/environment including `x`, shipped to each executor's task).
2. A closure captures a huge in-memory Python dictionary (500MB) used for row-wise lookups in a `map`. Explain why this is dangerous, and how a broadcast variable specifically fixes it (avoids re-serializing/shipping the large object with every single task; instead sends it once per executor and shares across tasks).
3. A closure accidentally captures a Spark `DataFrame` or `SparkSession` reference itself (not serializable) inside a `map` function. What error results, and why is this a common mistake? (`NotSerializableException` or similar — Spark's driver-side session/connection objects aren't meant to be shipped to executors.)

**Trap / Conceptual**
4. Does Spark serialize the entire enclosing scope of a closure, or only the variables actually referenced inside it? (Spark's closure cleaner attempts to prune unreferenced variables/fields where possible, but naive references to outer object fields — e.g., `self.x` inside a class method used as a closure — can inadvertently pull in the entire enclosing object, a classic trap.)
5. True/false: using a broadcast variable instead of a raw closure variable changes what data is *sent*, not *how many times* it's serialized/sent. (False — the key benefit of broadcast is reducing *how often* the data is sent (once per executor vs potentially once per task), which is the actual efficiency win.)
6. Are closures in PySpark serialized differently than closures in Scala Spark? (Yes, mechanically — PySpark relies on Python's `pickle`/`cloudpickle` plus py4j/Arrow for JVM↔Python interaction, whereas Scala closures use JVM serialization (Java/Kryo) directly without a cross-language boundary.)

**Coding / Edge Cases**
7. Rewrite the dangerous large-dictionary closure example using a broadcast variable, showing the before/after code.
8. Edge case: a closure referencing a mutable global variable that's modified by driver-side code concurrently with job execution — what inconsistency risk does this create? (Executors may see stale or inconsistent snapshots depending on when serialization occurred, since captured values are effectively fixed at serialization time — a subtle non-determinism/bug risk.)

**Expected Output**
9. If a closure captures a non-serializable object, at what point does the error surface — job submission, or actual task execution on the executor? (Typically surfaces when the task is serialized for shipping (often during job submission/task creation) or immediately on the executor when deserialization is attempted — either way, before the actual row-processing logic runs.)

**Internals & Performance**
10. Why is large Python object usage inside closures specifically dangerous (repeated serialization cost per task, driver-side memory pressure building the closure, network overhead shipping it repeatedly)?
11. How does the concept of closures/serialization here directly connect back to the broadcast variables module — why are broadcast variables essentially "the fix" for the closure-serialization-overhead problem?

**Follow-ups**
12. "How would you detect in production that closure serialization overhead (not computation) is your bottleneck?" (High "Task Deserialization Time" in Spark UI relative to actual compute time per task.)
13. "What's Spark's closure cleaner, and what are its limitations?" (A mechanism attempting to strip unnecessary references before serialization to reduce closure size — but it can't always perfectly prune references buried in complex object graphs, especially with Python's more dynamic capture semantics.)
14. "Give an example from your own experience of a closure-serialization bug that caused a confusing production failure."

---

## 28. Spark UI

**Scenario / Complex**
1. A senior interviewer shows you a Spark UI screenshot: one stage, 500 tasks, most complete in under 10 seconds, but a handful show "Shuffle Read Size" 50x the median and take 20+ minutes. Diagnose and explain your reasoning chain out loud. (Classic data skew signature — concentrate on the skewed key(s), consider salting/AQE skew handling.)
2. A job shows high "GC Time" across most executors relative to task duration. What root causes would you investigate (excessive object creation from RDD/UDF-heavy code, insufficient executor memory, too many small partitions/tasks creating overhead, off-heap not enabled)?
3. Several tasks show "Failed" with retries visible, but the job ultimately succeeds. What would you check to determine whether this is a transient infrastructure blip or a symptom of a deeper data/skew/OOM problem worth fixing proactively? (Check failure reason/stack trace per failed task — OOM vs network/node-loss vs data-specific errors — and whether failures cluster on specific partitions/keys.)
4. Given a job with reasonable stage durations but enormous total wall-clock time, and you notice many small jobs/stages triggered sequentially without caching between them — what would you recommend? (Consolidate redundant recomputation via caching/restructuring the pipeline to avoid repeated identical upstream work across multiple actions.)
5. "Why is this job slow?" — given only a Spark UI Jobs page (no stage detail yet), what's your structured first move to narrow down the investigation? (Drill into the longest-running job → its stages, sorted by duration, then into the worst stage's task-level metrics.)

**Trap / Conceptual**
6. Does a stage with 100% task success and no failures guarantee it's performing optimally? (No — success doesn't imply efficiency; still need to check duration distribution, spill, GC, and shuffle metrics.)
7. True/false: "Input Size" and "Shuffle Read Size" measure the same thing. (False — Input is data read from the original source (e.g., files); Shuffle Read is data received from a prior stage's shuffle write — distinct metrics representing different stages of the pipeline.)
8. Does the Spark UI show you the AQE-adjusted runtime plan by default in the initial "SQL" tab view, or do you need to check after completion? (The most accurate/final adjusted plan is best viewed after execution completes, since AQE can adapt after the query starts — checking mid-flight or only the pre-execution plan can be misleading.)

**Coding / Edge Cases**
9. How would you programmatically pull the same metrics shown in Spark UI (stage/task durations, shuffle read/write, spill) via the Spark REST API or `SparkListener` for automated monitoring/alerting rather than manual inspection?
10. Edge case: a job's Spark UI is inaccessible because the application already terminated — how do you investigate after the fact? (Use the Spark History Server with persisted event logs, assuming event logging was enabled for the application.)

**Expected Output**
11. Given task metrics showing "Spill (Disk): 2.3 GB" for several tasks in a stage, what specific remediation would you propose first? (Increase shuffle partitions to reduce per-task data size, and/or address underlying skew, and/or increase executor memory if genuinely under-provisioned.)

**Internals & Performance**
12. Walk through the full navigation hierarchy: Jobs → Stages → Tasks → Executors → SQL — what unique diagnostic value does each tab provide that the others don't?
13. Interpret each of these together for a holistic diagnosis: task duration, input size, output size, shuffle read, shuffle write, spill, GC time, skew, failed tasks — explain how they combine into a coherent root-cause story rather than being read in isolation.
14. Why is the SQL tab (showing the physical plan with runtime metrics annotated per operator) often the single most powerful tab for diagnosing DataFrame/SQL-based job issues?

**Follow-ups**
15. "If you had to pick only 3 Spark UI metrics to monitor as automated production alerts across all jobs, which would you choose and why?" (Likely candidates: task duration skew/outliers, spill (disk), and failed task rate/executor loss — justify tradeoffs.)
16. "How would you explain a Spark UI diagnosis to a non-technical stakeholder asking 'why is the pipeline late today'?" (Translate technical findings — e.g., skew — into plain business-impact language without losing accuracy.)
17. "Walk me through a real incident where the Spark UI led you to a root cause you wouldn't have guessed from just reading the code." (Expect a genuine, specific story demonstrating hands-on debugging depth.)

---

## How to Use This Guide
- Treat each module's questions as a **mock interview round**: answer out loud or in writing before checking the embedded hints/answers.
- For every "Scenario/Complex" question, practice sketching the DAG/plan on a whiteboard — Databricks and FAANG interviewers frequently ask you to draw this live.
- Pair this guide with hands-on practice: run `df.explain("formatted")` and the Spark UI against real jobs for every module, since interviewers often probe whether your understanding is theoretical or battle-tested.
- Modules 5, 6, 17, 18 (Shuffle, Partitions, Skew, AQE) and Module 21 (Delta Internals) are the highest-yield areas for senior/staff-level Databricks interviews — allocate proportionally more practice time there.
