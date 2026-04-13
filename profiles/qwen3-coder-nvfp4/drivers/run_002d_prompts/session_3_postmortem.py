"""Session 3 prompts: multi-turn incident postmortem drafting.

Scenario: Redis OOMKill cascade at fictional SaaS "Lumenstack" affecting the
"sessionary" session-store service. All names, timestamps, engineers, and
numbers are invented. Do not map these to any real incident.
"""

SEED_TOPIC: str = "Incident postmortem: Redis OOMKill cascade + missed alert"

SEED_PROMPT: str = r"""
You are helping me draft an incident postmortem for last night's production
incident. Below is the entire evidence bundle I have so far: a narrative, a
config diff, two stack traces, log extracts, a metric dashboard description,
the PromQL alert rule that failed us, a service dependency diagram, and the
task I need you to do in this first turn. Read everything carefully before
you start writing. Stay internally consistent with every timestamp, service
name, and numeric value I give you here — I will reference them in the
follow-up turns.

=== INCIDENT NARRATIVE ===

Lumenstack runs a multi-tenant B2B SaaS platform. Our core services run in
the `lumenstack-prod` Kubernetes namespace on an internal cluster nicknamed
"bluejay". We run a Redis 7.2 cluster called `redis-sessions` as a
three-shard, six-pod (primary + replica per shard) deployment using the
Bitnami-style Helm chart we forked internally. Each Redis pod has a memory
request of 3Gi and a limit of 4Gi. The cluster fronts our session store,
which backs login, payment checkout, and the "save for later" cart feature.

On the night of Tuesday, April 8 (starting around 23:12 local time in the
`us-central` region), the payment checkout flow began returning 502s at an
elevated rate. The on-call engineer for the Platform Reliability rotation,
Priya Ostroff, was paged at 23:57 — but not by any automated alert. She was
paged via a manual `/incident declare` from the customer-support lead, Jae
Tanaka, who had noticed a spike in support tickets mentioning "stuck at
payment" and "it logged me out in the middle of adding a seat".

The incident ran from 23:12 (first Redis OOMKill) through 00:42 (cache
warm-up stabilized, error rate back under 0.5%), for a total impact window
of roughly 90 minutes. Peak error rate was 8.1% of customer-facing HTTP
requests across the `web-edge` fleet at 23:41. The payment gateway service,
`paygate-api`, saw its p99 latency climb from a baseline of 140ms to a peak
of 9.8s during the cascade. The session store, `sessionary`, had a hard
blip of 20 minutes (23:22–23:42) where roughly one in four requests
returned `SESSION_BACKEND_UNAVAILABLE`. During that blip, users who were
mid-session were silently logged out; when they retried, their carts had
been re-fetched from the authoritative Postgres store, so no actual data
was lost, but the perceived experience was ugly.

What nobody noticed until post-incident review: three weeks ago (Tuesday,
March 18), during a cost-optimization sprint we nicknamed "Project Sandstone",
one of the platform engineers, Mohit Karimi, made two changes in a single
pull request (#4412):
  1. Switched the Redis `maxmemory-policy` from `allkeys-lru` to
     `volatile-lru`, reasoning that "basically everything we write has a
     TTL anyway, and `volatile-lru` is less aggressive so we'll get better
     hit rates on hot session keys".
  2. Tweaked the Prometheus alert `RedisMemoryPressureHigh` to use a
     30-minute `avg_over_time` window on `redis_memory_used_bytes`, instead
     of the previous 5-minute window, because the old alert had been
     "noisy" during nightly backup jobs.

The unverified assumption in (1) was that ALL keys had TTLs. In fact, a
background reconciliation job we run every four hours, `session-shadow-sync`,
writes "shadow" entries into Redis under the keyspace
`sess:shadow:{tenant_id}:{user_id}` for cross-region drift detection. Those
keys were historically written without a TTL because the job itself used to
delete them at the end of each run. A refactor in late February (PR #4287)
moved the cleanup into a separate goroutine that silently stopped running
after a panic-recovery path was hit the first time. Nobody noticed because
under `allkeys-lru` those keys were getting evicted anyway. Under
`volatile-lru`, they were not — `volatile-lru` only evicts keys that have a
TTL set.

For three weeks the shadow keys accumulated. Last night, around 23:11,
shard 2's primary pod (`redis-sessions-2-0`) crossed its 4Gi limit. The pod
was OOMKilled by the kubelet at 23:12:04. On restart, `sessionary` and
`paygate-api` both retried aggressively — neither has a circuit breaker on
the Redis client — and the cache warm-up from disk (RDB snapshot on restart)
was slow enough that reconnect storms exhausted the Redis connection slot
pool at 23:14. Shard 2's replica was promoted, took on the same load, and
was itself OOMKilled at 23:19. Shard 1 followed at 23:26 as connection
churn pushed it past its limit. By 23:31 we were in a full cascade across
all three shards.

The alert `RedisMemoryPressureHigh` did not fire until 00:04 — 52 minutes
after the first OOMKill, and only because by that point enough sustained
elevated memory had accumulated inside the 30-minute averaging window to
cross the 80% threshold. The incident had been manually declared 7 minutes
earlier.

=== CONFIG DIFF (Helm values.yaml, PR #4412, three weeks ago) ===

```diff
--- charts/redis-sessions/values.yaml
+++ charts/redis-sessions/values.yaml
@@ -34,11 +34,11 @@ cluster:
   master:
     resources:
       requests:
         cpu: "500m"
         memory: "3Gi"
       limits:
         cpu: "2"
         memory: "4Gi"
     configuration: |-
       maxmemory 3584mb
-      maxmemory-policy allkeys-lru
+      maxmemory-policy volatile-lru
       timeout 0
       tcp-keepalive 60
       save ""
       appendonly yes
       appendfsync everysec
@@ -58,7 +58,7 @@ metrics:
     prometheusRule:
       enabled: true
       additionalLabels:
         release: kube-prometheus-stack
       rules:
         - alert: RedisMemoryPressureHigh
-          expr: (avg_over_time(redis_memory_used_bytes[5m]) / redis_memory_max_bytes) > 0.80
+          expr: (avg_over_time(redis_memory_used_bytes[30m]) / redis_memory_max_bytes) > 0.80
           for: 5m
           labels:
             severity: warning
             team: platform-reliability
           annotations:
             summary: "Redis memory pressure above 80%"
             description: "Redis shard {{ $labels.pod }} memory usage has been above 80% of limit."
```

=== STACK TRACE 1: Python sessionary service (redis-py 5.0.4) ===

Captured from `sessionary-api-7d9b4fc5b8-k2xph` at 23:14:37 local time.

```
ERROR    sessionary.session_store:store.py:218 Failed to read session by id
Traceback (most recent call last):
  File "/app/sessionary/session_store.py", line 212, in get_session
    raw = self._redis.get(self._key_for(session_id))
  File "/usr/local/lib/python3.11/site-packages/redis/commands/core.py", line 1805, in get
    return self.execute_command("GET", name, keys=[name])
  File "/usr/local/lib/python3.11/site-packages/redis/client.py", line 536, in execute_command
    return conn.retry.call_with_retry(
  File "/usr/local/lib/python3.11/site-packages/redis/retry.py", line 62, in call_with_retry
    fail(error)
  File "/usr/local/lib/python3.11/site-packages/redis/client.py", line 537, in <lambda>
    lambda error: self._disconnect_raise(conn, error),
  File "/usr/local/lib/python3.11/site-packages/redis/client.py", line 510, in _disconnect_raise
    raise error
  File "/usr/local/lib/python3.11/site-packages/redis/retry.py", line 59, in call_with_retry
    return do()
  File "/usr/local/lib/python3.11/site-packages/redis/client.py", line 538, in <lambda>
    lambda: self._send_command_parse_response(
  File "/usr/local/lib/python3.11/site-packages/redis/client.py", line 513, in _send_command_parse_response
    conn.send_command(*args, check_health=check_health)
  File "/usr/local/lib/python3.11/site-packages/redis/connection.py", line 481, in send_command
    self.send_packed_command(
  File "/usr/local/lib/python3.11/site-packages/redis/connection.py", line 458, in send_packed_command
    raise ConnectionError(
redis.exceptions.ConnectionError: Error while sending command to Redis: [Errno 32] Broken pipe

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/sessionary/api/handlers.py", line 147, in get_cart
    session = self.store.get_session(sid)
  File "/app/sessionary/session_store.py", line 218, in get_session
    raise SessionBackendUnavailable(str(e)) from e
sessionary.errors.SessionBackendUnavailable: Error while sending command to Redis: [Errno 32] Broken pipe

WARNING  sessionary.pool:pool.py:88 Connection pool exhausted (in_use=64, waiters=31). Rejecting request from /v1/carts/2193abe9.
ERROR    sessionary.api.handlers:handlers.py:155 Returning 503 SESSION_BACKEND_UNAVAILABLE for tenant=acme-co user=u_88412 trace=3b91...c07d
```

=== STACK TRACE 2: Go paygate-api service (go-redis/v9 9.5.1) ===

Captured from `paygate-api-6c54f79c7-q8l9v` at 23:18:02 local time.

```
2026-04-08T23:18:02.117Z ERROR  paygate/checkout   checkout.go:312  redis op failed step=reserve_idempotency_key err="context deadline exceeded" trace_id=9f7c...2a11
goroutine 4481 [running]:
runtime/debug.Stack()
    /usr/local/go/src/runtime/debug/stack.go:24 +0x5e
github.com/lumenstack/paygate/internal/telemetry.captureStack(...)
    /src/internal/telemetry/stack.go:18
github.com/lumenstack/paygate/internal/sessioncache.(*Client).Reserve(0xc000284180, {0x1a2f3c0, 0xc0007b4000}, {0xc0005b2180, 0x29}, 0x1bf08eb000)
    /src/internal/sessioncache/client.go:147 +0x2c4
github.com/lumenstack/paygate/internal/checkout.(*Handler).reserveIdempotency(0xc0002a0280, {0x1a2f3c0, 0xc0007b4000}, 0xc0004d2000)
    /src/internal/checkout/checkout.go:298 +0x1a1
github.com/lumenstack/paygate/internal/checkout.(*Handler).StartCheckout(0xc0002a0280, {0x1a30240, 0xc00031c1c0}, 0xc00048c000)
    /src/internal/checkout/checkout.go:181 +0x4f3
github.com/lumenstack/paygate/internal/httpapi.wrap.func1({0x1a30240, 0xc00031c1c0}, 0xc00048c000)
    /src/internal/httpapi/wrap.go:62 +0xd1
net/http.HandlerFunc.ServeHTTP(0x0, {0x1a30240, 0xc00031c1c0}, 0xc00048c000)
    /usr/local/go/src/net/http/server.go:2166 +0x29
github.com/lumenstack/paygate/internal/httpapi/middleware.Trace.func1({0x1a30240, 0xc00031c1c0}, 0xc00048c000)
    /src/internal/httpapi/middleware/trace.go:41 +0x185
net/http.(*ServeMux).ServeHTTP(0xc0001f2040, {0x1a30240, 0xc00031c1c0}, 0xc00048c000)
    /usr/local/go/src/net/http/server.go:2683 +0x1ad

error chain:
  github.com/redis/go-redis/v9.(*baseClient)._process: context deadline exceeded
  github.com/redis/go-redis/v9.(*Client).SetNX: context deadline exceeded
  github.com/lumenstack/paygate/internal/sessioncache.(*Client).Reserve: reserve timeout after 750ms: context deadline exceeded
```

=== LOG EXTRACT (mixed sources) ===

```
2026-04-08T23:10:58Z redis-sessions-2-0     INFO   Background saving started by pid 1
2026-04-08T23:11:02Z redis-sessions-2-0     WARN   WARNING Memory overcommit must be enabled! Current value: 0
2026-04-08T23:11:41Z redis-sessions-2-0     INFO   used_memory_human:3.91G  used_memory_rss_human:4.02G
2026-04-08T23:11:48Z redis-sessions-2-0     WARN   Can't save in background: fork: Cannot allocate memory
2026-04-08T23:11:59Z redis-sessions-2-0     WARN   OOM command not allowed when used memory > 'maxmemory'
2026-04-08T23:12:04Z kubelet/bluejay-node17 INFO   Killing container "redis" as OOMKilled, exit code 137, pod redis-sessions-2-0
2026-04-08T23:12:04Z kubelet/bluejay-node17 INFO   Event: BackOff pod redis-sessions-2-0 container redis reason=OOMKilled
2026-04-08T23:12:07Z sessionary-api-...k2xph ERROR redis pool read: connection reset by peer (target=redis-sessions-2.redis-sessions.lumenstack-prod.svc:6379)
2026-04-08T23:12:12Z sessionary-api-...nv88b WARN  retry 1/3 GET sess:live:acme-co:u_44218 after broken pipe
2026-04-08T23:12:15Z paygate-api-...q8l9v   WARN  redis SETNX retry 1/2 key=idem:9f7c2a11 err="connection reset by peer"
2026-04-08T23:12:21Z redis-sessions-2-0     INFO   Ready to accept connections
2026-04-08T23:12:22Z redis-sessions-2-0     INFO   Loading RDB produced by version 7.2.4
2026-04-08T23:12:22Z redis-sessions-2-0     INFO   RDB age 43 seconds
2026-04-08T23:12:29Z redis-sessions-2-0     INFO   DB loaded from disk: 6.88 seconds
2026-04-08T23:12:34Z sessionary-api-...k2xph WARN  pool waiters=14 in_use=64 max=64
2026-04-08T23:12:41Z paygate-api-...q8l9v   ERROR sessioncache.Reserve: context deadline exceeded after 750ms
2026-04-08T23:13:02Z sessionary-api-...k2xph ERROR SessionBackendUnavailable: Broken pipe tenant=acme-co
2026-04-08T23:13:18Z redis-sessions-2-0     WARN   Client id=118432 closed connection
2026-04-08T23:13:44Z kubelet/bluejay-node17 WARN   Liveness probe failed: HTTP probe failed with statuscode: 500
2026-04-08T23:14:01Z kubelet/bluejay-node17 INFO   Container redis failed liveness probe, will be restarted
2026-04-08T23:14:02Z redis-sessions-2-0     INFO   Ready to accept connections
2026-04-08T23:14:12Z sessionary-api-...nv88b ERROR pool exhausted; rejecting 17 pending requests
2026-04-08T23:14:37Z sessionary-api-...k2xph ERROR store.py:218 SessionBackendUnavailable (trace=3b91c07d)
2026-04-08T23:15:08Z redis-sessions-2-1     WARN   replica-of redis-sessions-2-0 promoted; role=master
2026-04-08T23:15:44Z redis-sessions-2-1     INFO   used_memory_human:3.12G  used_memory_rss_human:3.30G
2026-04-08T23:16:02Z paygate-api-...q8l9v   WARN  circuit-breaker=DISABLED; continuing retries on redis-sessions-2
2026-04-08T23:16:33Z sessionary-api-...5mc4f ERROR pool exhausted in_use=64 waiters=58
2026-04-08T23:17:11Z redis-sessions-2-1     WARN   OOM command not allowed when used memory > 'maxmemory'
2026-04-08T23:18:02Z paygate-api-...q8l9v   ERROR checkout.go:312 reserve timeout after 750ms
2026-04-08T23:19:06Z kubelet/bluejay-node09 INFO   Killing container "redis" as OOMKilled pod redis-sessions-2-1
2026-04-08T23:19:22Z sessionary-api-...k2xph ERROR redis pool exhausted; 37 requests rejected in last 60s
2026-04-08T23:20:14Z paygate-api-...q8l9v   ERROR SESSION_BACKEND_UNAVAILABLE propagated to /v1/checkout/start
2026-04-08T23:21:58Z sessionary-api-...nv88b WARN  half-open retry wave reached shard redis-sessions-2
2026-04-08T23:22:00Z paygate-api-...4r1w2   ERROR p99 checkout latency 8.4s
2026-04-08T23:23:44Z redis-sessions-1-0     WARN   used_memory_human:3.88G  used_memory_rss_human:3.94G
2026-04-08T23:25:12Z sessionary-api-...k2xph WARN  retry budget exhausted for /v1/session/get
2026-04-08T23:26:41Z kubelet/bluejay-node11 INFO   Killing container "redis" as OOMKilled pod redis-sessions-1-0
2026-04-08T23:27:09Z web-edge-...2g8ph      ERROR upstream 502 from sessionary-api (trace=a1bf...2244)
2026-04-08T23:28:55Z sessionary-api-...k2xph ERROR SessionBackendUnavailable rate=112/s
2026-04-08T23:31:02Z paygate-api-...q8l9v   ERROR all shards unhealthy; failing open to DENY
2026-04-08T23:33:17Z sessionary-api-...nv88b WARN  global error rate 8.1% (trace sample 1/200)
2026-04-08T23:36:41Z redis-sessions-1-1     INFO   used_memory_human:2.11G  used_memory_rss_human:2.40G
2026-04-08T23:38:55Z sessionary-api-...k2xph INFO  pool recovering in_use=41 waiters=2
2026-04-08T23:41:03Z web-edge-...2g8ph      WARN  peak error rate 8.1% reached
2026-04-08T23:43:12Z paygate-api-...q8l9v   INFO  p99 checkout latency dropping (4.2s)
2026-04-08T23:47:40Z sessionary-api-...k2xph INFO  pool nominal in_use=22 waiters=0
2026-04-08T23:52:14Z redis-sessions-2-1     INFO  used_memory_human:1.80G  used_memory_rss_human:2.02G
2026-04-08T23:57:33Z incident-bot           INFO  Incident #INC-2026-0408-01 declared by jae.tanaka (manual)
2026-04-09T00:04:02Z alertmanager           WARN  RedisMemoryPressureHigh firing for pod=redis-sessions-1-1
2026-04-09T00:18:51Z sessionary-api-...k2xph INFO  error rate 0.8% (recovering)
2026-04-09T00:42:10Z web-edge-...2g8ph      INFO  global error rate 0.42% — nominal
```

=== GRAFANA DASHBOARD DESCRIPTION ===

Dashboard: `redis-sessions // incident view // Apr 08 23:00 — Apr 09 00:30`.
Five panels, one metric per panel, aggregated at 1-minute resolution.
Reading the trajectory in 5-minute steps across the 90-minute window
(23:00 baseline through 00:30 recovery):

- redis_memory_used_bytes (shard 2 primary, bytes; limit = 4,294,967,296):
  - 23:00  3.63G
  - 23:05  3.71G
  - 23:10  3.89G
  - 23:15  (pod just restarted; loading RDB) 3.12G
  - 23:20  3.74G
  - 23:25  3.89G (shard 1 primary now also at 3.88G)
  - 23:30  shard 2 primary dead; shard 2 replica 2.95G
  - 23:35  shard 1 replica 2.41G
  - 23:40  shard 1 replica 1.92G
  - 23:45  shard 2 replica 1.77G
  - 23:50  shard 2 replica 1.80G
  - 23:55  shard 2 replica 1.79G
  - 00:00  shard 2 replica 1.78G
  - 00:05  shard 2 replica 1.78G
  - 00:10  shard 2 replica 1.79G
  - 00:15  shard 2 replica 1.80G
  - 00:20  shard 2 replica 1.81G
  - 00:25  shard 2 replica 1.81G
  - 00:30  shard 2 replica 1.82G

- redis_evicted_keys_total (shard 2 primary, monotonic counter, delta over 5m):
  - 23:00   0
  - 23:05   0
  - 23:10   0  (volatile-lru; no TTL'd keys under pressure)
  - 23:15   0  (pod just restarted)
  - 23:20   0
  - 23:25   0
  - 23:30   0
  - 23:35   0
  - 23:40   0
  - 23:45   0
  - 23:50   0
  - 23:55   0
  - 00:00   0
  - 00:05   0
  - 00:10   0
  - 00:15   0
  - 00:20   0
  - 00:25   0
  - 00:30   0
  (Confirmed: zero evictions throughout. This is the smoking gun.)

- redis_connected_clients (sum across all shards):
  - 23:00   408
  - 23:05   414
  - 23:10   419
  - 23:15   812  (retry storm)
  - 23:20   1104 (pool saturation across services)
  - 23:25   1389
  - 23:30   1501 (capped at max clients)
  - 23:35   1247
  - 23:40    889
  - 23:45    552
  - 23:50    460
  - 23:55    429
  - 00:00    421
  - 00:05    418
  - 00:10    414
  - 00:15    411
  - 00:20    409
  - 00:25    408
  - 00:30    408

- redis_command_duration_seconds p99 (shard 2 primary/successor, p99 in ms):
  - 23:00   2.1
  - 23:05   2.3
  - 23:10   2.4
  - 23:15   188
  - 23:20   612
  - 23:25   1411
  - 23:30   1980
  - 23:35   1604
  - 23:40    842
  - 23:45    318
  - 23:50    104
  - 23:55     41
  - 00:00     11
  - 00:05      4.8
  - 00:10      3.1
  - 00:15      2.6
  - 00:20      2.4
  - 00:25      2.3
  - 00:30      2.2

- redis_commands_processed_total (rate, ops/sec, all shards):
  - 23:00   28,440
  - 23:05   28,910
  - 23:10   29,200
  - 23:15   14,880 (shard 2 flapping)
  - 23:20    9,210
  - 23:25    5,402
  - 23:30    3,844
  - 23:35    6,121
  - 23:40   12,490
  - 23:45   19,220
  - 23:50   24,300
  - 23:55   26,910
  - 00:00   28,110
  - 00:05   28,500
  - 00:10   28,790
  - 00:15   28,880
  - 00:20   28,930
  - 00:25   29,000
  - 00:30   29,050

=== THE ALERT THAT FAILED TO FIRE (current, broken) ===

```promql
- alert: RedisMemoryPressureHigh
  expr: (avg_over_time(redis_memory_used_bytes[30m]) / redis_memory_max_bytes) > 0.80
  for: 5m
  labels:
    severity: warning
    team: platform-reliability
  annotations:
    summary: "Redis memory pressure above 80%"
    description: "Redis shard {{ $labels.pod }} memory usage has been above 80% of limit."
```

What it SHOULD be (at minimum — we can debate the exact window in a later
turn):

```promql
- alert: RedisMemoryPressureHigh
  expr: (max_over_time(redis_memory_used_bytes[2m]) / redis_memory_max_bytes) > 0.85
  for: 2m
  labels:
    severity: critical
    team: platform-reliability
  annotations:
    summary: "Redis shard {{ $labels.pod }} memory > 85% of pod limit"
    description: "Sustained high memory on Redis shard {{ $labels.pod }}. OOMKill imminent."
```

=== SERVICE DEPENDENCY DIAGRAM (partial, ASCII) ===

```
                 +----------------+
                 |    web-edge    |
                 |  (envoy/edge)  |
                 +-------+--------+
                         |
          +--------------+---------------+
          |              |               |
          v              v               v
   +------+----+  +------+------+  +-----+------+
   | sessionary|  | paygate-api |  | cart-svc   |
   |  -api     |  |             |  |            |
   +-----+-----+  +------+------+  +-----+------+
         |               |               |
         +-------+-------+               |
                 |                       |
                 v                       v
         +-------+-------+       +-------+-------+
         | redis-sessions|       |   postgres    |
         |   (3 shards)  |       |  (authorit.)  |
         +---------------+       +---------------+

        (session-shadow-sync is a k8s CronJob that writes to
         redis-sessions every 4h; not shown as a live caller.)
```

=== TASK FOR THIS TURN ===

Draft the initial Incident Summary and Timeline sections of the postmortem.
Include exact timestamps from the logs and metrics above, the user-facing
impact, and your first plausible hypothesis for root cause (state it as a
hypothesis, not a conclusion).
"""


FOLLOW_UPS: list[str] = [
    # 1. Five-whys RCA
    (
        "Now do the Root Cause Analysis section using the five-whys framework, "
        "starting from the OOMKill event at 23:12:04 on redis-sessions-2-0. For "
        "each 'why' layer, anchor your claim in a specific log line, metric "
        "reading, or config field from earlier in this conversation — for "
        "example, the zero redis_evicted_keys_total delta across the entire "
        "90-minute window, the `maxmemory-policy` flip from `allkeys-lru` to "
        "`volatile-lru` in PR #4412, the fact that sess:shadow:{tenant_id}:{user_id} "
        "keys were written without TTL, the session-shadow-sync cleanup "
        "goroutine that stopped running after PR #4287, and the 30-minute "
        "`avg_over_time` window on RedisMemoryPressureHigh. The chain should "
        "bottom out at the Project Sandstone sprint decision to change "
        "maxmemory-policy, the unverified assumption that all keys had TTLs, "
        "and the alert-rule window change. Be specific about the boundary "
        "conditions that turned a reasonable change into an incident — why it "
        "stayed latent for three weeks and what tipped it over last night at "
        "23:11. I want to see at least five 'why' layers, and I want the last "
        "two to be organizational/process, not just technical."
    ),
    # 2. Python code fix for sessionary
    (
        "Propose the code fix for the Python sessionary service. Show the "
        "complete updated Redis client initialization — `SessionStore.__init__`, "
        "the connection pool construction, and the `get_session` / `put_session` "
        "methods — including TTL policies (every key the service writes must "
        "have an explicit TTL, enforced at the wrapper layer, not trusted to "
        "callers), circuit breaker (open/half-open/closed with explicit "
        "thresholds justified against the pool exhaustion we saw at 23:14:12 "
        "with in_use=64 waiters=58), connection pool sizing (the current "
        "max=64 was clearly wrong; pick a number and justify it against the "
        "1104 sum-across-shards client count peak at 23:20), retry budget "
        "(token bucket, explicit max), and command timeouts (socket_timeout "
        "and socket_connect_timeout — the redis-py defaults are too generous "
        "for a session store backing payment checkout). Every choice must be "
        "justified against a specific failure mode you identified in the RCA. "
        "Full code, not a sketch — I should be able to paste it into "
        "sessionary/session_store.py and have it compile."
    ),
    # 3. Go code fix for paygate-api
    (
        "Propose the code fix for the Go paygate-api service. Same "
        "requirements — complete code, every choice justified against a "
        "specific failure mode. Specifically: show the full "
        "`internal/sessioncache/client.go` with the new `Client` struct, "
        "`NewClient` constructor, and `Reserve` method that replaces the one "
        "that blew up at checkout.go:312 with `context deadline exceeded after "
        "750ms`. Use go-redis/v9, include a `context.WithTimeout` budget that "
        "is shorter than the upstream HTTP handler budget, a circuit breaker "
        "(sony/gobreaker or hand-rolled, your call — justify), a retry policy "
        "that respects idempotency (Reserve is SETNX on an idempotency key — "
        "you CANNOT blindly retry), and explicit pool configuration on the "
        "`redis.Options` struct. Flag any places where the Python sessionary "
        "fix and this Go paygate-api fix must coordinate at deploy time — e.g. "
        "both must be rolled before any future Redis `maxmemory-policy` change, "
        "or both must share the same TTL convention on the sess:* keyspace, "
        "or both must agree on a circuit-breaker trip threshold so one service "
        "doesn't keep slamming Redis while the other has backed off. Be "
        "explicit about the ordering: what ships first, what ships second, "
        "and what breaks if we ship them in the wrong order."
    ),
    # 4. Remediation plan
    (
        "Write the remediation plan in three horizons: immediate mitigation "
        "(day 0, meaning tonight and tomorrow morning), short-term fix "
        "(week 1), and structural fix (this quarter). For each horizon, "
        "include the specific action, the owner role (SRE on-call, platform "
        "eng lead, etc. — do not name individuals), the rollback criteria "
        "(what signal tells you 'undo this change right now'), and the "
        "verification signal that tells you it actually worked. Reference the "
        "specific metric thresholds from the Grafana dashboard in the seed — "
        "for example, 'redis_connected_clients sum-across-shards must stay "
        "under 600 under normal load (baseline was 408 at 23:00)', or "
        "'redis_command_duration_p99 must return to under 5ms within 10 "
        "minutes of a shard restart (we saw it stuck at 1980ms at 23:30 "
        "during the cascade)'. Day-0 actions must be rollback-safe and "
        "verifiable within one on-call shift. Quarter actions should tie back "
        "to the Project Sandstone decision path that put us here."
    ),
    # 5. Prometheus alert rules
    (
        "Write the updated Prometheus alert rules. Not just the corrected "
        "version of RedisMemoryPressureHigh (the one that didn't fire until "
        "00:04 because of the 30-minute `avg_over_time` window) — also two "
        "new alerts that would have paged earlier in the failure sequence. "
        "One of those new alerts should have fired around 23:12 when the OOM "
        "actually happened, and the other should have fired in the earlier "
        "days/weeks of the incubation period so we'd have caught the shadow-"
        "key accumulation before it ever reached the pod limit. For all three "
        "rules, include full PromQL, alert annotations (summary + description "
        "with template variables), runbook link (you can invent the URL "
        "slug), severity label, and routing labels (team, tier). For each, "
        "explain why the chosen window and threshold are the right size — "
        "i.e., why this window won't be too noisy during nightly backup jobs "
        "the way the original 5-minute window supposedly was, and why this "
        "threshold won't miss a 4-minute memory spike the way the 30-minute "
        "average did. Include the recording rules you'd add to support them "
        "if that makes the alert expressions cleaner."
    ),
    # 6. Runbook entry
    (
        "Write the runbook entry the on-call engineer would have used if "
        "those new alerts had fired at 3 AM on a holiday — i.e., the person "
        "paged is tired, possibly junior, and alone. Must be executable "
        "step-by-step with specific `kubectl` commands (namespace "
        "`lumenstack-prod`, pod label selectors for the redis-sessions "
        "StatefulSet, the sessionary-api Deployment, and the paygate-api "
        "Deployment), specific `redis-cli` commands (how to connect, how to "
        "check `INFO memory`, how to scan for no-TTL keys using "
        "`redis-cli --scan --pattern 'sess:shadow:*'` followed by a TTL "
        "probe, how to check `CLIENT LIST`), and specific PromQL to paste "
        "into Grafana to confirm the metrics from the seed dashboard. "
        "Reference the exact service names (sessionary-api, paygate-api, "
        "redis-sessions), the `redis-sessions-{shard}-{replica}` pod naming "
        "scheme, and the cluster name (bluejay). Include a decision tree for "
        "the three most likely root-cause branches: (a) shadow-key "
        "accumulation again, (b) a legitimate traffic spike saturating the "
        "pool, (c) a bad deploy that changed Redis call patterns in "
        "sessionary or paygate-api. For each branch, say what command "
        "distinguishes it from the others and what to do next."
    ),
    # 7. Stakeholder comms
    (
        "Draft the stakeholder communication. First, the status-page update "
        "we WOULD have posted from during the incident at 23:35 (10 minutes "
        "after shard 1 OOMKill at 23:26:41, well before Jae manually declared "
        "at 23:57) — must be 2 paragraphs, public-facing, honest about impact "
        "(reference the 8.1% peak error rate on web-edge and the 20-minute "
        "session blip from 23:22 through 23:42) without exposing internal "
        "detail like 'maxmemory-policy' or PR #4412 or the fact that our "
        "alert was broken. Second, the post-incident customer email we'll "
        "send tomorrow — 1 page, apologetic, technical enough to rebuild "
        "trust with an engineering audience (reference that we use Redis for "
        "session storage, that a configuration change three weeks ago "
        "interacted badly with a background job's data pattern, that our "
        "monitoring window was too coarse to catch a 4-minute spike, and "
        "that we are shipping the fixes you described above). Use only "
        "numbers and details already established in the conversation; do not "
        "invent new facts. Sign the status page update as 'Lumenstack "
        "Platform Reliability' and the email as 'Priya Ostroff, Director of "
        "Platform Reliability, Lumenstack' (matching the people already "
        "named in the seed)."
    ),
    # 8. Lessons Learned
    (
        "Write the complete Lessons Learned section, organized by category: "
        "prevention (what avoids this entirely), detection (what finds it "
        "faster), response (what shortens recovery), and process "
        "(organizational/cultural changes that prevent the decision path "
        "that led here — specifically, the Project Sandstone sprint "
        "bundling a maxmemory-policy flip and an alert-window relaxation "
        "into one PR #4412 that nobody caught on review). Each lesson must "
        "tie back to a concrete action item with an owner role (not an "
        "individual), acceptance criteria (how you know the action is done "
        "and working — reference real signals like "
        "redis_evicted_keys_total staying nonzero under sustained load, or "
        "RedisMemoryPressureHigh firing within 3 minutes of a simulated "
        "memory spike in staging), and the earliest-acceptable close date "
        "(use relative dates like 'end of week 1', 'end of Q2', not real "
        "calendar dates). Include a short 'what we are choosing not to do "
        "and why' list — for example, we could move the session store off "
        "Redis entirely, or add a second cache layer, or ban `volatile-*` "
        "eviction policies across the whole org, and for each one explain "
        "the tradeoff and why we're not doing it right now. This section "
        "should read like the final, authoritative thing the VP of "
        "Engineering will sign off on."
    ),
]


def estimate_tokens() -> None:
    """Print rough token estimates for the seed and each follow-up."""
    seed_estimate = len(SEED_PROMPT.split()) * 1.3
    print(f"SEED_PROMPT: ~{seed_estimate:.0f} tokens")
    for idx, text in enumerate(FOLLOW_UPS, start=1):
        est = len(text.split()) * 1.3
        print(f"FOLLOW_UPS[{idx}]: ~{est:.0f} tokens")


if __name__ == "__main__":
    estimate_tokens()
