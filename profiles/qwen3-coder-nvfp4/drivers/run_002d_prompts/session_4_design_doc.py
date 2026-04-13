"""Run 002d -- Session 4 prompts: multi-turn ADR/RFC evolution.

Scenario: A principal architect at a fictional B2B healthcare SaaS company
("Medrel Health Systems") is evolving an ADR proposing a centralized
feature-flag rollout platform (codename ``flagforge``). The company operates
under HIPAA and HITRUST compliance requirements. Three existing ad-hoc flag
libraries need consolidation after a near-miss incident on 2026-01-14 where
a flag toggle exposed duplicated insulin pump reorder line items.

All names, companies, authors, dates, and technical details are fictional.
Do not map these to any real organization.

Token counts reported by ``estimate_tokens()`` use ``len(text) // 4``
(approximate lower bound for BPE token count on mixed code/prose).
"""

from __future__ import annotations

SEED_TOPIC: str = (
    "ADR evolution: centralized feature-flag platform (flagforge) "
    "for HIPAA-regulated B2B healthcare SaaS"
)


# ---------------------------------------------------------------------------
# SEED PROMPT
# ---------------------------------------------------------------------------

SEED_PROMPT: str = r'''You are helping me evolve an Architecture Decision Record (ADR) for our company. Below is the complete draft of ADR-047. Read it carefully, then evaluate its completeness: identify gaps in the compliance story, weaknesses in the proposed architecture, and areas where the alternatives analysis is insufficient. Do not rewrite the ADR yet -- just provide your analysis.

=== ADR-047: Centralized Feature-Flag Rollout Platform ("flagforge") ===

Status: Proposed
Date: 2026-02-03
Authors:
  - Priya Ramanathan, Principal Architect
  - Marcus Ojukwu, Staff Engineer (Platform)
  - Helena Vogt, Security Lead

---

## 1. Context

Medrel Health Systems is a B2B healthcare SaaS company serving 240+ hospital networks and specialty clinics across the United States. Our platform manages patient intake workflows, medication ordering (including controlled substances and insulin pump supplies), clinical trial enrollment, and insurance pre-authorization. We operate under HIPAA (45 CFR Parts 160 and 164) and are pursuing HITRUST CSF r11.3 certification (target: Q4 2026).

Our backend is polyglot:
  - **Python (Django 4.2)**: Patient intake, medication management, clinical workflows. ~180 microservices. Deployed on Kubernetes (EKS, us-east-1 and us-west-2).
  - **Go 1.22**: Real-time event processing, notification dispatch, audit log aggregation. ~95 microservices. Same EKS clusters.
  - **TypeScript (Node 20 + React 18)**: Backend-for-Frontend (BFF) layer, clinician-facing dashboard, patient portal. ~65 services (BFF) + 3 SPAs.

We currently have three independent, ad-hoc feature flag systems:

### 1.1 medrel-pyflags (Python)
- Django-specific. Reads flags from a YAML config file baked into each Docker image at build time.
- No runtime toggle capability -- changing a flag requires a full redeploy.
- No audit trail. No access control. Any developer with repo access can change any flag.
- Covers ~47 flags across 12 Django services (medication ordering, intake forms, lab integration).
- Maintained by the Medication Platform team. Last meaningful update: 2025-08.

### 1.2 gflagon (Go)
- Environment-variable-based. Each Go service reads its own `GFLAGON_*` env vars at startup.
- Supports runtime reload via SIGHUP, but only for the individual pod -- no fleet-wide propagation.
- No central management UI. Flag changes require updating Kubernetes ConfigMaps and rolling pods.
- No audit logging beyond what Kubernetes records in its audit log (which is noisy and not indexed for flag queries).
- Covers ~23 flags across 8 Go services (event processing, notification routing, audit pipeline).
- Maintained by the Platform Infrastructure team. Single maintainer (Marcus Ojukwu).

### 1.3 @medrel/flagsmith-local (TypeScript)
- Wraps a locally forked Flagsmith 2.x instance running as a sidecar in each BFF pod.
- Has partial audit logging: records flag evaluations to a local SQLite file per pod (not aggregated, not queryable, not backed up).
- The Flagsmith fork is 14 months behind upstream. Two CVEs (CVE-2025-31842, CVE-2025-33109) are unpatched in our fork.
- No HIPAA-compliant storage -- the SQLite audit files contain patient-context identifiers (tenant_id, anonymized user hashes) that constitute ePHI under our data classification policy.
- Covers ~12 flags across 4 BFF services and the clinician dashboard.
- Maintained by the Frontend Platform team. Original author left the company in 2025-11.

### 1.4 The diab_autoreorder_v2 Incident (2026-01-14)

On January 14, 2026, at approximately 09:47 EST, a junior developer on the Medication Platform team toggled the feature flag `diab_autoreorder_v2` from `false` to `true` in the `medrel-pyflags` YAML configuration for the `med-ordering-api` service. This flag controlled a new version of the diabetes auto-reorder workflow that had been developed by a different team (Chronic Care Management) and was not yet approved for production by the Clinical Safety Review Board.

The flag had been added to the `med-ordering-api` config file during a cross-team code merge three weeks earlier. The junior developer was working on an unrelated flag (`intake_v3_ab_test`) in the same YAML file and inadvertently toggled `diab_autoreorder_v2` while resolving a merge conflict. Because `medrel-pyflags` has no access control, no confirmation prompt, and no pre-deploy validation, the change was merged, built, and deployed through the standard CI/CD pipeline without any additional review gate.

The result: for approximately 18 minutes (09:47 to 10:05 EST), 14% of insulin pump auto-reorder requests processed by `med-ordering-api` were submitted with duplicated line items. Specifically, the v2 workflow double-counted the "reservoir refill" line item when the patient's insurance plan required a prior authorization step. This affected 23 patient orders across 7 hospital networks.

The issue was caught at 10:05 by a manual spot-check from the Pharmacy Operations team, not by any automated system. A member of the Pharmacy Ops team noticed that a batch of orders had unusually high quantities and escalated to the on-call engineer, who identified the flag change in the deployment log and rolled back the `med-ordering-api` image to the previous version.

No patient harm occurred: the duplicated line items were caught before any orders were fulfilled, and the affected hospital pharmacies were notified within 2 hours. However, the incident triggered an internal HIPAA compliance review (ICR-2026-003) because:
  1. The ordering system processed clinically significant data (insulin pump supplies) based on an unauthorized configuration change.
  2. No audit trail existed to determine who changed the flag, when, or why.
  3. The flag's scope (production, all tenants) was not gated by any role-based access control.

The compliance review concluded on 2026-01-28 with a finding that our current feature flag systems do not meet the access control and audit requirements of HIPAA Security Rule sections 164.312(a)(1) (Access Control), 164.312(b) (Audit Controls), and 164.312(d) (Person or Entity Authentication). The review recommended that Medrel implement a centralized, HIPAA-compliant feature flag platform with role-based access control, environment-scoped permissions, and comprehensive audit logging.

This ADR is the direct response to that recommendation.

## 2. Decision

We will build **flagforge**, a centralized feature-flag platform, as an internal service owned by the Platform Infrastructure team.

### 2.1 Core Components

- **Flag Evaluation API**: gRPC + REST (OpenAPI 3.1) service. Handles flag resolution requests from SDKs. Stateless, horizontally scalable. Deployed as a Kubernetes Deployment with HPA (target: p99 latency < 5ms for cached evaluations, < 50ms for uncached).

- **SDK Wrappers**: Thin client libraries for Python, Go, and TypeScript. Each SDK:
  - Maintains a local cache of flag state (configurable TTL, default 30s).
  - Falls back to a hardcoded default value if the API is unreachable (circuit breaker pattern).
  - Emits structured evaluation events to a local buffer, flushed to the Audit Ingestion Pipeline every 10s.

- **Admin UI**: React SPA for flag management. Features: create/edit/archive flags, configure targeting rules, view audit log, manage RBAC roles. Authentication via Medrel SSO (SAML 2.0 + OIDC).

- **PostgreSQL Audit Store**: All flag changes, evaluations, and access events are written to a dedicated PostgreSQL 16 cluster (RDS, Multi-AZ). Encrypted at rest (AES-256) and in transit (TLS 1.3). Retention: 7 years (per HIPAA requirement). Separate from the application database to prevent cross-contamination of access patterns.

- **Redis Cache Layer**: Redis 7.2 cluster (ElastiCache) for flag state caching. The API reads from Redis first; on miss, reads from PostgreSQL and populates cache. Cache invalidation via Redis Pub/Sub: when a flag is updated via the Admin UI, a NOTIFY event is published and all API instances invalidate their local cache entry.

- **Audit Ingestion Pipeline**: Kinesis Data Stream -> Lambda -> PostgreSQL. Receives evaluation events from SDKs. Designed for eventual consistency (evaluation audit records may lag by up to 60s). Separate from the flag state path to avoid coupling evaluation latency to audit write latency.

### 2.2 Deployment Topology

```
                    +-----------------+
                    |   Admin UI      |
                    |  (React SPA)    |
                    +--------+--------+
                             |
                             v
                    +--------+--------+
                    | Flag Eval API   |
                    | (gRPC + REST)   |
                    | K8s Deployment  |
                    | HPA: 3-12 pods  |
                    +---+--------+----+
                        |        |
              +---------+        +---------+
              v                            v
     +--------+--------+         +--------+--------+
     |  Redis Cache     |         | PostgreSQL 16   |
     |  (ElastiCache)   |         | (RDS Multi-AZ)  |
     |  3-node cluster  |         | Audit Store     |
     +-----------------+          +-----------------+
                                          ^
                                          |
                                  +-------+-------+
                                  | Kinesis Stream |
                                  | + Lambda       |
                                  | Audit Ingest   |
                                  +-------+-------+
                                          ^
                                          |
                      +-------------------+-------------------+
                      |                   |                   |
              +-------+------+   +-------+------+   +-------+------+
              | Python SDK   |   | Go SDK       |   | TS SDK       |
              | (Django)     |   | (services)   |   | (BFF + SPA)  |
              | 180 services |   | 95 services  |   | 65 services  |
              +--------------+   +--------------+   +--------------+
```

### 2.3 RBAC Model

| Role | Permissions |
|------|------------|
| Flag Admin | Create, edit, archive flags in any environment. Manage targeting rules. View full audit log. |
| Environment Owner | Edit flags only in their assigned environment(s). Cannot create or archive. |
| Deployer | Toggle flags on/off in staging and production. Cannot edit targeting rules. |
| Viewer | Read-only access to flag state and audit log. No mutation permissions. |
| Auditor | Read-only access to the full audit log, including evaluation records. Can export audit data. Cannot view or modify flag state. |

Environment scopes: `development`, `staging`, `production-us-east`, `production-us-west`.

Every mutation (flag create, edit, toggle, archive) requires:
  1. Authentication via Medrel SSO.
  2. Authorization check against the RBAC role + environment scope.
  3. A mandatory "change reason" field (free text, minimum 20 characters).
  4. An audit record written synchronously to PostgreSQL before the mutation is applied.

### 2.4 Compliance Mapping

| HIPAA Section | Requirement | flagforge Control |
|--------------|-------------|-------------------|
| 164.312(a)(1) | Access Control | RBAC with environment-scoped permissions. SSO-only authentication. |
| 164.312(b) | Audit Controls | Synchronous audit logging of all mutations. Eventual-consistency logging of all evaluations. 7-year retention. |
| 164.312(d) | Person or Entity Authentication | SAML 2.0 + OIDC via Medrel SSO. No shared credentials. |
| 164.312(e)(1) | Transmission Security | TLS 1.3 for all API and SDK communication. mTLS between API and SDKs in production. |

| HITRUST CSF Control | flagforge Coverage |
|--------------------|--------------------|
| 01.b User Registration | SSO integration, no local accounts |
| 01.v Information Access Restriction | Environment-scoped RBAC |
| 09.aa Audit Logging | Synchronous mutation audit + eventual evaluation audit |
| 09.ab Monitoring System Use | Admin UI activity dashboard, anomaly detection on flag toggle frequency |
| 09.ad Administrator and Operator Logs | All Admin UI actions logged with user identity, timestamp, change reason |

## 3. Consequences

### 3.1 Positive
- Single source of truth for all feature flags across Python, Go, and TypeScript services.
- HIPAA-compliant audit trail for all flag mutations and evaluations.
- RBAC prevents unauthorized flag changes (addresses the root cause of the diab_autoreorder_v2 incident).
- Runtime flag toggles without redeployment (eliminates the current redeploy requirement in medrel-pyflags).
- Fleet-wide propagation via cache invalidation (eliminates the per-pod SIGHUP limitation in gflagon).
- Eliminates the unpatched Flagsmith fork and its associated CVE exposure.

### 3.2 Negative
- New infrastructure to operate: Redis cluster, PostgreSQL audit store, Kinesis pipeline, Flag Eval API.
- Migration effort: 82 flags across 3 systems, 340+ services. Estimated 3-4 months.
- SDK dependency: all services now depend on the flagforge SDK. If the SDK has a bug, it affects the entire fleet.
- Latency budget: flag evaluation adds a network hop (mitigated by local caching, but cache misses add ~50ms).

## 4. Alternatives Considered

### 4.1 LaunchDarkly (SaaS)
- Pro: Mature, HIPAA BAA available, SOC 2 Type II certified.
- Con: $180K/year at our scale (340 services, 12M evaluations/day). Data residency concerns -- evaluation data transits LaunchDarkly infrastructure (US-based, but not in our VPC). Vendor lock-in on SDK and API surface.
- Decision: Rejected due to cost and data residency requirements from the compliance team.

### 4.2 Unleash (self-hosted open source)
- Pro: Open source (Apache 2.0). Self-hosted, so data stays in our VPC.
- Con: No built-in HIPAA audit trail. RBAC is basic (no environment-scoped permissions). Would require significant customization (~2 months estimated) to meet our compliance requirements. Community edition lacks enterprise features (A/B testing, stale flag detection). Enterprise license: $60K/year.
- Decision: Rejected because the customization effort approaches the cost of building flagforge, without the benefit of full control.

### 4.3 Extend the Flagsmith Fork
- Pro: We already have a running Flagsmith instance (sort of). The @medrel/flagsmith-local fork covers the TypeScript ecosystem.
- Con: Fork is 14 months behind upstream. Two unpatched CVEs. Single-maintainer bus factor (original author left). SQLite audit storage is not HIPAA-compliant. Extending to support Python and Go SDKs would require porting the Flagsmith client protocol, which is underdocumented.
- Decision: Rejected. The fork is a liability, not an asset. Continuing to invest in it increases risk.

## 5. Open Questions

1. **SDK update mechanism**: How do we ensure all 340 services pick up SDK updates? Pin to a specific version and use Renovate? Or use a shared library that is always at HEAD?
2. **Percentage rollout stickiness**: For gradual rollouts, how is user stickiness guaranteed? Hash(user_id + flag_key) mod 100? What happens when the denominator changes?
3. **Flag dependencies**: Can flags depend on other flags? (e.g., "enable feature X only if feature Y is also enabled"). If so, how do we prevent circular dependencies?
4. **Disaster recovery**: What is the RTO/RPO for the Flag Eval API? If PostgreSQL is down, do we serve stale cache? For how long?
5. **Multi-region consistency**: Flags are updated in us-east-1 Admin UI. How quickly do they propagate to us-west-2? Is eventual consistency acceptable for safety-critical flags?

---

END OF ADR-047

Your analysis should cover: (a) gaps in the compliance story that would concern Helena Vogt, (b) architectural weaknesses that Marcus Ojukwu should address before the architecture review board, (c) areas where the alternatives analysis is biased or incomplete, and (d) open questions that should be answered before this ADR is approved. Be specific -- cite section numbers and control IDs where applicable.'''


# ---------------------------------------------------------------------------
# FOLLOW-UPS
# ---------------------------------------------------------------------------

FOLLOW_UPS: list[str] = [
    # 1. Evaluation pipeline
    (
        "Design the flag evaluation pipeline in detail. Walk through the "
        "complete request flow: SDK call originates in a Django view (Python "
        "SDK), hits the local cache (TTL-based, configurable per flag), on "
        "cache miss makes a gRPC call to the Flag Eval API, which checks "
        "Redis, on Redis miss reads from PostgreSQL, resolves the flag value "
        "using targeting rules (user attributes, tenant ID, environment, "
        "percentage rollout bucket), and returns the result. For each hop, "
        "specify the data format (protobuf message for gRPC, JSON for REST), "
        "the error handling (what happens if Redis is down? if PostgreSQL is "
        "down? if the API pod is OOM-killed mid-request?), and the latency "
        "budget. Include the PostgreSQL schema for the flag definitions table "
        "and the targeting rules table -- show the CREATE TABLE statements "
        "with column types, constraints, and indexes. Include the Redis key "
        "schema and the pub/sub channel naming convention. Show a sequence "
        "diagram for the cache-miss path."
    ),
    # 2. State transitions
    (
        "Define the complete flag lifecycle state machine. The states from "
        "the ADR are Draft, Active, Percentage Rollout, Full Rollout, "
        "Deprecated, and Archived. For each state transition, specify: "
        "(a) which RBAC roles can trigger it, (b) what validations run before "
        "the transition is allowed (e.g., a flag cannot move from Draft to "
        "Active without at least one targeting rule defined; a flag cannot be "
        "Archived if any service evaluated it in the last 30 days), (c) what "
        "audit events are emitted (event type, payload schema, destination), "
        "and (d) what happens to in-flight evaluations during the transition "
        "(e.g., if a flag moves from Percentage Rollout to Full Rollout, do "
        "all cached partial-rollout values immediately expire, or do they "
        "drain naturally?). Pay special attention to the Percentage Rollout "
        "state: explain how stickiness is guaranteed using a deterministic "
        "hash of (user_id, flag_key, rollout_salt), how the rollout "
        "percentage can be increased but never decreased without resetting "
        "the salt, and how you handle the HIPAA requirement that a specific "
        "patient's feature exposure history must be retrievable for audit "
        "purposes for 7 years. Show the state machine as an ASCII diagram."
    ),
    # 3. Capacity model
    (
        "Build a capacity model for flagforge serving Medrel's production "
        "traffic: 12 million flag evaluations per day across 340 microservices "
        "in two AWS regions (us-east-1 and us-west-2, 60/40 traffic split). "
        "Account for: (a) SDK-side caching -- at a 30s default TTL, how many "
        "cache misses per second hit the Flag Eval API? Assume each service "
        "has 3 pods on average and evaluates 5 distinct flags per request. "
        "(b) Flag Eval API sizing -- how many pods, what CPU/memory requests, "
        "what HPA thresholds? (c) Redis cluster sizing -- how many nodes, "
        "what instance type, what is the memory footprint for 82 flags with "
        "targeting rules averaging 2KB per flag? (d) PostgreSQL read replica "
        "count -- the audit store receives evaluation events via Kinesis at "
        "~139 writes/second sustained; what is the write IOPS requirement, "
        "and do we need read replicas for the Admin UI audit log queries? "
        "(e) Kinesis shard count -- at 12M events/day with an average payload "
        "of 500 bytes, how many shards? (f) What is the p99 latency budget "
        "for a flag evaluation on the cache-hit path vs. cache-miss path? "
        "(g) What happens when a new flag is created that applies to all 340 "
        "services simultaneously -- model the thundering herd on the Redis "
        "cache invalidation channel."
    ),
    # 4. Security review
    (
        "Conduct a security review of the flagforge architecture as Helena "
        "Vogt would. Assess the following threat vectors and for each one, "
        "identify the current mitigation in the ADR (if any), the residual "
        "risk, and your recommended additional controls: (a) Authentication "
        "between SDKs and the Flag Eval API -- the ADR mentions mTLS in "
        "production but does not specify how certificates are provisioned, "
        "rotated, or revoked. What happens if a pod's certificate is "
        "compromised? (b) Secrets management for the PostgreSQL audit "
        "database connection string -- is it in a Kubernetes Secret, AWS "
        "Secrets Manager, or HashiCorp Vault? How is access to that secret "
        "scoped? (c) Blast radius of a compromised SDK credential -- if an "
        "attacker gains access to the Python SDK's mTLS certificate, can "
        "they read all flags for all tenants? Can they write flags? Can they "
        "access the audit log? (d) Flag poisoning -- an adversary with "
        "Deployer role toggles a flag that controls PHI visibility, exposing "
        "patient data to unauthorized staff. How does the RBAC model prevent "
        "this? Is the 'change reason' field sufficient, or do safety-critical "
        "flags need a two-person approval workflow? (e) Supply chain risk -- "
        "the SDKs are distributed as internal packages. What prevents a "
        "malicious insider from publishing a backdoored SDK version? "
        "Reference specific HITRUST CSF control IDs (01.b, 01.v, 09.aa, "
        "09.ab, 09.ad, 10.a, 10.m) where applicable."
    ),
    # 5. Migration plan
    (
        "Write the migration plan from the three existing flag libraries to "
        "flagforge. The migration must be zero-downtime, reversible at each "
        "stage, and must not require all 340 services to migrate "
        "simultaneously. Define the following: (a) Migration phases -- Phase "
        "1: deploy flagforge infrastructure (API, Redis, PostgreSQL, Admin "
        "UI). Phase 2: shadow mode -- SDKs read from both old and new "
        "systems, compare results, log discrepancies, but serve from old. "
        "Phase 3: cutover -- SDKs serve from flagforge, fall back to old on "
        "error. Phase 4: cleanup -- remove old flag libraries, decommission "
        "Flagsmith fork. For each phase, specify the duration, the success "
        "criteria for advancing to the next phase, and the rollback "
        "procedure. (b) The dual-read strategy in detail: how does the "
        "Python SDK simultaneously read from `medrel-pyflags` YAML and "
        "flagforge API? How do you handle the case where a flag exists in "
        "the old system but not yet in flagforge? (c) A per-service "
        "migration checklist that a team can follow independently. (d) The "
        "flag inventory migration: who creates the 82 existing flags in "
        "flagforge, who maps the old flag names to new names, and who "
        "validates that the targeting rules match? Reference the specific "
        "flag counts from the ADR: 47 in medrel-pyflags, 23 in gflagon, "
        "12 in @medrel/flagsmith-local."
    ),
    # 6. Rollout plan
    (
        "Write the production rollout plan for flagforge itself. This is "
        "meta-interesting: since flagforge manages feature flags, how do you "
        "feature-flag the feature-flag system without a circular dependency? "
        "Define: (a) The bootstrap sequence -- flagforge cannot use itself "
        "for its own rollout, so what mechanism controls the initial "
        "deployment? (Kubernetes feature gates? A hardcoded environment "
        "variable? A separate, minimal flag service?) (b) The canary "
        "deployment strategy -- which 3-5 services will be in the first "
        "migration cohort and why? Consider services that are non-critical, "
        "have good test coverage, and are owned by teams willing to be early "
        "adopters. (c) The smoke tests that must pass before each rollout "
        "phase (flag evaluation latency < 5ms p99, cache hit rate > 95%, "
        "audit log write latency < 100ms p99, zero flag evaluation errors). "
        "(d) The monitoring signals that trigger automatic rollback -- if "
        "the Flag Eval API error rate exceeds 0.1% for 5 consecutive "
        "minutes, roll back the most recent canary batch. (e) The "
        "communication plan: who is notified at each phase, what Slack "
        "channels, what status page updates. Reference the specific "
        "services, teams, and infrastructure from the ADR."
    ),
    # 7. Alternative re-evaluation
    (
        "Now that you have worked through the detailed design (evaluation "
        "pipeline, state machine, capacity model, security review, migration "
        "plan, and rollout plan), revisit the three alternatives from the "
        "ADR: LaunchDarkly, Unleash, and extending the Flagsmith fork. For "
        "each alternative, produce an updated comparison against flagforge "
        "across these dimensions: (a) HIPAA compliance readiness -- which "
        "controls are covered out of the box, which require customization, "
        "and which are impossible? (b) HITRUST CSF control coverage -- "
        "map each alternative to the same control IDs from ADR section 2.4. "
        "(c) Total cost of ownership over 3 years including engineering "
        "time -- use $200K/year loaded cost per engineer as the baseline; "
        "flagforge requires 2 FTEs for 4 months to build + 0.5 FTE ongoing; "
        "how do LaunchDarkly and Unleash compare? (d) Migration effort from "
        "the three existing systems. (e) Vendor lock-in risk and exit cost. "
        "Has your opinion changed after the detailed design work? If the "
        "architecture review board asked you to justify building over buying, "
        "what is the single strongest argument for each side? Be honest -- "
        "if LaunchDarkly is actually the better choice, say so."
    ),
    # 8. Final complete document
    (
        "Produce the final, complete ADR-047 incorporating everything from "
        "turns 1 through 7. This document will go to the architecture review "
        "board for approval. It must include: (a) the original Context and "
        "Decision sections, revised based on our analysis (update the "
        "deployment topology if the capacity model revealed sizing issues, "
        "update the RBAC model if the security review identified gaps), "
        "(b) a new 'Detailed Design' section with the evaluation pipeline, "
        "state machine, PostgreSQL schema, Redis schema, and sequence "
        "diagrams, (c) a new 'Capacity Model' section with the sizing "
        "numbers and latency budgets, (d) a new 'Security Assessment' "
        "section with Helena Vogt's findings and recommended mitigations, "
        "(e) a revised 'Migration Plan' section, (f) a revised 'Alternatives "
        "Considered' section with the updated comparison matrix, (g) a new "
        "'Risks' section cataloging the top 5 risks with likelihood, impact, "
        "and mitigation strategy, and (h) a new 'Timeline' section with "
        "quarterly milestones for 2026: Q1 (infrastructure + SDK alpha), Q2 "
        "(shadow mode + first cohort migration), Q3 (fleet-wide cutover), Q4 "
        "(cleanup + HITRUST audit). The final document should be "
        "self-contained: a reader who has not seen our conversation should "
        "be able to understand and evaluate the proposal from this document "
        "alone. Use the section numbers and control IDs consistently with "
        "earlier turns."
    ),
]


def estimate_tokens() -> None:
    """Print rough token estimates using len(text) // 4."""
    print(f"SEED_PROMPT: ~{len(SEED_PROMPT) // 4} tokens (len//4)")
    for idx, text in enumerate(FOLLOW_UPS):
        print(f"FOLLOW_UPS[{idx}]: ~{len(text) // 4} tokens (len//4)")


if __name__ == "__main__":
    estimate_tokens()
