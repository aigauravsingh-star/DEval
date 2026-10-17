# Security and Privacy

DEval is alpha evaluation software, not a certified security boundary.

- Configuration and custom Python adapters/hooks are trusted server-side code.
  An endpoint change changes where evaluation data goes. Do not accept config
  files, credential variable names, or backend identities from untrusted prompts.
- Hosted providers receive selected states and rubric questions. Local inference
  privacy also depends on serving code, network egress, dependencies, and logs.
- Core DEval performs no telemetry or automatic report upload. Optional frameworks
  and identity SDKs are separate products with separate behavior.
- Plain HTTP is limited to loopback unless explicitly allowed. Redirects are
  refused. TLS certificate verification uses Python's default trust configuration.
- Duplicate JSON keys, non-finite numbers, missing/extra answers, mismatched types,
  invalid probabilities, and reported truncation are errors, not pass verdicts.
- Native probabilities can be miscalibrated or confidently wrong. Capability flags
  and self-reported token/model metadata are assertions, not remote attestation.
- Reports omit case input states but can expose rubric text, IDs, endpoint metadata,
  probabilities, and hashes. Hashes of low-entropy data can be guessed; they are
  not anonymization. Custom backend identity metadata must be sanitized by its author.
- The engine records exception classes instead of provider error bodies. External
  framework logs/caches may retain more data; configure them separately.
- Benchmarks require independent labels. Keep evaluation data separate from
  training/threshold tuning and defend the judge against prompt injection.
- There is no automatic cross-provider fallback, retry, code execution from config,
  model download, claim extraction, or generated explanation in the core.

Report vulnerabilities privately to the repository maintainer through GitHub's
private vulnerability reporting when enabled. Do not put real secrets, prompts,
or customer data in public issues.
