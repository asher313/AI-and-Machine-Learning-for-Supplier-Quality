# Chapter 24: verify dated model configuration

The script does not run on import. Generate a synthetic catalog and inspect it without contacting a provider:

```bash
uv run python scripts/verify_snapshot.py --write-fixture work/models.json
uv run python scripts/verify_snapshot.py --catalog work/models.json --out work/snapshot-report.json
```

For an explicitly authorized provider metadata check, use `--live --out work/live-snapshot.json` with configured credentials. It lists all pages from the first-party Models API; Bedrock and other endpoints need their own configured checks. It sends no generation prompt. The output is timestamped and distinguishes LISTED, NOT_LISTED and CHECK_FAILED. Exit codes are 0 for all listed, 1 for any absent and 2 for a failed check.

Catalog presence does not establish service health, capabilities, prices, legal approval or model quality. Absence alone does not prove retirement: inspect scope, permissions, alias behavior and official availability/deprecation notices. Respond to an actual outage promptly through the incident/fallback procedure. Validate replacement behavior and deployment permissions before release. A fixed twice-yearly check is a teaching example; production monitoring and provider-change response need risk-appropriate intervals.

The local HTTP-mocked SDK test verifies automatic pagination across two pages and the absence status. No live metadata or generation API was called for this review. The six `docs/design/build*.md` records describe the teaching implementations and their remaining deployment work, not company approvals.
