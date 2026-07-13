# M1 Target Plan State Machine

## Logical path

1. `LOGICAL_READY`
2. `CLAIMED`
3. `BOUND`
4. `EXECUTING`
5. terminal:
   - `COMPLETED`
   - `FAILED`
   - `EXPIRED`
   - `CANCELLED`
   - `REJECTED`

## Formal path now used

- publish:
  - `publish_logical(...)`
- claim:
  - `claim(...)`
- bind:
  - `bind(...)`
- execution start:
  - `start_execution(...)`
- terminal complete:
  - `complete(...)`

## Compatibility retained

- `consume_once(...)`
- `claim_for_reconciliation(...)`
- `close_key_if_unclaimed(...)`

These remain for legacy callers and tests, but new M1 work is moving toward the explicit state path.
