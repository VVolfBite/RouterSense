# M1 Hook Event Flow

## Formal dispatch

1. `DispatchReadyEvent`
2. native dispatch
3. `DispatchCompleteEvent`

Formal wrapper no longer performs a second `on_dispatch(...)` callback.

## Formal combine

1. `CombineReadyEvent`
2. native combine
3. `CombineCompleteEvent`

Formal wrapper no longer performs a second `on_combine(...)` callback.

## Event entrypoint

All wrapped formal dispatcher hooks now enter lifecycle through:

`RouterSenseInjectionRuntime.handle(event: RuntimeEvent) -> RuntimeDecision`

## Forward envelope

Root model forward now emits:

1. `ForwardBeginEvent`
2. dispatch/combine event sequence
3. `ForwardEndEvent`
