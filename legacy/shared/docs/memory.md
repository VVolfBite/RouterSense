# Working Memory

This file records the collaboration contract used in this repository.

## Collaboration mode

- An external LLM may review the code and provide the next-step instruction.
- The user acts as the relay between that reviewer and the implementation agent.
- The implementation agent is responsible for making the code changes.
- If there are blockers, ambiguities, technical risks, or tradeoffs worth escalating, surface them clearly so the user can relay them.
- The user may explicitly grant full local execution permission; when that is true, actively verify the environment instead of assuming.

## Delivery routine

After each completed code task:

1. update the repository code
2. run lightweight verification when possible
3. overwrite the archive `/root/autodl-tmp/RouterSense.TAR.GZ`
4. create a git commit for that round of work

## Archive rule

When creating `/root/autodl-tmp/RouterSense.TAR.GZ`:

- include repository code and project files
- exclude model weights
- exclude cache directories
- exclude `.git`
- exclude generated output directories unless explicitly requested

## Current local paths

- repo root: `/root/autodl-tmp/RouterSense`
- archive path: `/root/autodl-tmp/RouterSense.TAR.GZ`
- model root: `/root/autodl-tmp/models`
- OLMoE path: `/root/autodl-tmp/models/OLMoE-1B-7B-0924`
- Qwen MoE path: `/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B`
- Mixtral path: `/root/autodl-tmp/models/Mixtral-8x7B-Instruct-v0.1`

## Current machine state

- Full local execution is available in this session.
- GPU checks should be run directly when relevant.
- Current target machine may provide 4x `NVIDIA GeForce RTX 4090 D` for real POC2 validation.

## Tooling note

- Prefer keeping `pytest` available for fast local verification.
- Preserve the existing rule: do not revert unrelated user changes.
