# Deploy

Deployment-facing entrypoints live here. They are intentionally thin wrappers around the existing platform scripts so the project can move toward a clearer layout without breaking known-good commands.

Linux server smoke / trace flow:

```bash
bash deploy/run_poc1_linux.sh
```

Windows local smoke flow:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\run_poc1_windows.ps1 -UseCurrentPython -ModelId model\OLMoE
```

The underlying compatibility scripts are still in `scripts/`:

- `scripts/run_poc1_linux.sh`
- `scripts/run_poc1_windows.ps1`
- `scripts/setup_env.sh`
- `scripts/preflight_gpu.py`

These deployment flows validate environment setup and real trace export. They do not run real ablation, adapter patching, or a full runtime scheduler.
