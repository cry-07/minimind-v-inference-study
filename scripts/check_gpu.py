"""Read-only PyTorch/GPU probe; run with each candidate Python interpreter."""
import json
import platform
import sys
import traceback
from datetime import datetime, timezone

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
}
try:
    import torch

    report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(),
                  compiled_architectures=torch.cuda.get_arch_list())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter")
    props = torch.cuda.get_device_properties(0)
    report.update(gpu=props.name, capability=list(torch.cuda.get_device_capability(0)),
                  total_memory_bytes=props.total_memory)
    # Compare a real CUDA matmul against a CPU result with exact integer entries.
    a_cpu = torch.arange(4096, dtype=torch.float32).reshape(64, 64) % 7
    expected = a_cpu @ a_cpu.T
    a_gpu = a_cpu.to("cuda")
    actual = a_gpu @ a_gpu.T
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    report.update(matmul_check="passed", result_device=str(actual.device),
                  max_absolute_error=(actual.cpu() - expected).abs().max().item(),
                  status="passed")
except Exception:
    report.update(status="failed", error=traceback.format_exc())

print(json.dumps(report, ensure_ascii=True, indent=2))
sys.exit(0 if report["status"] == "passed" else 1)
