"""Record the execution environment to results/environment.json."""
import json
import os
import platform
import re
import subprocess
import sys

from common import RESULTS, load_config


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # tool missing on this platform
        return f"unavailable ({e.__class__.__name__})"


def cpu_info():
    info = {"logical_processors": os.cpu_count()}
    if sys.platform.startswith("linux"):
        out = sh(["lscpu"])
        fields = dict(re.findall(r"^([^:]+):\s+(.*)$", out, re.M))
        info["model"] = fields.get("Model name")
        cps = int(fields.get("Core(s) per socket", 0) or 0)
        sockets = int(fields.get("Socket(s)", 0) or 0)
        info["physical_cores"] = cps * sockets if cps and sockets else None
        info["threads_per_core"] = fields.get("Thread(s) per core")
        info["hypervisor"] = fields.get("Hypervisor vendor")
        info["caches"] = {k: v for k, v in fields.items() if "cache" in k}
        mem = open("/proc/meminfo").read()
        info["ram_bytes"] = int(re.search(r"MemTotal:\s+(\d+)", mem).group(1)) * 1024
    elif sys.platform == "darwin":
        info["model"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["physical_cores"] = int(sh(["sysctl", "-n", "hw.physicalcpu"]))
        info["ram_bytes"] = int(sh(["sysctl", "-n", "hw.memsize"]))
        info["caches"] = {"l1d": sh(["sysctl", "-n", "hw.l1dcachesize"]),
                          "l2": sh(["sysctl", "-n", "hw.l2cachesize"])}
    return info


def main():
    cfg = load_config()
    env = {
        "cpu": cpu_info(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "c_compiler": sh([cfg["c_compiler"], "--version"]).splitlines()[0],
        "c_flags": cfg["c_flags"],
        "java_version": sh(["java", "-version"]),
        "javac_version": sh(["javac", "-version"]),
        "java_opts": cfg["java_opts"],
        "threads": {
            "c": "1 (single-threaded program, no libraries)",
            "python": "1 (CPython, pure-Python loops, GIL)",
            "java": "1 application thread; the JVM also runs JIT-compiler and "
                    "(Serial) GC threads, see java_vm_threads",
        },
        "java_vm_threads": sh(["java", *cfg["java_opts"], "-XX:+PrintFlagsFinal", "-version"]),
    }
    # keep only the thread-related JVM flags
    env["java_vm_threads"] = "\n".join(
        l.strip() for l in env["java_vm_threads"].splitlines()
        if re.search(r"\b(CICompilerCount|ParallelGCThreads|ConcGCThreads|UseSerialGC|MaxHeapSize)\b", l))
    try:
        import numpy
        env["numpy_reference_version"] = numpy.__version__
    except ImportError:
        pass
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
