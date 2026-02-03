#!/usr/bin/env python3
import csv
import os
import subprocess
import tempfile
import time
import sys


RESULTS_DIR = "results"
REPOS_DIR = "repos"
MACOS_AVG_POWER_W = 20
SCRIPTS_VENV_PY = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")


def read_energy_linux():
    total = 0
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None
    for name in os.listdir(base):
        if not name.startswith("intel-rapl:"):
            continue
        path = os.path.join(base, name, "energy_uj")
        try:
            with open(path, "r", encoding="utf-8") as f:
                val = int(f.read().strip() or 0)
                total += val
        except (OSError, ValueError):
            continue
    return total or None


def read_accuracy(output_path):
    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            acc = ""
            for line in f:
                line = line.strip()
                if line.startswith("ACCURACY="):
                    acc = line.split("=", 1)[1]
            return acc
    except OSError:
        return ""


def ensure_requirements(project):
    req = os.path.join(project, "requirements.txt")
    pip = os.path.join(project, "venv", "bin", "pip")
    if os.path.isfile(req) and os.path.isfile(pip):
        subprocess.run([pip, "install", "-r", req], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mem_delta_mb_for_process(proc):
    try:
        import psutil  # type: ignore
    except Exception:
        return 0.0

    try:
        p = psutil.Process(proc.pid)
    except Exception:
        return 0.0

    try:
        baseline = p.memory_info().rss
    except Exception:
        baseline = 0

    peak = baseline
    samples = 0
    while True:
        if proc.poll() is not None:
            break
        try:
            rss = p.memory_info().rss
            if rss > peak:
                peak = rss
        except Exception:
            pass
        samples += 1
        time.sleep(0.05)

    if samples < 3:
        for _ in range(5):
            try:
                rss = p.memory_info().rss
                if rss > peak:
                    peak = rss
            except Exception:
                pass
            time.sleep(0.05)

    delta = max(0, peak - baseline)
    return delta / (1024 ** 2)


def main():
    # Re-exec with scripts/venv python if available and not already using it.
    if os.path.isfile(SCRIPTS_VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(SCRIPTS_VENV_PY):
        os.execv(SCRIPTS_VENV_PY, [SCRIPTS_VENV_PY] + sys.argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "project",
                "script",
                "status",
                "accuracy",
                "mem_delta_mb",
                "exec_time_s",
                "energy_j",
                "notes",
            ]
        )

        repos_root = os.path.abspath(REPOS_DIR)
        for project in sorted(os.listdir(repos_root)):
            project_path = os.path.join(repos_root, project)
            if not os.path.isdir(project_path):
                continue

            script_files = []
            for name in os.listdir(project_path):
                if name.startswith("GENAIGREENML") and name.endswith(".py"):
                    script_files.append(os.path.join(project_path, name))

            def script_priority(path):
                name = os.path.basename(path)
                if "original" in name:
                    return (0, name)
                if "assisted" in name:
                    return (1, name)
                if "autonomous" in name:
                    return (2, name)
                return (3, name)

            script_files.sort(key=script_priority)

            if not script_files:
                writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S%z"), project, "GENAIGREENML*.py", "SKIPPED", "", "", "", "", ""])
                continue

            ensure_requirements(project_path)

            venv_python = os.path.join(project_path, "venv", "bin", "python")
            if not (os.path.isfile(venv_python) and os.access(venv_python, os.X_OK)):
                for script_path in script_files:
                    writer.writerow(
                        [
                            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            project,
                            os.path.basename(script_path),
                            "FAILED",
                            "",
                            "",
                            "",
                            "",
                            "venv python missing",
                        ]
                    )
                continue

            for script_path in script_files:
                script_name = os.path.basename(script_path)
                print(f"▶ Running {project} / {script_name}")

                energy_before = read_energy_linux()
                start = time.time()

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    output_path = tmp.name

                proc = subprocess.Popen(
                    [venv_python, script_name],
                    cwd=project_path,
                    stdout=open(output_path, "w"),
                    stderr=subprocess.STDOUT,
                )

                mem_delta_mb = mem_delta_mb_for_process(proc)
                exit_code = proc.wait()
                end = time.time()

                exec_time = end - start
                accuracy = read_accuracy(output_path)

                notes = ""
                energy_j = ""
                if energy_before is not None:
                    energy_after = read_energy_linux()
                    if energy_after is not None:
                        energy_j = f"{(energy_after - energy_before) / 1_000_000:.6f}"
                    else:
                        notes = "energy not available"
                else:
                    energy_j = f"{(exec_time * MACOS_AVG_POWER_W):.3f}"
                    notes = f"energy approx on macOS using {MACOS_AVG_POWER_W}W"

                status = "OK" if exit_code == 0 else "FAILED"

                writer.writerow(
                    [
                        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        project,
                        script_name,
                        status,
                        accuracy,
                        f"{mem_delta_mb:.3f}",
                        f"{exec_time:.9f}",
                        energy_j,
                        notes,
                    ]
                )

                try:
                    os.unlink(output_path)
                except OSError:
                    pass

    print(f"✅ Finished. Results written to {log_path}")


if __name__ == "__main__":
    main()
