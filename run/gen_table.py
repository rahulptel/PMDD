#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

# Farm configurations
FARMS = {
    "coup":        {"runner": "baseline-dd", "backend": "cpu", "workers": 1,  "forced_method": 3, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "dpa":         {"runner": "baseline-dpa", "backend": "cpu", "workers": 1,  "forced_method": None, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True, "modes": ["", "-a"]},
    "cpu-coup":    {"backend": "cpu", "workers": 1,  "forced_method": 0, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-coup-4":  {"backend": "cpu", "workers": 4,  "forced_method": 0, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-coup-8":  {"backend": "cpu", "workers": 8,  "forced_method": 0, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-coup-16": {"backend": "cpu", "workers": 16, "forced_method": 0, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-td":      {"backend": "cpu", "workers": 1,  "forced_method": 1, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-td-4":    {"backend": "cpu", "workers": 4,  "forced_method": 1, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-td-8":    {"backend": "cpu", "workers": 8,  "forced_method": 1, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "cpu-td-16":   {"backend": "cpu", "workers": 16, "forced_method": 1, "max_knapsack": 70,  "max_setpacking": 225,  "skip_setpacking_175": True},
    "gpu-coup":    {"backend": "gpu", "workers": 1,  "forced_method": 3, "max_knapsack": None, "max_setpacking": None, "skip_setpacking_175": False},
    "gpu-td":      {"backend": "gpu", "workers": 1,  "forced_method": 1, "max_knapsack": None, "max_setpacking": None, "skip_setpacking_175": False},
}

KNAPSACK_CASES = {
    (40, 6), (40, 7),
    (50, 5), (50, 6), (50, 7),
    (60, 4), (60, 5), (60, 6), (60, 7),
    (70, 3), (70, 4), (70, 5), (70, 6), (70, 7),
}

SETPACKING_CASES = {
    (150, 3), (150, 4), (150, 5), (150, 6), (150, 7),
    (200, 3), (200, 4),
}

TSP_CASES = {
    (15, 4), (15, 5), (15, 6), (15, 7),
}

def iter_dat_files(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.dat") if p.is_file())

def parse_knapsack_nvars(name: str):
    m = re.match(r"^KP_p-\d+_n-(\d+)_ins-\d+\.dat$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^knapsack-(\d+)-\d+-\d+-\d+\.dat$", name)
    if m:
        return int(m.group(1))
    return None

def parse_setpacking_nvars(name: str):
    m = re.match(r"^bp-(\d+)-\d+-\d+-\d+-\d+\.dat$", name)
    return int(m.group(1)) if m else None

def parse_tsp_cities(name: str):
    m = re.match(r"^tsp-nobj\d+-ncities(\d+)-seed\d+\.dat$", name)
    return int(m.group(1)) if m else None

def dpa_instance_path(instance: Path):
    if instance.suffix == ".lp":
        return instance
    return instance.with_name(f"{instance.stem}-dpa.lp")

def append_case(lines, config, binary, instance, problem_type, method, dominance):
    runner = config.get("runner", "pmdd")
    backend = config["backend"]
    cpu_workers = config["workers"]

    if runner == "baseline-dpa":
        dpa_instance = dpa_instance_path(instance)
        for mode in config.get("modes", [""]):
            if mode:
                lines.append(f"{binary} {dpa_instance} {mode}")
            else:
                lines.append(f"{binary} {dpa_instance}")
        return

    if runner == "baseline-dd":
        baseline_problem_type = 6 if problem_type == 3 else problem_type
        lines.append(
            f"{binary} {instance} {baseline_problem_type} 0 {method} 0 0 {dominance}"
        )
        return

    if backend == "gpu":
        lines.append(
            f"{binary} {instance} {problem_type} {method} {dominance} --backend gpu --save-frontier --save-stats"
        )
        return

    if cpu_workers > 1:
        lines.append(
            f"{binary} {instance} {problem_type} {method} {dominance} --backend cpu --cpu-threads {cpu_workers} --save-frontier --save-stats"
        )
    else:
        lines.append(
            f"{binary} {instance} {problem_type} {method} {dominance} --backend cpu --save-frontier --save-stats"
        )

def generate_table(farm_name, config, project_root):
    binary_base = project_root / "resources" / "bin"
    target_data_base = project_root / "data"

    nobjs_min = 3
    nobjs_max = 7

    # We scan data from the local repository copy
    script_dir = Path(__file__).resolve().parent
    local_project_root = script_dir.parent
    local_data_base = local_project_root / "data"

    lines = []
    for nobjs in range(nobjs_min, nobjs_max + 1):
        source_data_root = local_data_base / str(nobjs)
        if not source_data_root.is_dir():
            continue

        runner = config.get("runner", "pmdd")
        if runner == "baseline-dd":
            binary = binary_base / "baseline" / "dd" / f"multiobj{nobjs}"
        elif runner == "baseline-dpa":
            binary = binary_base / "baseline" / "dpa" / "main"
        else:
            binary = binary_base / "pmdd" / f"multiobj_nobjs{nobjs}"

        forced_method = config["forced_method"]
        if forced_method == 0:
            method_knapsack = 3
            method_binproblem = 3
            method_tsp = 3
        else:
            method_knapsack = forced_method
            method_binproblem = forced_method
            method_tsp = forced_method

        # Knapsack
        for path in iter_dat_files(source_data_root / "knapsack"):
            nvars = parse_knapsack_nvars(path.name)
            if nvars is None or (nvars, nobjs) not in KNAPSACK_CASES:
                continue
            if config["max_knapsack"] is not None and nvars > config["max_knapsack"]:
                continue
            target_instance = target_data_base / str(nobjs) / "knapsack" / path.name
            append_case(lines, config, binary, target_instance, 1, method_knapsack, 1)

        # Set packing (binproblem)
        for path in iter_dat_files(source_data_root / "binproblem"):
            nvars = parse_setpacking_nvars(path.name)
            if nvars is None or (nvars, nobjs) not in SETPACKING_CASES:
                continue
            if config["max_setpacking"] is not None and nvars > config["max_setpacking"]:
                continue
            if config["skip_setpacking_175"] and nvars == 175:
                continue
            target_instance = target_data_base / str(nobjs) / "binproblem" / path.name
            append_case(lines, config, binary, target_instance, 2, method_binproblem, 0)

        # TSP
        for path in iter_dat_files(source_data_root / "tsp"):
            ncities = parse_tsp_cities(path.name)
            if ncities is None or (ncities, nobjs) not in TSP_CASES:
                continue
            target_instance = target_data_base / str(nobjs) / "tsp" / path.name
            append_case(lines, config, binary, target_instance, 3, method_tsp, 0)

    return "\n".join(str(line) for line in lines) + ("\n" if lines else "")


def main():
    script_dir = Path(__file__).resolve().parent
    local_project_root = script_dir.parent

    cc_project_root_str = os.environ.get("PROJECT_ROOT", "/home/rahulpat/scratch/PMDD")
    cc_project_root = Path(cc_project_root_str).resolve()

    print(f"Generating tables for CC environment (root: {cc_project_root})")
    print(f"Generating tables and runner script for Local environment (root: {local_project_root})")

    local_dir = script_dir / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    for farm_name, config in FARMS.items():
        # CC table
        cc_farm_dir = script_dir / "cc" / farm_name
        cc_farm_dir.mkdir(parents=True, exist_ok=True)
        cc_table_content = generate_table(farm_name, config, cc_project_root)
        (cc_farm_dir / "table.dat").write_text(cc_table_content, encoding="utf-8")
        print(f"  Written run/cc/{farm_name}/table.dat ({len(cc_table_content.splitlines())} cases)")

        # Local table
        local_table_content = generate_table(farm_name, config, local_project_root)
        local_table_path = local_dir / f"{farm_name}.dat"
        local_table_path.write_text(local_table_content, encoding="utf-8")
        print(f"  Written run/local/{farm_name}.dat ({len(local_table_content.splitlines())} cases)")


if __name__ == "__main__":
    main()
