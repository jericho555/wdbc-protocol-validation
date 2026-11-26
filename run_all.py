"""Run all experiments in order. Total wall-time ~6-7 hours on a modern laptop."""
import subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(HERE, "scripts")

ORDER = [
    "run_panel.py",
    "run_reduct.py",
    "run_extended.py",
    "run_ensemble.py",
    "run_protocol.py",
    "run_interp.py",
    "run_perm_exact.py",
    "run_clinical.py",
    "run_lime.py",
    "run_corr.py",
    "run_validate.py",
    "run_seeds.py",
    "run_explain.py",
    "run_final.py",
    "visualize.py",
]

def main():
    print("=" * 72)
    print("WDBC Protocol Validation — full experiment pipeline")
    print("=" * 72)
    t0 = time.time()
    for i, name in enumerate(ORDER, 1):
        path = os.path.join(SCRIPTS_DIR, name)
        if not os.path.exists(path):
            print(f"  [{i:2d}/{len(ORDER)}] SKIP (not found): {name}")
            continue
        print(f"\n  [{i:2d}/{len(ORDER)}] Running: {name}")
        print(f"  {'─' * 50}")
        t1 = time.time()
        result = subprocess.run([sys.executable, path], cwd=SCRIPTS_DIR)
        dt = time.time() - t1
        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"  {'─' * 50}")
        print(f"  [{i:2d}/{len(ORDER)}] {name} — {status} ({dt:.1f}s)")
        if result.returncode != 0:
            print("\n  Stopping pipeline on failure. Check output above.")
            sys.exit(1)
    total = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"All experiments completed in {total/60:.1f} minutes.")
    print(f"{'=' * 72}")

if __name__ == "__main__":
    main()
