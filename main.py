print("=" * 55)
print(" LOAN APPROVAL PREDICTOR")
print(" CodTech IT Solutions — ML Internship")
print(" Intern   : Abhishek Prasad")
print(" Intern ID: CITS5099")
print("=" * 55)

import subprocess, sys

steps = [
    ("Step 1: Generating loan dataset...",       "generate_data.py"),
    ("Step 2: Creating visualizations...",        "visualize.py"),
    ("Step 3: Training prediction models...",     "model.py"),
]

for msg, script in steps:
    print(f"\n{'─'*55}")
    print(f" {msg}")
    print(f"{'─'*55}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Stopping.")
        sys.exit(1)

print("\n" + "=" * 55)
print(" ✅ ALL DONE! Check outputs/ folder for charts.")
print("=" * 55)
