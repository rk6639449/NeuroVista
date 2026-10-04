import os
from pathlib import Path

# Automatically targets the script's directory or set manually
dataset_dir = Path(__file__).parent if '__file__' in globals() else Path(".")
output_txt = dataset_dir / "dataset_summary.txt"

summary_lines = [
    "==================================================",
    "          COMPLETE DATASET FILE LISTING           ",
    "==================================================\n"
]

total_subjects = 0

for subject in sorted(dataset_dir.iterdir()):
    # Process only subject directories starting with ReMIND-
    if not subject.is_dir() or not subject.name.startswith("ReMIND-"):
        continue

    total_subjects += 1
    summary_lines.append(f"Subject: {subject.name}")

    for subfolder_name in ["Intraop", "Preop"]:
        folder_path = subject / subfolder_name
        summary_lines.append(f"  [{subfolder_name}]")

        if not folder_path.exists():
            summary_lines.append("    └── [MISSING FOLDER]")
            continue

        # Get all non-hidden files inside the subfolder
        files = sorted([f for f in folder_path.iterdir() if not f.name.startswith(".")])

        if not files:
            summary_lines.append("    └── [EMPTY FOLDER]")
        else:
            for idx, file_path in enumerate(files):
                connector = "└──" if idx == len(files) - 1 else "├──"
                summary_lines.append(f"    {connector} {file_path.name}")

    summary_lines.append("-" * 50)

summary_lines.append("\n==================================================")
summary_lines.append(f"Total Subjects Processed: {total_subjects}")
summary_lines.append("==================================================")

# Write output text file
with open(output_txt, "w") as f:
    f.write("\n".join(summary_lines))

print(f"Summary generated successfully: {output_txt}")