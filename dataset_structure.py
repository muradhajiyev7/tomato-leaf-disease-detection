import os
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

FINAL_DATASET = BASE_DIR / "data" / "raw" / "Universal_Tomato_Dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

STRUCTURE_OUTPUT_FILE = BASE_DIR / "final_dataset_structure.txt"
SUMMARY_OUTPUT_FILE = BASE_DIR / "final_dataset_summary.txt"


# =========================================================
# HELPERS
# =========================================================
def count_direct_files(folder_path: Path):
    total_files = 0
    image_files = 0

    try:
        for item in folder_path.iterdir():
            if item.is_file():
                total_files += 1
                if item.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files += 1
    except PermissionError:
        pass

    return total_files, image_files


def summarize_dataset(dataset_path: Path):
    total_dirs = 0
    total_files = 0
    total_images = 0

    for root, dirs, files in os.walk(dataset_path):
        total_dirs += len(dirs)
        total_files += len(files)
        total_images += sum(
            1 for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        )

    return total_dirs, total_files, total_images


def build_tree_lines(folder_path: Path, prefix="", max_depth=20, current_depth=0):
    lines = []

    if current_depth > max_depth:
        return lines

    try:
        items = sorted(
            list(folder_path.iterdir()),
            key=lambda x: (x.is_file(), x.name.lower())
        )
    except PermissionError:
        lines.append(prefix + "└── [Permission Denied]")
        return lines
    except FileNotFoundError:
        lines.append(prefix + "└── [Folder Not Found]")
        return lines

    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "

        if item.is_dir():
            total_files, image_files = count_direct_files(item)
            lines.append(
                f"{prefix}{connector}{item.name}/  [files: {total_files}, images: {image_files}]"
            )

            extension = "    " if is_last else "│   "
            lines.extend(
                build_tree_lines(
                    item,
                    prefix=prefix + extension,
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
            )
        else:
            file_type = "IMAGE" if item.suffix.lower() in IMAGE_EXTENSIONS else "FILE"
            lines.append(f"{prefix}{connector}{item.name}  [{file_type}]")

    return lines


def detect_class_folders_by_split(dataset_path: Path):
    split_summary = {}

    for split_dir in sorted(dataset_path.iterdir()):
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        split_summary[split_name] = []

        for class_dir in sorted(split_dir.iterdir(), key=lambda x: x.name.lower()):
            if not class_dir.is_dir():
                continue

            image_count = sum(
                1 for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

            split_summary[split_name].append({
                "class_name": class_dir.name,
                "image_count": image_count
            })

    return split_summary


# =========================================================
# MAIN
# =========================================================
def main():
    if not FINAL_DATASET.exists():
        print(f"[ERROR] Folder not found: {FINAL_DATASET}")
        print("Place the dataset locally under: data/raw/Universal_Tomato_Dataset/")
        return

    structure_lines = []
    summary_lines = []

    total_dirs, total_files, total_images = summarize_dataset(FINAL_DATASET)

    structure_lines.append("=" * 100)
    structure_lines.append("FINAL MERGED DATASET STRUCTURE REPORT")
    structure_lines.append("=" * 100)
    structure_lines.append(f"Path         : {FINAL_DATASET}")
    structure_lines.append(f"Total folders: {total_dirs}")
    structure_lines.append(f"Total files  : {total_files}")
    structure_lines.append(f"Total images : {total_images}")
    structure_lines.append("-" * 100)
    structure_lines.append(f"{FINAL_DATASET.name}/")
    structure_lines.extend(build_tree_lines(FINAL_DATASET, prefix="", max_depth=30, current_depth=0))
    structure_lines.append("")
    structure_lines.append("=" * 100)
    structure_lines.append("DONE")
    structure_lines.append("=" * 100)

    split_summary = detect_class_folders_by_split(FINAL_DATASET)

    summary_lines.append("=" * 100)
    summary_lines.append("FINAL MERGED DATASET SUMMARY")
    summary_lines.append("=" * 100)
    summary_lines.append(f"Path         : {FINAL_DATASET}")
    summary_lines.append(f"Total folders: {total_dirs}")
    summary_lines.append(f"Total files  : {total_files}")
    summary_lines.append(f"Total images : {total_images}")
    summary_lines.append("-" * 100)

    for split_name, classes in split_summary.items():
        summary_lines.append("")
        summary_lines.append("=" * 100)
        summary_lines.append(f"SPLIT: {split_name}")
        summary_lines.append("=" * 100)

        split_total = 0
        for idx, cls in enumerate(classes, start=1):
            split_total += cls["image_count"]
            summary_lines.append(
                f"{idx}. Class Name : {cls['class_name']}\n"
                f"   Image Count: {cls['image_count']}\n"
            )

        summary_lines.append(f"Split Total Images: {split_total}")

    summary_lines.append("")
    summary_lines.append("=" * 100)
    summary_lines.append("DONE")
    summary_lines.append("=" * 100)

    with open(STRUCTURE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(structure_lines))

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved full structure to: {STRUCTURE_OUTPUT_FILE}")
    print(f"Saved summary to       : {SUMMARY_OUTPUT_FILE}")
    print("\nDataset summary generation complete.")


if __name__ == "__main__":
    main()
