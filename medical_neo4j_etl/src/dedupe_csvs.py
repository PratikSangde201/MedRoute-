import csv
from pathlib import Path


BASE_DIR = Path(__file__).parent / "data"
TARGETS = [
    "disease_symptom.csv",
    "disease_precaution.csv",
]


def dedupe_file(file_name: str) -> dict:
    path = BASE_DIR / file_name
    if not path.exists():
        return {"file": file_name, "found": False, "before": 0, "after": 0}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    before = len(rows)
    seen = set()
    deduped = []
    for row in rows:
        key = tuple((column, (row.get(column) or "").strip()) for column in fieldnames)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    return {"file": file_name, "found": True, "before": before, "after": len(deduped)}


def main():
    results = [dedupe_file(file_name) for file_name in TARGETS]
    print("CSV dedupe results:")
    for result in results:
        if not result["found"]:
            print(f"- {result['file']}: file not found")
            continue
        removed = result["before"] - result["after"]
        print(
            f"- {result['file']}: before={result['before']} after={result['after']} removed={removed}"
        )


if __name__ == "__main__":
    main()
