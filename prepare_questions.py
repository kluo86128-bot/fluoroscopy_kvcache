"""Update existing question files in place; never reads private text or labels."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


QUESTION_VERSION = "target_relation_semantic_v4"


def structured_question(output, tracking, relation, required, question):
    return (f"Output format: {output}\n\n"
            f"Tracking rule: {tracking}\n"
            f"Target relation: {relation}\n"
            f"Required value: {required}\n\n"
            f"Question: {question}")


def test_question(output, question):
    return f"Output format: {output}\n\nQuestion: {question}"


QUESTIONS = {
    "party_size": (
        structured_question("Return only the number in digits.",
                            "Track the dining party associated with the reservation across revisions. Apply explicit party-size corrections; retain the existing party size when only other fields change. Interpret corrections field by field. Include all diners, rather than counting only the person making the booking.",
                            "For the final confirmed restaurant reservation, retrieve the total number of diners included in that booking.",
                            "Extract the complete numerical count stated for this field.",
                            "How many people are included in the final confirmed restaurant reservation?"),
        test_question("Return only the number in digits.",
                      "What is the party size of the final confirmed restaurant booking?"),
        "digits"),
    "city": (
        structured_question("Return only the complete English city name.",
                            "Track the restaurant's identity across revisions. Retain its city when only other fields change; apply explicit city corrections. If the restaurant changes, use the replacement's city. Interpret corrections field by field.",
                            "For the restaurant in the final confirmed reservation, retrieve the city explicitly assigned to that restaurant.",
                            "Copy the complete city name as stated in the conversation, preserving all words and the original spelling.",
                            "In which city is the restaurant associated with the final confirmed reservation located?"),
        test_question("Return only the complete English city name.",
                      "What is the location city of the restaurant in the final confirmed booking?"),
        "english_city"),
    "date": (
        structured_question("Return only the full English month name, one space, and the day number with its ordinal suffix. Do not include a year.",
                            "Track the dining event associated with the reservation across revisions. Apply explicit dining-date corrections; retain the existing date when only other fields change. Interpret corrections field by field. Resolve relative dates using calendar references supplied in the conversation. Treat equivalent date expressions as the same date, and distinguish the dining date from dates of other events.",
                            "For the final confirmed restaurant reservation, retrieve the calendar date on which the diners are scheduled to eat.",
                            "Extract both the month and the day of the same dining date.",
                            "What are the month and day of the dining date for the final confirmed restaurant reservation?"),
        test_question("Return only the full English month name, one space, and the day number with its ordinal suffix. Do not include a year.",
                      "What month and day were confirmed for dining in the final restaurant booking?"),
        "month_ordinal"),
}


def update_questions(manifest=None, *, slots=("city",)):
    manifest = Path(manifest or ROOT / "test_samples" / "tasks.jsonl").resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or any(row.get("slot") not in QUESTIONS for row in rows):
        raise ValueError("源样本包含未定义的问题字段")
    slots = set(slots)
    if not slots or slots - QUESTIONS.keys():
        raise ValueError("必须选择已定义的问题字段")
    selected = [row for row in rows if row["slot"] in slots]
    protected = {manifest}
    for row in rows:
        if row["slot"] not in slots:
            protected.update((manifest.parent / value).resolve() for value in
                             (row["receiver_inputs"]["question_file"], row["held_out_question_file"]))
        for key in ("evaluation_only_file", "held_out_evaluation_only_file", "source_snapshot_file"):
            if row.get(key):
                protected.add((manifest.parent / row[key]).resolve())
        for mapping in (row.get("teacher_only", {}), row.get("receiver_inputs", {})):
            for key, value in mapping.items():
                if key.endswith("_file") and key != "question_file":
                    protected.add((manifest.parent / value).resolve())
    updates = {}
    # Validate every destination before writing. Repeated questions share templates.
    for row in selected:
        training, held, answer_format = QUESTIONS[row["slot"]]
        for relative, text in ((row["receiver_inputs"]["question_file"], training),
                               (row["held_out_question_file"], held)):
            path = (manifest.parent / relative).resolve()
            if path in protected or not path.is_relative_to(manifest.parent) or path.suffix != ".txt" or not path.is_file():
                raise ValueError(f"问题文件路径无效: {path}")
            if path in updates and updates[path] != text + "\n":
                raise ValueError(f"同一问题文件被不同字段/问法共用: {path}")
            updates[path] = text + "\n"
        row["answer_format"] = answer_format
        row["question_version"] = QUESTION_VERSION
    for path, text in updates.items():
        if path.read_text(encoding="utf-8-sig") != text:
            path.write_text(text, encoding="utf-8")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if manifest.read_text(encoding="utf-8-sig") != text:
        manifest.write_text(text, encoding="utf-8")
    return len(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量原地更新样本的问题，不读取或改写私有文本和评估答案")
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_samples" / "tasks.jsonl")
    parser.add_argument("--slots", nargs="+", choices=tuple(QUESTIONS), default=["city"],
                        help="只修改指定字段；默认仅 city")
    args = parser.parse_args()
    print(f"Updated {update_questions(args.manifest, slots=args.slots)} samples in place.")
