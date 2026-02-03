#!/usr/bin/env python3
"""
Download and prepare the AI2 ARC (AI2 Reasoning Challenge) dataset.

This script downloads the ARC dataset and prepares it for testing
the Retrieval and AgenticSearch tools.
"""

import json
from pathlib import Path

# Try to import datasets (Hugging Face)
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' package not installed. Install with: pip install datasets")


def download_arc_dataset(output_dir: str = None, split: str = "test", challenge: bool = False):
    """
    Download the ARC dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset
        split: Dataset split (train, validation, test)
        challenge: If True, use ARC-Challenge; else ARC-Easy

    Returns:
        Path to the saved dataset
    """
    if not HAS_DATASETS:
        raise RuntimeError("'datasets' package is required. Install with: pip install datasets")

    # Default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset name
    subset = "ARC-Challenge" if challenge else "ARC-Easy"
    print(f"Downloading {subset} ({split} split)...")

    # Load dataset
    dataset = load_dataset("allenai/ai2_arc", subset, split=split)

    # Prepare output files
    # 1. JSONL file with questions and answers
    jsonl_path = output_dir / f"arc_{'challenge' if challenge else 'easy'}_{split}.jsonl"

    # 2. Text file with formatted Q&A for grep-based search
    txt_path = output_dir / f"arc_{'challenge' if challenge else 'easy'}_{split}.txt"

    # 3. JSON knowledge base for retrieval
    kb_path = output_dir / f"arc_{'challenge' if challenge else 'easy'}_{split}_kb.json"

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f, \
         open(txt_path, "w", encoding="utf-8") as txt_f, \
         open(kb_path, "w", encoding="utf-8") as kb_f:

        kb_entries = []

        for i, item in enumerate(dataset):
            # Extract data
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            # Format choices
            choice_texts = choices["text"]
            choice_labels = choices["label"]

            # Find correct answer
            correct_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
            correct_answer = choice_texts[correct_idx]

            # Build formatted entry
            entry = {
                "id": item["id"],
                "question": question,
                "choices": {label: text for label, text in zip(choice_labels, choice_texts)},
                "answer_key": answer_key,
                "answer": correct_answer,
            }

            # Write JSONL
            jsonl_f.write(json.dumps(entry) + "\n")

            # Write text format (good for grep search)
            txt_f.write(f"=== Question {i+1} (ID: {item['id']}) ===\n")
            txt_f.write(f"Q: {question}\n")
            txt_f.write("Choices:\n")
            for label, text in zip(choice_labels, choice_texts):
                marker = "*" if label == answer_key else " "
                txt_f.write(f"  {marker}({label}) {text}\n")
            txt_f.write(f"Answer: ({answer_key}) {correct_answer}\n")
            txt_f.write("\n")

            # Build KB entry (for semantic retrieval)
            kb_entry = {
                "id": item["id"],
                "content": f"Question: {question}\nAnswer: {correct_answer}",
                "question": question,
                "answer": correct_answer,
                "choices": {label: text for label, text in zip(choice_labels, choice_texts)},
                "answer_key": answer_key,
            }
            kb_entries.append(kb_entry)

        # Write KB JSON
        json.dump(kb_entries, kb_f, indent=2)

    print(f"Dataset saved to:")
    print(f"  - JSONL: {jsonl_path}")
    print(f"  - Text: {txt_path}")
    print(f"  - KB JSON: {kb_path}")
    print(f"  Total questions: {len(kb_entries)}")

    return {
        "jsonl": str(jsonl_path),
        "txt": str(txt_path),
        "kb": str(kb_path),
        "count": len(kb_entries),
    }


def create_sample_knowledge_base(output_dir: str = None, num_samples: int = 100):
    """
    Create a sample knowledge base from ARC for quick testing.

    Args:
        output_dir: Directory to save the sample
        num_samples: Number of samples to include

    Returns:
        Path to the sample file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    output_dir = Path(output_dir)

    # Check if full dataset exists
    full_txt = output_dir / "arc_easy_test.txt"
    if not full_txt.exists():
        print("Full dataset not found. Downloading first...")
        download_arc_dataset(output_dir, split="test", challenge=False)

    # Create sample
    sample_path = output_dir / f"arc_sample_{num_samples}.txt"

    with open(full_txt, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by question markers
    questions = content.split("=== Question")
    questions = ["=== Question" + q for q in questions[1:num_samples+1]]

    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("\n".join(questions))

    print(f"Sample KB saved to: {sample_path}")
    return str(sample_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download AI2 ARC dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--challenge", action="store_true", help="Use ARC-Challenge (harder)")
    parser.add_argument("--sample", type=int, default=0, help="Create sample of N questions")

    args = parser.parse_args()

    # Download dataset
    result = download_arc_dataset(args.output_dir, args.split, args.challenge)

    # Create sample if requested
    if args.sample > 0:
        create_sample_knowledge_base(args.output_dir or Path(__file__).parent, args.sample)

    print("\nDone!")
