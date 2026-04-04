#!/usr/bin/env python3
"""Next-token generation demo using an n-gram language model.

Displays the top N next-token candidates, but samples from all candidates with
probability greater than a threshold.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict


CORPUS_FILES = {
    "austen": "pride_and_prejudice.txt",
    "shakespeare": "hamlet.txt",
    "bible": "bible_kjv.txt",
}

TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]"
PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}


def locate_data_file(filename: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / filename,
        Path.cwd() / "data" / filename,
        script_dir / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find corpus file: {filename}")


def clean_gutenberg_text(text: str) -> str:
    text = re.sub(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", " ", text, flags=re.S)
    text = re.sub(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*", " ", text, flags=re.S)
    text = re.sub(r"\b\d+:\d+\b", " ", text)
    text = re.sub(r"[_\[\]()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_tokenize(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return re.findall(TOKEN_PATTERN, text)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.replace(" n't", "n't")
    return text


def load_sentences(choice: str) -> list[str]:
    if choice not in CORPUS_FILES:
        raise ValueError(f"Unknown corpus '{choice}'. Choose from: {', '.join(CORPUS_FILES)}")
    path = locate_data_file(CORPUS_FILES[choice])
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_gutenberg_text(raw_text)
    return sentence_tokenize(cleaned)


def build_ngram_model(sentences: list[str], n: int) -> DefaultDict[tuple[str, ...], Counter]:
    model: DefaultDict[tuple[str, ...], Counter] = defaultdict(Counter)
    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)
        if not sentence_tokens:
            continue
        padded = ["<s>"] * (n - 1) + sentence_tokens + ["</s>"]
        for i in range(len(padded) - n + 1):
            context = tuple(padded[i : i + n - 1]) if n > 1 else tuple()
            next_token = padded[i + n - 1]
            model[context][next_token] += 1
    return model


def choose_start_context(model: DefaultDict[tuple[str, ...], Counter], n: int) -> tuple[str, ...]:
    start_context = tuple(["<s>"] * (n - 1)) if n > 1 else tuple()
    if start_context in model:
        return start_context
    return random.choice(list(model.keys()))


def next_token_candidates(
    model: DefaultDict[tuple[str, ...], Counter],
    context: tuple[str, ...],
    min_probability: float,
) -> list[tuple[str, int, float]]:
    counter = model.get(context, Counter())
    total = sum(counter.values())
    if total == 0:
        return []

    candidates: list[tuple[str, int, float]] = []
    for token, count in counter.most_common():
        probability = count / total
        if probability > min_probability:
            candidates.append((token, count, probability))
    return candidates


def sample_next_token(candidates: list[tuple[str, int, float]]) -> str:
    tokens = [token for token, _, _ in candidates]
    probabilities = [p for _, _, p in candidates]
    return random.choices(tokens, weights=probabilities, k=1)[0]


def format_context(context: tuple[str, ...]) -> str:
    visible = [tok for tok in context if tok != "<s>"]
    return " ".join(visible) if visible else "<start>"


def print_candidates(candidates: list[tuple[str, int, float]], display_n: int) -> None:
    print("Top candidates:")
    print(f"{'rank':>4}  {'token':<20} {'count':>8} {'probability':>12}")
    for idx, (token, count, prob) in enumerate(candidates[:display_n], start=1):
        print(f"{idx:>4}  {token:<20} {count:>8} {prob:>12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show next-token candidates and sample one token.")
    parser.add_argument("--seed", default="i am", help="Seed text used to derive context.")
    parser.add_argument("--corpus", choices=list(CORPUS_FILES.keys()), default="bible", help="Corpus to train n-gram model.")
    parser.add_argument("--n", type=int, default=3, help="N-gram order (1, 2, 3, ...).")
    parser.add_argument("--display-n", type=int, default=10, help="How many top candidates to display.")
    parser.add_argument("--min-prob", type=float, default=0.000001, help="Only consider candidates where p > min-prob.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on generated tokens (0 = unlimited).")
    args = parser.parse_args()

    if args.n < 1:
        raise ValueError("--n must be at least 1")

    random.seed(args.random_seed)

    sentences = load_sentences(args.corpus)
    model = build_ngram_model(sentences, n=args.n)

    seed_tokens = [t for t in word_tokenize(args.seed) if t not in PUNCT_TOKENS]
    generated_tokens = list(seed_tokens)
    context = tuple(seed_tokens[-(args.n - 1) :]) if args.n > 1 else tuple()

    if args.n > 1 and (len(context) < args.n - 1 or context not in model):
        context = choose_start_context(model, args.n)

    print(f"Corpus: {args.corpus}")
    print(f"Seed text: {args.seed}")
    print(f"Current text: {detokenize(generated_tokens)}")
    print("Press Enter to generate one token at a time. Type 'q' then Enter to quit.")

    steps = 0
    while True:
        if args.max_steps > 0 and steps >= args.max_steps:
            print(f"Reached max steps: {args.max_steps}")
            break

        user_input = input("\nEnter=next token | q=quit: ").strip().lower()
        if user_input in {"q", "quit", "exit"}:
            break

        if args.n > 1 and context not in model:
            context = choose_start_context(model, args.n)

        candidates = next_token_candidates(model, context, args.min_prob)
        if not candidates:
            print(f"No next-token candidates found for context: {context}")
            break

        print(f"\nContext: {format_context(context)}")
        print(f"Candidates with p > {args.min_prob}: {len(candidates)}")
        print_candidates(candidates, args.display_n)

        sampled_token = sample_next_token(candidates)
        steps += 1

        if sampled_token == "</s>":
            print("Randomly selected next token: </s> (end of sentence)")
            if args.n > 1:
                context = choose_start_context(model, args.n)
            continue

        print(f"Randomly selected next token: {sampled_token}")
        generated_tokens.append(sampled_token)

        if args.n > 1:
            window = (list(context) + [sampled_token])[-(args.n - 1) :]
            context = tuple(window)

        print(f"Updated text: {detokenize(generated_tokens)}")

    print(f"\nFinal text: {detokenize(generated_tokens)}")


if __name__ == "__main__":
    main()
