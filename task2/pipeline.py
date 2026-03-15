import os
import ast
import jsonlines
import random
import argparse
import re

from rank_bm25 import BM25Okapi

argparser = argparse.ArgumentParser()

argparser.add_argument("--stage", type=str, default="practice")
argparser.add_argument("--lang", type=str, default="python")
argparser.add_argument("--strategy", type=str, default="hybrid")

argparser.add_argument("--trim-prefix", action="store_true")
argparser.add_argument("--trim-suffix", action="store_true")

args = argparser.parse_args()

stage = args.stage
language = args.lang
strategy = args.strategy

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"Running improved pipeline for stage '{stage}'")

FILE_SEP_SYMBOL = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

TOP_K_FILES = 3


# -----------------------------
# Utility Functions
# -----------------------------

def prepare_bm25_str(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    return text.split()


def trim_prefix(prefix):
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > 10:
        prefix = "\n".join(prefix_lines[-10:])
    return prefix


def trim_suffix(suffix):
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > 10:
        suffix = "\n".join(suffix_lines[:10])
    return suffix


# -----------------------------
# Identifier Extraction
# -----------------------------

def extract_identifiers(code):
    identifiers = set()

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)

            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)

            elif isinstance(node, ast.FunctionDef):
                identifiers.add(node.name)

            elif isinstance(node, ast.ClassDef):
                identifiers.add(node.name)

    except Exception:
        pass

    return identifiers


# -----------------------------
# Repository Indexing
# -----------------------------

def index_repository(root_dir, min_lines=10):

    corpus = []
    file_names = []
    identifier_map = {}

    for dirpath, _, filenames in os.walk(root_dir):

        for filename in filenames:

            if not filename.endswith(extension):
                continue

            file_path = os.path.join(dirpath, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:

                    content = f.read()
                    lines = content.splitlines()

                    if len(lines) < min_lines:
                        continue

                    tokens = prepare_bm25_str(content)

                    corpus.append(tokens)
                    file_names.append(file_path)

                    ids = extract_identifiers(content)
                    identifier_map[file_path] = ids

            except Exception:
                pass

    return corpus, file_names, identifier_map


# -----------------------------
# Hybrid Retrieval
# -----------------------------

def retrieve_files(root_dir, prefix, suffix):

    query = prefix + "\n" + suffix
    query_tokens = prepare_bm25_str(query)

    query_ids = extract_identifiers(query)

    corpus, file_names, identifier_map = index_repository(root_dir)

    if not corpus:
        return []

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    hybrid_scores = []

    for idx, file_path in enumerate(file_names):

        bm25_score = bm25_scores[idx]

        ids = identifier_map.get(file_path, set())

        overlap = len(query_ids.intersection(ids))
        symbol_score = overlap

        final_score = 0.7 * bm25_score + 0.3 * symbol_score

        hybrid_scores.append((final_score, file_path))

    hybrid_scores.sort(reverse=True)

    top_files = [f for _, f in hybrid_scores[:TOP_K_FILES]]

    return top_files


# -----------------------------
# Compose Context
# -----------------------------

def compose_context(root_directory, file_paths):

    contexts = []

    for file_name in file_paths:

        try:
            content = open(file_name, "r", encoding="utf-8").read()
            clean_name = file_name[len(root_directory) + 1:]

            contexts.append(
                FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL,
                    file_name=clean_name,
                    file_content=content
                )
            )

        except Exception:
            pass

    return "\n".join(contexts)


# -----------------------------
# Dataset Processing
# -----------------------------

completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")

prediction_file_name = f"{language}-{stage}-{strategy}"

if args.trim_prefix:
    prediction_file_name += "-short-prefix"

if args.trim_suffix:
    prediction_file_name += "-short-suffix"

predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")


with jsonlines.open(completion_points_file, "r") as reader:

    with jsonlines.open(predictions_file, "w") as writer:

        for datapoint in reader:

            repo_path = datapoint["repo"].replace("/", "__")
            repo_revision = datapoint["revision"]

            root_directory = os.path.join(
                "data",
                f"repositories-{language}-{stage}",
                f"{repo_path}-{repo_revision}"
            )

            prefix = datapoint["prefix"]
            suffix = datapoint["suffix"]

            selected_files = retrieve_files(
                root_directory,
                prefix,
                suffix
            )

            if not selected_files:
                continue

            context = compose_context(root_directory, selected_files)

            submission = {"context": context}

            if args.trim_prefix:
                submission["prefix"] = trim_prefix(prefix)

            if args.trim_suffix:
                submission["suffix"] = trim_suffix(suffix)

            for f in selected_files:
                clean = f[len(root_directory) + 1:]
                print(f"Picked file: {clean}")

            writer.write(submission)
