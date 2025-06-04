#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
import zlib
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

import regex as re

from dpr.data.retriever_data import TableChunk
from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple("QAMatchStats", ["top_k_hits", "questions_doc_hits"])

QATableMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_chunk_hits", "top_k_table_hits", "questions_doc_hits"]
)

# Global variable to store all_docs in worker processes
WORKER_ALL_DOCS = None

def init_worker_all_docs(all_docs_data):
    """Initializer for worker processes to set the global WORKER_ALL_DOCS."""
    global WORKER_ALL_DOCS
    WORKER_ALL_DOCS = all_docs_data

def calculate_matches(
    all_docs: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> QAMatchStats:
    """
    Validate passages retrieval results by calculating top k hits for a given set of answers and retrieved documents
    Args:
        all_docs: a dictionary of {id S-> (text, title)}
        answers: a list of answers, each answer is a list of strings
        closest_docs: a list of tuples, each tuple contains list of ids and list of scores
        workers_num: amount of parallel processes to validate results
        match_type: type of answer matching. Refer to has_answer docs
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    logger.info("all_docs size %d", len(all_docs))
    # global dpr_all_documents # Removed global
    # dpr_all_documents = all_docs # Removed assignment to global
    # logger.info("dpr_all_documents size %d", len(dpr_all_documents)) # Using len(all_docs) directly or removing

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    # Initialize WORKER_ALL_DOCS for the main process too, in case workers_num is 0 or 1
    # though typically workers_num > 1 for multiprocessing.
    global WORKER_ALL_DOCS
    WORKER_ALL_DOCS = all_docs

    processes = ProcessPool(processes=workers_num, initializer=init_worker_all_docs, initargs=(all_docs,))
    logger.info("Matching answers in top docs...")
    # Pass all_docs as the first argument to check_answer
    # No longer pass all_docs via partial, it will be accessed via global WORKER_ALL_DOCS
    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    logger.info("all_docs size %d", len(all_docs))
    questions_answers_docs = zip(answers, closest_docs)
    # Convert to list for len() call for logging, but pass the original iterator to map
    questions_answers_docs_list_for_logging = list(questions_answers_docs)
    logger.info("questions_answers_docs size %d", len(questions_answers_docs_list_for_logging))
    # Re-create the zip object (iterator) to be passed to processes.map
    # as the previous list conversion would have consumed it.
    questions_answers_docs_iterator = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs_iterator)

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def calculate_matches_from_meta(
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> QAMatchStats:

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    get_score_partial = partial(
        check_answer_from_meta,
        match_type=match_type,
        tokenizer=tokenizer,
        use_title=use_title,
        meta_compressed=meta_compressed,
    )

    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = questions_answers_docs

    # global dpr_all_documents # Removed global
    # Use the global WORKER_ALL_DOCS initialized in each worker
    global WORKER_ALL_DOCS
    if WORKER_ALL_DOCS is None:
        # This should ideally not happen if initializer worked
        logger.error("WORKER_ALL_DOCS is not initialized in worker process!")
        # Fallback or raise error, for now, let's try to make it obvious if it fails
        # Or, consider if the main process all_docs could be used, but that defeats the purpose.
        # For safety, return empty hits or raise an exception.
        # For this example, returning all False.
        return [False] * len(doc_ids)

    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = WORKER_ALL_DOCS.get(doc_id) # Use .get() for safety
        if doc is None:
            logger.warning(f"Document ID {doc_id} not found in WORKER_ALL_DOCS.")
            hits.append(False)
            continue

        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        if match_type == "kilt":
            if has_answer_kilt(answers, text):
                answer_found = True
        elif has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def check_answer_from_meta(
    questions_answers_docs,
    tokenizer,
    match_type,
    meta_body_idx: int = 1,
    meta_title_idx: int = 2,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (docs_meta, doc_scores) = questions_answers_docs

    hits = []

    for i, doc_meta in enumerate(docs_meta):

        text = doc_meta[meta_body_idx]
        title = doc_meta[meta_title_idx] if len(doc_meta) > meta_title_idx else ""
        if meta_compressed:
            text = zlib.decompress(text).decode()
            title = zlib.decompress(title).decode()

        if use_title:
            text = title + " . " + text
        answer_found = False
        if has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def calculate_chunked_matches(
    all_docs: Dict[object, TableChunk],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> QATableMatchStats:
    global dpr_all_documents
    dpr_all_documents = all_docs

    global dpr_all_tables
    dpr_all_tables = {}

    for key, table_chunk in all_docs.items():
        table_str, title, table_id = table_chunk
        table_chunks = dpr_all_tables.get(table_id, [])
        table_chunks.append((table_str, title))
        dpr_all_tables[table_id] = table_chunks

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)

    logger.info("Matching answers in top docs...")
    get_score_partial = partial(check_chunked_docs_answer, match_type=match_type, tokenizer=tokenizer)
    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)
    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    top_k_orig_hits = [0] * n_docs
    for s in scores:
        question_hits, question_orig_doc_hits = s
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

        best_hit = next((i for i, x in enumerate(question_orig_doc_hits) if x), None)
        if best_hit is not None:
            top_k_orig_hits[best_hit:] = [v + 1 for v in top_k_orig_hits[best_hit:]]

    return QATableMatchStats(top_k_hits, top_k_orig_hits, scores)


# -------------------- KILT eval ---------------------------------


def has_answer_kilt(answers, text) -> bool:
    text = normalize_kilt(text)
    for single_answer in answers:
        single_answer = normalize_kilt(single_answer)
        if single_answer in text:
            return True
    return False


# answer normalization
def normalize_kilt(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
