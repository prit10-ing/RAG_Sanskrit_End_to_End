"""
Main File for RAG Pipeline

Commands:

python app.py --ingest
python app.py --query "your question"
python app.py
"""

import os
import sys
import argparse
import logging

from src.utils.logger import setup_logging
from src.pipeline.rag_pipeline import RAGPipeline

from src.config import (
    DATA_FOLDER,
    VECTOR_DB_DIR,
)

# setup logger
setup_logging()

logger = logging.getLogger(__name__)


# =========================
# Check Vector Store
# =========================

def check_vector_store():

    return (
        os.path.isdir(VECTOR_DB_DIR)
        and
        len(os.listdir(VECTOR_DB_DIR)) > 0
    )


# =========================
# Banner
# =========================

def print_banner():

    print("\n==============================")
    print("      RAG PIPELINE APP")
    print("==============================\n")


# =========================
# Ingestion Mode
# =========================

def run_ingest(pipeline):

    try:

        logger.info("Starting ingestion")

        pipeline.ingest()

        print(
            f"\nVector store saved in: "
            f"{VECTOR_DB_DIR}\n"
        )

    except Exception as e:

        logger.error(
            f"Ingestion Error: {e}"
        )

        print(f"\nError: {e}\n")

        sys.exit(1)


# =========================
# Single Query Mode
# =========================

def run_query(
    pipeline,
    question,
):

    # check vector db
    if not check_vector_store():

        print(
            "\nVector store not found."
        )

        print(
            "Run ingestion first.\n"
        )

        sys.exit(1)

    try:

        answer = pipeline.query(
            question
        )

        print("\nAnswer:\n")

        print(answer)

        print()

    except Exception as e:

        logger.error(
            f"Query Error: {e}"
        )

        print(f"\nError: {e}\n")


# =========================
# Interactive Chat Mode
# =========================

def run_chat(pipeline):

    # check vector db
    if not check_vector_store():

        print(
            "\nVector store not found."
        )

        print(
            "Run ingestion first.\n"
        )

        sys.exit(1)

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        try:

            question = input(
                "You: "
            ).strip()

            if question.lower() == "exit":

                print("\nGoodbye!\n")

                break

            if question == "":
                continue

            answer = pipeline.query(
                question
            )

            print(f"\nBot: {answer}\n")

        except KeyboardInterrupt:

            print("\nGoodbye!\n")

            break

        except Exception as e:

            logger.error(
                f"Chat Error: {e}"
            )

            print(f"\nError: {e}\n")


# =========================
# Main Function
# =========================

def main():

    print_banner()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ingest",
        action="store_true",
    )

    parser.add_argument(
        "--query",
        type=str,
    )

    args = parser.parse_args()

    pipeline = RAGPipeline(
        data_folder=DATA_FOLDER
    )

    # ingestion mode
    if args.ingest:

        run_ingest(pipeline)

    # single query mode
    elif args.query:

        run_query(
            pipeline,
            args.query,
        )

    # interactive mode
    else:

        run_chat(pipeline)


# =========================
# Start Application
# =========================

if __name__ == "__main__":

    main()