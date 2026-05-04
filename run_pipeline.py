import argparse
from utils.logger import get_logger

log = get_logger("pipeline")

PHASES = ["ingestion", "features", "training", "all"]

def parse_args():
    parser = argparse.ArgumentParser(description="AI Code Detector Pipeline")
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Pipeline phase to run (default: all)"
    )
    return parser.parse_args()

def run_ingestion():
    log.info("=== Phase 1: Ingestion ===")
    from pipeline.ingestion.github_scraper   import run as scrape_github
    from pipeline.ingestion.gemini_generator import run as generate_ai
    scrape_github()
    generate_ai()

def run_features():
    log.info("=== Phase 2: Feature Engineering ===")
    from pipeline.processing.spark_processor import run as process
    process()

def run_training():
    log.info("=== Phase 3: Training ===")
    from pipeline.training.export import run as export
    from pipeline.training.train  import run as train
    export()
    train()

def main():
    args = parse_args()

    if args.phase in ("ingestion", "all"):
        run_ingestion()
    if args.phase in ("features", "all"):
        run_features()
    if args.phase in ("training", "all"):
        run_training()

    log.info("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()