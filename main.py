from src.config import parse_args
from src.pipeline import run_pipeline


def main() -> None:
    config = parse_args()
    output_path = run_pipeline(config)
    print(f"Done. Output video: {output_path}")


if __name__ == "__main__":
    main()
