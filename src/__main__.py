import fire
import sys
import json
from rich import print
from .pipeline import Pipeline


def main() -> None:
    """Parse command-line arguments and run the RAG CLI pipeline.

    Returns:
        None: Does not return a value.
    """
    try:
        fire.Fire(Pipeline, name="RAG")
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        raise SystemExit("Program exit successfully")
    except BaseException as err:
        print(err)


if __name__ == "__main__":
    main()
