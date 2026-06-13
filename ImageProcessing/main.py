import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from ImageProcessing.runtime import Runtime


def main() -> None:
    Runtime().run()


if __name__ == "__main__":
    main()
