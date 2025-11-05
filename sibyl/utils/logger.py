import sys
import os
import logging
import pathlib
from datetime import datetime, timedelta
from typing import Union

def start_capture(
    debug: bool,
    save_path: Union[str, pathlib.Path],
):
    if isinstance(save_path, str):
        save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = save_path.with_suffix(".log")

    logger = logging.getLogger("stdout_logger")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(save_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    class LoggerWriter:
        def __init__(self, level):
            self.level = level
            self._buffer = ""

        def write(self, message):
            self._buffer += message
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    logger.log(self.level, line)

        def flush(self):
            if self._buffer:
                logger.log(self.level, self._buffer)
                self._buffer = ""
        
        def isatty(self):
            return False

    sys.stdout = LoggerWriter(logging.INFO)
    sys.stderr = LoggerWriter(logging.ERROR)
    return logger

if __name__ == "__main__":
    
    REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

    logger = start_capture(
        debug=True,
        save_path = REPO_ROOT / "outputs" / "test_log.log",
    )
    print("This is a test message.")
    print("This is an error message.", file=sys.stderr)