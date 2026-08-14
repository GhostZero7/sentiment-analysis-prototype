from __future__ import annotations

import subprocess
import sys


def test_sentiment_package_does_not_eagerly_load_transformers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.sentiment; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'src.sentiment.roberta_analyzer' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
