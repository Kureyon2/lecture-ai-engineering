# 演習1/main.py
# 親ディレクトリをパスに追加（演習1へのパス）
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "演習1")))

from main import run_model

def test_accuracy_threshold():
    accuracy, _ = run_model()
    assert accuracy >= 0.8, f"精度が低すぎ: {accuracy}"

def test_inference_time():
    _, inference_time = run_model()
    assert inference_time <= 10.0, f"推論時間が長すぎ: {inference_time}"