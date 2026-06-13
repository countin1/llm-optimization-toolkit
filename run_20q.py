"""
快速测试：20 题 x 3 模板

等同于: python run_100q.py --max-questions 20 --output quick_test_results.json
"""
import subprocess
import sys

sys.exit(subprocess.call([
    sys.executable, 'run_100q.py',
    '--max-questions', '20',
    '--output', 'quick_test_results.json',
]))
