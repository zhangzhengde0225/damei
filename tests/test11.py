"""
统计一个目录下所有代码的行数和字符数
"""
import damei as dm
from pathlib import Path

root_path = f'{Path(__file__).parent.parent}/damei'

num_lines, num_chars = dm.misc.count_lines_and_chars(
    root_path, suffixes='.py', show_detail=True)

print(num_lines, num_chars)
