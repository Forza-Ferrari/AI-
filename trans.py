import pandas as pd
import json
from pathlib import Path

def merge_csvs_to_json(
    csv_dir: str,
    json_path: str,
    encoding: str = "utf-8",
    **read_csv_kwargs
):
    """
    将目录下所有 CSV 文件合并为一个 JSON 文件。

    参数：
      csv_dir：CSV 文件所在目录
      json_path：输出 JSON 文件路径
      encoding：CSV 文件编码（如 "utf-8" 或 "gb18030"）
      read_csv_kwargs：传给 pd.read_csv 的额外参数（如 delimiter, usecols 等）
    """
    csv_folder = Path(csv_dir)
    all_records = []

    # 遍历所有 .csv
    for csv_file in csv_folder.glob("*.csv"):
        print(f"读取 {csv_file.name} ...")
        df = pd.read_csv(csv_file, encoding=encoding, **read_csv_kwargs)
        # 将 DataFrame 每行转为 dict，追加到列表
        all_records.extend(df.to_dict(orient="records"))

    # 包装为 contexts
    output = {"contexts": all_records}

    # 写出 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"已生成 JSON：{json_path}")

if __name__ == "__main__":
    # 示例用法
    merge_csvs_to_json(
        csv_dir="data/",                   # 你的 CSV 所在文件夹
        json_path="data/contexts.json",
        encoding="gb18030",                # 如有需要可改为 utf-8
        # 如果需要指定分隔符或只读部分列：
        # delimiter=",",
        # usecols=["department", "title", "ask", "answer"],
    )
