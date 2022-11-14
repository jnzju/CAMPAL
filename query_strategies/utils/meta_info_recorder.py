import csv
import os
from typing import List, Any, Dict, OrderedDict


def cycle_info_recorder(path: str, info: List[Dict[str, Any]]):
    file_path = os.path.join(path, "info.csv")
    with open(file_path, "w", newline='') as f:
        header = list(info[0].keys())
        writer = csv.writer(f)
        writer.writerow(header)
        for info_elem in info:
            writer.writerow(list(info_elem.values()))
        f.close()


def cycle_info_reader(path: str) -> List[OrderedDict[str, Any]]:
    info = list()
    file_path = os.path.join(path, "info.csv")
    with open(file_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            row_out = dict(row)
            for key, val in row_out:
                row_out[key] = eval(val)
            info.append(row_out)
        f.close()
    return info
