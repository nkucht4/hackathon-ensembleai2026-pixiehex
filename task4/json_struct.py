import json
import glob
from collections import defaultdict

def analyze_json_files(folder_path, file_pattern="*.json"):
    files = glob.glob(f"{folder_path}/{file_pattern}")
    data_list = []

    count = 0

    for file in files:
        count += 1
        print(f"Processing file {count}: {file}")
        if count > 500:
            break
            pass
        with open(file, "r", encoding="utf-8") as f:
            data_list.append(json.load(f))

    def merge_types(val_list):
        types = set()
        values = set()
        min_val = float('inf')
        max_val = float('-inf')

        for val in val_list:
            types.add(type(val).__name__)
            if isinstance(val, (int, float)):
                min_val = min(min_val, val)
                max_val = max(max_val, val)
            elif isinstance(val, str) or isinstance(val, bool):
                values.add(val)
            elif isinstance(val, list):
                types.add(type(val).__name__)
                if isinstance(val, (int, float)):
                    min_val = min(min_val, val)
                    max_val = max(max_val, val)
                elif isinstance(val, str) or isinstance(val, bool):
                    values.add(val)
                elif isinstance(val, list):
                    values.add(f"list of length {len(val)}")
                elif isinstance(val, dict):
                    values.add("dict")
            elif isinstance(val, dict):
                values.add("dict")

        result = {"types": list(types)}
        if values:
            if len(values) <= 20:
                result["values"] = list(values)
            else:
                result["values"] = "too many to list"
        if min_val != float('inf') and max_val != float('-inf'):
            result["range"] = [min_val, max_val]
        return result

    def recursive_analyze(items):
        if not isinstance(items, list):
            items = [items]

        analysis = {}
        all_keys = set()
        dict_items = [i for i in items if isinstance(i, dict)]
        list_items = [i for i in items if isinstance(i, list)]
        scalar_items = [i for i in items if not isinstance(i, (dict, list))]

        if scalar_items:
            return merge_types(scalar_items)

        for item in dict_items:
            all_keys.update(item.keys())

        for key in all_keys:
            key_values = [item[key] for item in dict_items if key in item]
            analysis[key] = recursive_analyze(key_values)

        if list_items:
            analysis["_list_elements"] = recursive_analyze([el for l in list_items for el in l])

        return analysis

    final_analysis = recursive_analyze(data_list)
    return final_analysis

folder_path = "./data/train"
analysis = analyze_json_files(folder_path)

import pprint
pprint.pprint(analysis, width=120)