import json

if __name__ == "__main__":
    with open("test.json", "r") as f:
        data = json.load(f)
        ## 计算其中matrix字段的矩阵的shape
        matrix = data["matrix"]
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        print(f"Matrix shape: {rows} rows x {cols} cols")