import pickle


# 基础打开 pkl 文件
def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# 打印 pkl 内容的结构
def print_structure(data, level=0):
    prefix = "  " * level
    if isinstance(data, dict):
        print(f"{prefix}Dictionary with {len(data)} items:")
        for key, value in data.items():
            print(f"{prefix}Key: {key}")
            print_structure(value, level + 1)
    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items:")
        if len(data) > 0:
            print_structure(data[0], level + 1)
    elif isinstance(data, (int, float, str, bool)):
        print(f"{prefix}Value: {data}")
    else:
        print(f"{prefix}Type: {type(data)}")


# 主函数
def main():
    # 替换为你的 pkl 文件路径
    file_path = '/data/jiehao/dataset/deflow/demo_data/av2/preprocess_v2/sensor/vis/index_total.pkl'

    try:
        # 读取数据
        data = read_pickle(file_path)

        # 打印基本信息
        print(f"Successfully loaded: {file_path}")
        print("\nData structure:")
        print_structure(data)

        # 如果是字典，打印所有键
        if isinstance(data, dict):
            print("\nAvailable keys:")
            for key in data.keys():
                print(f"- {key}")

        # 如果需要访问特定的值
        # 例如: if 'specific_key' in data:
        #          print(data['specific_key'])

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()