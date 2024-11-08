import pickle

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 打印所有内容
def print_all_items(data):
    for i, item in enumerate(data):
        print(f"Item {i}: {item}")

# 打印前N个项目
def print_first_n_items(data, n=5):
    print(f"\nFirst {n} items:")
    for i, item in enumerate(data[:n]):
        print(f"Item {i}: {item}")

file_path = '/data/jiehao/dataset/deflow/demo_data/av2/preprocess_v2/sensor/vis/index_total.pkl'
data = read_pickle(file_path)

# 打印基本信息
print(f"Total number of items: {len(data)}")
print(f"Type of data: {type(data)}")
print(f"Type of first item: {type(data[0])}")

# 打印前5个项目
print_first_n_items(data)

# 如果你想保存为更易读的格式，比如txt
with open('output.txt', 'w') as f:
    for item in data:
        f.write(f"{item}\n")