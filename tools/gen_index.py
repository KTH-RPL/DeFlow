import pickle


def filter_scene_items(file_path, scene_name, output_path=None):
    # 如果没有指定输出路径，就覆盖原文件
    if output_path is None:
        output_path = file_path

    # 读取原始数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 过滤数据
    filtered_data = [item for item in data if item[0] == scene_name]

    # 打印信息
    print(f"Original number of items: {len(data)}")
    print(f"Number of items after filtering: {len(filtered_data)}")

    # 保存筛选后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_data, f)

    print(f"Filtered data saved to: {output_path}")

    # 打印前5个项目（如果有的话）
    print("\nFirst 5 items after filtering:")
    for i, item in enumerate(filtered_data[:5]):
        print(f"Item {i}: {item}")


# 使用示例
if __name__ == "__main__":
    # 文件路径
    file_path = '/data/jiehao/dataset/deflow/demo_data/av2/preprocess_v2/sensor/vis_test/index_total.pkl'

    # 你想保留的场景名称
    scene_name = '25e5c600-36fe-3245-9cc0-40ef91620c22'  # 替换为你想要的场景名称

    # 可以指定新的输出文件名（可选）
    output_path = '/data/jiehao/dataset/deflow/demo_data/av2/preprocess_v2/sensor/vis_test/index_total.pkl'  # 如果不想覆盖原文件，可以指定新的文件名

    # 执行过滤
    filter_scene_items(file_path, scene_name, output_path)