import h5py

# 打开 .h5 文件
file_path = '/data/jiehao/dataset/deflow/av2/preprocess/sensor/vis/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68.h5'

with h5py.File(file_path, 'r') as file:
    # 打印文件结构
    def print_structure(name, obj):
        print(name)
        # if isinstance(obj, h5py.Dataset):
        #     print("    Shape:", obj.shape)
        #     print("    Type:", obj.dtype)


    file.visititems(print_structure)

    # 读取特定数据集
    # dataset = file['dataset_name'][()]
    # print(dataset)

    # 如果你知道具体的数据集名称，可以直接访问
    # 例如：data = file['group_name']['dataset_name'][()]