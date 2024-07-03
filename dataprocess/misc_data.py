import numpy as np
import pickle, h5py, os, time
from pathlib import Path
from tqdm import tqdm

def create_reading_index(data_dir: Path):
    start_time = time.time()
    data_index = []
    for file_name in tqdm(os.listdir(data_dir), ncols=100):
        if not file_name.endswith(".h5"):
            continue
        scene_id = file_name.split(".")[0]
        timestamps = []
        with h5py.File(data_dir/file_name, 'r') as f:
            timestamps.extend(f.keys())
        timestamps.sort(key=lambda x: int(x)) # make sure the timestamps are in order
        for timestamp in timestamps:
            data_index.append([scene_id, timestamp])

    with open(data_dir/'index_total.pkl', 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create reading index Successfully, cost: {time.time() - start_time:.2f} s")

class SE2:

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize.
        Args:
            rotation: np.ndarray of shape (2,2).
            translation: np.ndarray of shape (2,1).
        Raises:
            ValueError: if rotation or translation do not have the required shapes.
        """
        assert rotation.shape == (2, 2)
        assert translation.shape == (2, )
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = np.eye(3)
        self.transform_matrix[:2, :2] = self.rotation
        self.transform_matrix[:2, 2] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(2) transformation to point_cloud.
        Args:
            point_cloud: np.ndarray of shape (N, 2).
        Returns:
            transformed_point_cloud: np.ndarray of shape (N, 2).
        Raises:
            ValueError: if point_cloud does not have the required shape.
        """
        assert point_cloud.ndim == 2
        assert point_cloud.shape[1] == 2
        num_points = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :2]

    def inverse(self) -> "SE2":
        """Return the inverse of the current SE2 transformation.
        For example, if the current object represents target_SE2_src, we will return instead src_SE2_target.
        Returns:
            inverse of this SE2 transformation.
        """
        return SE2(rotation=self.rotation.T,
                   translation=self.rotation.T.dot(-self.translation))

    def inverse_transform_point_cloud(self,
                                      point_cloud: np.ndarray) -> np.ndarray:
        """Transform the point_cloud by the inverse of this SE2.
        Args:
            point_cloud: Numpy array of shape (N,2).
        Returns:
            point_cloud transformed by the inverse of this SE2.
        """
        return self.inverse().transform_point_cloud(point_cloud)

    def compose(self, right_se2: "SE2") -> "SE2":
        """Multiply this SE2 from right by right_se2 and return the composed transformation.
        Args:
            right_se2: SE2 object to multiply this object by from right.
        Returns:
            The composed transformation.
        """
        chained_transform_matrix = self.transform_matrix.dot(
            right_se2.transform_matrix)
        chained_se2 = SE2(
            rotation=chained_transform_matrix[:2, :2],
            translation=chained_transform_matrix[:2, 2],
        )
        return chained_se2
    