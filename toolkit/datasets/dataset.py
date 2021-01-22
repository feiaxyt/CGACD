import os
from tqdm import tqdm

class Dataset(object):
    def __init__(self, name, dataset_root):
        self.name = name
        self.dataset_root = dataset_root
        self.videos = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, path, tracker_names, ):
        """
        Args:
            path: path to tracker results,
            tracker_names: list of tracker name
        """
        self.tracker_path = path
        self.tracker_names = []
        seq_nums = len(self.videos)
        for tracker in tracker_names:
            t_path = os.path.join(path, tracker)
            if 'VOT' in self.name:
                t_path = os.path.join(path, tracker, 'baseline')
            seqs = os.listdir(t_path)
            if len(seqs) == seq_nums:
                self.tracker_names.append(tracker)

        # for video in tqdm(self.videos.values(), 
        #         desc='loading tacker result', ncols=100):
        #     video.load_tracker(path, tracker_names)
