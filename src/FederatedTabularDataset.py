import torch
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import math

class FederatedTabularDataset(Dataset):
    def __init__(self, client_id, root_dir='../federated_data', test_file='../30_test_data.pt', transform=None, test=False):
        self.client_id = client_id
        self.root_dir = root_dir
        self.test_file = test_file
        self.transform = transform
        self.Test = test
        
        self.TrainData = np.array([])
        self.TrainLabels = np.array([])
        self.TestData = np.array([])
        self.TestLabels = np.array([])
        self.current_task = 0
        self.last_replay_counts = {}

    def concatenate(self, datas, labels):
        if not datas:
            return np.array([]), np.array([])
        
        # Ensure we have consistent numpy arrays and flatten 1D arrays for concat
        datas_np = [d.numpy() if isinstance(d, torch.Tensor) else d for d in datas]
        labels_np = [l.numpy() if isinstance(l, torch.Tensor) else l for l in labels]
        
        non_empty = [(d, l) for d, l in zip(datas_np, labels_np) if len(d) > 0]
        if len(non_empty) == 0:
             return np.array([]), np.array([])
             
        datas_filtered, labels_filtered = zip(*non_empty)
        con_data = datas_filtered[0]
        con_label = labels_filtered[0]
        for i in range(1, len(datas_filtered)):
            con_data = np.concatenate((con_data, datas_filtered[i]), axis=0)
            con_label = np.concatenate((con_label, labels_filtered[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        # We load the entire test dataset
        obj = torch.load(self.test_file, map_location='cpu', weights_only=False)
        
        # Support both old tuple format and new dict format
        if isinstance(obj, dict):
            data = obj.get('x', torch.tensor([]))
            targets = obj.get('y', torch.tensor([]))
        else:
            data, targets = obj

        datas, labels = [], []
        # Support class filtering if needed
        for label in range(classes[0], classes[1]):
            subset_data = data[targets == label]
            if len(subset_data) > 0:
                datas.append(subset_data)
                labels.append(np.full((subset_data.shape[0]), label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def load_task(self, task_id):
        # Handle both client_x_task_y.pt and clientx_tasky.pt (for compatibility)
        filepath = osp.join(self.root_dir, f'client_{self.client_id}_task_{task_id}.pt')
        if not osp.exists(filepath):
            filepath = osp.join(self.root_dir, f'client{self.client_id}_task{task_id}.pt')
            
        if not osp.exists(filepath):
            return torch.tensor([]), torch.tensor([])
            
        obj = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # New data format uses dict with 'x' and 'y' keys
        if isinstance(obj, dict):
            data = obj.get('x', torch.tensor([]))
            targets = obj.get('y', torch.tensor([]))
        else:
            # Old data format returns a tuple (data, targets)
            data, targets = obj
            
        return data, targets
        
    def set_task(self, task_id):
        self.current_task = task_id + 1

    def _sample_previous_task_data(self, percent=0.01, seed=2021, min_samples=1):
        datas, labels = [], []
        self.last_replay_counts = {}

        if self.current_task <= 1 or percent <= 0:
            return datas, labels

        for task_id in range(1, self.current_task):
            data, targets = self.load_task(task_id)
            if len(data) == 0:
                self.last_replay_counts[task_id] = 0
                continue

            sample_count = int(math.ceil(len(data) * percent))
            sample_count = max(min_samples, sample_count)
            sample_count = min(len(data), sample_count)

            generator = torch.Generator()
            generator.manual_seed(seed + self.client_id * 100003 + task_id)
            indices = torch.randperm(len(data), generator=generator)[:sample_count]
            index_values = indices if isinstance(data, torch.Tensor) else indices.numpy()

            datas.append(data[index_values])
            labels.append(targets[index_values])
            self.last_replay_counts[task_id] = sample_count

        return datas, labels

    def getTrainData(self, classes, exemplar_set=[], exemplar_label_set=[],
                     previous_task_replay_percent=0.0, replay_seed=2021):
        datas, labels = [], []
        
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            # Assumes all exemplars have the same size as the first one
            length = len(exemplar_set[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        data, targets = self.load_task(self.current_task)
        
        if len(data) > 0:
            # We add all instances from the task ignoring random class subsets
            datas.append(data)
            labels.append(targets)

        replay_datas, replay_labels = self._sample_previous_task_data(
            percent=previous_task_replay_percent,
            seed=replay_seed,
        )
        datas.extend(replay_datas)
        labels.extend(replay_labels)
            
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def get_image_class(self, label):
        # Even though they're not images, it extracts instances by class wrapper
        if len(self.TrainData) > 0:
            return self.TrainData[self.TrainLabels == label]
        return np.array([])

    def __getitem__(self, index):
        if len(self.TrainData) > 0:
            img = self.TrainData[index]
            target = self.TrainLabels[index]
        elif len(self.TestData) > 0:
             img = self.TestData[index]
             target = self.TestLabels[index]
             
        # Optional transformations
        if self.transform:
             img = self.transform(img)
             
        # Returns float32 tensors for model compatibility
        return index, torch.tensor(img, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        elif len(self.TestData) > 0:
            return len(self.TestData)
        return 0
