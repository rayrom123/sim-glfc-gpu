import torch
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import os
import math
import re


CLIENT_TASK_RE = re.compile(r'^client_?(\d+)_task_?(\d+)\.pt$')


def extract_xy(obj):
    if isinstance(obj, dict):
        data = obj.get('x', torch.tensor([]))
        targets = obj.get('y', torch.tensor([]))
    else:
        data, targets = obj
    return data, targets


def task_id_for_index(task_ids, task_index):
    if task_ids and 0 <= task_index < len(task_ids):
        return task_ids[task_index]
    return task_index + 1


def discover_task_ids(root_dir):
    if not root_dir or not osp.isdir(root_dir):
        return []

    task_ids = set()
    for filename in os.listdir(root_dir):
        match = CLIENT_TASK_RE.match(filename)
        if match:
            task_ids.add(int(match.group(2)))
    return sorted(task_ids)


def discover_label_plan(root_dir):
    task_ids = discover_task_ids(root_dir)
    labels_by_task = []

    for task_id in task_ids:
        labels = set()
        for filename in os.listdir(root_dir):
            match = CLIENT_TASK_RE.match(filename)
            if not match or int(match.group(2)) != task_id:
                continue

            filepath = osp.join(root_dir, filename)
            obj = torch.load(filepath, map_location='cpu', weights_only=False)
            _, targets = extract_xy(obj)
            if len(targets) == 0:
                continue
            labels.update(int(x) for x in torch.unique(torch.as_tensor(targets)).tolist())

        labels_by_task.append(sorted(labels))

    output_dims = []
    learned_labels_by_task = []
    learned = set()
    for labels in labels_by_task:
        learned.update(labels)
        learned_labels = sorted(learned)
        learned_labels_by_task.append(learned_labels)
        output_dims.append(max(learned_labels) + 1 if learned_labels else 0)

    return {
        'task_ids': task_ids,
        'labels_by_task': labels_by_task,
        'learned_labels_by_task': learned_labels_by_task,
        'output_dims': output_dims,
    }

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
        self.current_task_index = 0
        self.task_ids = discover_task_ids(root_dir)
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
        data, targets = extract_xy(obj)

        datas, labels = [], []
        if isinstance(classes, dict) and 'labels' in classes:
            labels_to_eval = sorted(set(int(label) for label in classes['labels']))
        else:
            labels_to_eval = range(classes[0], classes[1])

        for label in labels_to_eval:
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

        return extract_xy(obj)

    def get_task_file_id(self, task_index):
        return task_id_for_index(self.task_ids, task_index)

    def load_task_by_index(self, task_index):
        return self.load_task(self.get_task_file_id(task_index))
        
    def set_task(self, task_id):
        self.current_task_index = task_id
        self.current_task = self.get_task_file_id(task_id)

    def _sample_previous_task_data(self, percent=0.01, seed=2021, min_samples=1):
        datas, labels = [], []
        self.last_replay_counts = {}

        if self.current_task_index <= 0 or percent <= 0:
            return datas, labels

        task_id = self.get_task_file_id(self.current_task_index - 1)
        data, targets = self.load_task(task_id)
        if len(data) == 0:
            self.last_replay_counts[task_id] = 0
            return datas, labels

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
