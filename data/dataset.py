import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision

data_dir = os.path.dirname(os.path.abspath(__file__))
mean = [0.5]
std = [0.5]


class dataset_transform(Dataset):
    def __init__(self, dataset, select_classes=None, target_transform=None):
        self.dataset = dataset
        self.target_transform = target_transform
        if select_classes is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = [idx for idx, (_, y) in enumerate(dataset) if y in select_classes]
            self.transform_dict = dict(zip(select_classes, list(range(len(select_classes)))))
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]  
        label = self.dataset[self.indices[idx]][1]
        if self.target_transform=='reindex':
            label = self.transform_dict[label]
        elif self.target_transform=='open':
            label = 999
        else:
            label = label
        return (image, label)
    def __len__(self):
        return len(self.indices)


class TrafficArrayDataset(Dataset):
    """Dataset wrapper used by the paper-aligned 8:1:1 split."""

    def __init__(self, data, targets, transform=None, select_classes=None, target_transform=None):
        self.data = np.asarray(data)
        self.targets = np.asarray(targets)
        self.transform = transform
        self.target_transform = target_transform
        if select_classes is None:
            select_classes = sorted(np.unique(self.targets).tolist())
        self.transform_dict = {label: index for index, label in enumerate(select_classes)}

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx].astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.targets[idx])
        if self.target_transform == 'reindex':
            label = self.transform_dict[label]
        elif self.target_transform == 'open':
            label = 999
        return image, label

    def __len__(self):
        return len(self.targets)


DATASET_FILES = {
    'mal': ('mal_32_1c_train.npz', 'mal_32_1c_test.npz'),
    'USTC': ('USTC_1c_train.npz', 'USTC_1c_test.npz'),
    'combined_USTC_mal': ('combined_train_data.npz', 'combined_test_data.npz'),
}


def _load_npz_arrays(dataset):
    """Load and combine the repository's existing train/test NPZ files."""
    if dataset not in DATASET_FILES:
        raise ValueError('Unsupported dataset: ' + dataset)

    arrays = []
    targets = []
    for filename in DATASET_FILES[dataset]:
        path = os.path.join(data_dir, 'dataset', filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                '{} is required. Download the dataset and place it in data/dataset.'.format(path)
            )
        loaded = np.load(path)
        arrays.append(np.asarray(loaded['data']))
        targets.append(np.asarray(loaded['target']))

    data = np.vstack(arrays).reshape(-1, 32, 32)
    labels = np.concatenate(targets).astype(np.int64)
    return data, labels


def stratified_split_indices(targets, seed=2022):
    """Return disjoint stratified train/validation/test indices in an 8:1:1 ratio."""
    targets = np.asarray(targets)
    indices = np.arange(len(targets))
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=targets,
    )
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.5,
        random_state=seed + 1,
        stratify=targets[holdout_indices],
    )
    return train_indices, val_indices, test_indices


def get_dataset_splits(dataset, select_classes=None, target_transform=None, seed=2022):
    """Create paper-aligned train/validation/test datasets.

    The two distributed NPZ partitions are first combined, then re-split with
    class stratification into 80% training, 10% validation, and 10% testing.
    """
    data, targets = _load_npz_arrays(dataset)
    if select_classes is not None:
        mask = np.isin(targets, np.asarray(select_classes))
        data = data[mask]
        targets = targets[mask]

    train_indices, val_indices, test_indices = stratified_split_indices(targets, seed=seed)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    dataset_args = {
        'select_classes': select_classes,
        'target_transform': target_transform,
    }
    train_set = TrafficArrayDataset(
        data[train_indices], targets[train_indices], transform=transform_train, **dataset_args
    )
    val_set = TrafficArrayDataset(
        data[val_indices], targets[val_indices], transform=transform_eval, **dataset_args
    )
    test_set = TrafficArrayDataset(
        data[test_indices], targets[test_indices], transform=transform_eval, **dataset_args
    )
    return train_set, val_set, test_set


class OPENWORLDmal(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDmal, self).__init__(root, train, transform, target_transform, download=True)

        if train:
            loaded_data = np.load(os.path.join(data_dir, 'dataset', 'mal_32_1c_train.npz'))
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)
        else:
            loaded_data = np.load(os.path.join(data_dir, 'dataset', 'mal_32_1c_test.npz'))
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)


class combined_USTC_mal(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(combined_USTC_mal, self).__init__(root, train, transform, target_transform, download=False)
        if train:
            loaded_data = np.load(os.path.join(data_dir, 'dataset', 'combined_train_data.npz'))
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)
        else:
            loaded_data = np.load(os.path.join(data_dir, 'dataset', 'combined_test_data.npz'))
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)

class USTC(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(USTC, self).__init__(root, train, transform, target_transform, download=False)

        if train:
            loaded_data = np.load(os.path.join(data_dir, 'dataset', 'USTC_1c_train.npz'))
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)
        else:
            loaded_data = np.load(os.path.join(data_dir, 'dataset', 'USTC_1c_test.npz'))
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)


def get_dataset(dataset, train=False, select_classes=None, target_transform=None):
    if dataset == 'mal':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = OPENWORLDmal(root=data_dir, train=True, transform=transform_train, download=False)
        test_set = OPENWORLDmal(root=data_dir, train=False, transform=transform_test, download=False)
    
    elif dataset == 'combined_USTC_mal':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = combined_USTC_mal(root=data_dir, train=True, transform=transform_train, download=False)
        test_set = combined_USTC_mal(root=data_dir, train=False, transform=transform_test, download=False)
    
    elif dataset == 'USTC':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = USTC(root=data_dir, train=True, transform=transform_train, download=False)
        test_set = USTC(root=data_dir, train=False, transform=transform_test, download=False)
    
    else:
        raise ValueError('Unsupported dataset: ' + dataset)
    
    if train:
        return dataset_transform(train_set, select_classes, target_transform)
    else:
        return dataset_transform(test_set, select_classes, target_transform)



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]
    total_classes = list(range(10))
    unknown_classes = splits[0]
    known_classes = list(set(total_classes) - set(unknown_classes))

    known_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6]
    train_set = get_dataset('mal', True, known_classes, 'reindex')
    test_set = get_dataset('mal', False, known_classes, 'reindex')
    open_set = get_dataset('mal', False, unknown_classes, 'open')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, drop_last = True)

