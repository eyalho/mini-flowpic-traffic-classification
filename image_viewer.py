import logging
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import streamlit as st
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("info")

import random

from torchvision.transforms import transforms
import statistics
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset


# ### Utilsg

# In[3]:


def get_times_and_sized(df, direction, trim_time):
    if direction == "up":
        condition = (df["pkt_direction"] == 1) & (df["pkt_relative_time"] < trim_time) & (
                df["pkt_relative_time"] > 0)
    elif direction == "down":
        condition = (df["pkt_direction"] == 0) & (df["pkt_relative_time"] < trim_time) & (
                df["pkt_relative_time"] > 0)
    elif direction == "both":
        condition = ((df["pkt_relative_time"] < trim_time) & (df["pkt_relative_time"] > 0))
    else:
        raise ValueError(f"Invalid direction ({direction})")

    pkt_times = df[condition].pkt_relative_time
    pkt_sizes = df[condition].pkt_size
    return pkt_times, pkt_sizes


def is_flow_df_valid(df, direction, trim_time):
    """
    Valid Flow should have at least one packet. If not, it is not valid.
    :param df: flow df
    :param direction: flow direction
    :param trim_time: max sample time of flow
    :return: True for valid flow and False for invalid flow
    """
    pkt_times, pkt_sizes = get_times_and_sized(df, direction, trim_time)
    if len(pkt_sizes) == 0 or len(pkt_times) == 0:
        return False
    return True


def df_take_n_samples_per_class(df: pd.DataFrame, class_col_name: str, n=None, random_state=0):
    max_n = min(df[class_col_name].value_counts())

    if n is None or n > max_n:
        n = max_n
        print(f"df_take_n_samples_per_class setting n={n}")

    first_n_per_class_rows = df.groupby(class_col_name).sample(n=n,
                                                               random_state=random_state, replace=False)
    rest_rows = df[~df.index.isin(first_n_per_class_rows.index)]
    return first_n_per_class_rows, rest_rows


# ### Augmentatinos


class ContrastiveTransformations:

    def __init__(self, hp, n_views=2):
        """
        :param hp: hyper parameters for the augmentations
        :param n_views: The contrastive learning framework can easily be extended to have more positive examples
            by sampling more than two augmentations of the same image.
            However, the most efficient training is usually obtained by using only two.
        """
        self.hp = hp
        self.n_views = n_views
        self.retry_idx = 0  # we want to make sure that the augmented flow is valid. we would retry if it is not valid.

    def torch_vision_transform(self, img):
        hp = self.hp
        img = self.cv_color_jitter(img, hp['color_jitter_apply_prob'],
                                   hp['color_jitter_brightness'],
                                   hp['color_jitter_contrast'],
                                   hp['color_jitter_saturation'],
                                   hp['color_jitter_hue'])

        img = self.cv_flip_time_axis(img, hp['flip_time_axis_apply_prob'])

        img = self.cv_rotate_img(img, hp['rotate_img_apply_prob'],
                                 hp['rotate_img_min_angle'],
                                 hp['rotate_img_max_angle'])

        t_tensor = transforms.ToTensor()
        return t_tensor(img)

    def data_frame_transform(self, df: pd.DataFrame):
        """
        :param df: the original data frame
        :return: modified data frame with the augmentation applied
        """
        while True:
            try:
                return self._data_frame_transform(df)
            except (ValueError, IndexError) as e:
                if self.retry_idx > 1000:
                    raise FlowPicAugRetryException(
                        f'Already retired {self.retry_idx} for {self.hp.get("aug_net", "aug_net")}\n{e}\n{df}')
                self.retry_idx += 1

    def _data_frame_transform(self, origin_df: pd.DataFrame):
        df = origin_df.copy()
        if self.hp['pkt_time_min_mult_factor'] != 1 or self.hp['pkt_time_max_mult_factor'] != 1:
            mult_factor = random.uniform(self.hp['pkt_time_min_mult_factor'],
                                         self.hp['pkt_time_max_mult_factor'])
            df = self.multiple_time_axis_by_constant(df, mult_factor)

        if self.hp['pkt_time_min_move_const'] != 0 or self.hp['pkt_time_max_move_const'] != 0:
            forward_time = random.uniform(self.hp['pkt_time_min_move_const'],
                                          self.hp['pkt_time_max_move_const'])
            df = self.move_time_axis_by_constant(df, forward_time)

        if self.hp['cutout_time_window_size'] != 0:
            df = self.cut_out_time_axis_where_there_are_pkts(df, self.hp['flow_max_duration'],
                                                             self.hp['cutout_time_window_size'])

        # ret_df = self.fix_time_axis_to_start_by_first_pkt_and_trim(df, self.hp['flow_max_duration'])
        ret_df = df
        if not is_flow_df_valid(ret_df, self.hp['flow_direction'], self.hp['flow_max_duration']):
            logging.debug(
                f"invalid augmented_df: Retry={self.retry_idx}"
                f"(origin={origin_df.shape[0]})")
            raise ValueError
        self.retry_idx = 0
        return ret_df

    @staticmethod
    def multiple_time_axis_by_constant(time_series_df: pd.DataFrame, factor: float) -> pd.DataFrame:
        time_series_df = time_series_df.copy()
        time_series_df.loc[:, 'pkt_relative_time'] *= factor
        return time_series_df

    @staticmethod
    def move_time_axis_by_constant(time_series_df: pd.DataFrame, added_time: float) -> pd.DataFrame:
        time_series_df = time_series_df.copy()
        time_series_df.loc[:, 'pkt_relative_time'] += added_time
        return time_series_df

    @staticmethod
    def fix_time_axis_to_start_by_first_pkt_and_trim(time_series_df: pd.DataFrame, trim_time: int) -> pd.DataFrame:
        time_series_df = time_series_df.copy()
        time_series_df = time_series_df[time_series_df['pkt_relative_time'] >= 0]
        min_time_pkt = time_series_df['pkt_relative_time'].min()
        time_series_df.loc[:, 'pkt_relative_time'] -= min_time_pkt
        time_series_df = time_series_df[time_series_df['pkt_relative_time'] <= trim_time]

        return time_series_df

    @staticmethod
    def cut_out_time_axis_where_there_are_pkts(time_series_df: pd.DataFrame, max_time, cut_size) -> pd.DataFrame:
        time_series_df = time_series_df.copy()
        possible_times = [t for t in time_series_df['pkt_relative_time'] if (t > 0) and (t < max_time)]
        t = random.choice(possible_times)
        time_series_df = time_series_df[(time_series_df['pkt_relative_time'] < t) |
                                        (time_series_df['pkt_relative_time'] >= t + cut_size)]
        return time_series_df

    @staticmethod
    def cv_rotate_img(img, p, angle_min: int, angle_max: int):
        if random.random() < p:
            angle = random.randint(angle_min, angle_max)
            img = img.rotate(angle)
        return img

    @staticmethod
    def cv_flip_time_axis(img, p):
        return transforms.RandomHorizontalFlip(p)(img)

    @staticmethod
    def cv_color_jitter(img, p, brightness, contrast, saturation, hue):
        t = transforms.ColorJitter(brightness, contrast, saturation, hue)
        if random.random() < p:
            img = t(img)
        return img


class FlowPicAugRetryException(Exception):
    pass


# ### Dataset

class absFlowPicDataset(ABC, Dataset):
    MINIMUM_PKTS_FLOWPIC = 10
    UP = "up"
    DOWN = "down"
    BOTH = "both"

    @property
    @classmethod
    @abstractmethod
    def labels_to_indices_dict(cls) -> dict:
        # {label: idx for idx, label in enumerate(self.labels_set)}
        pass

    @property
    @classmethod
    @abstractmethod
    def indices_to_labels_dict(cls) -> dict:
        #  {idx: label for idx, label in enumerate(self.labels_set)}
        pass

    @property
    @classmethod
    @abstractmethod
    def labels_ordered_list(cls) -> list:
        pass

    def __init__(self,
                 time_series_csv_paths,
                 labels,
                 direction: str,
                 flow_max_duration: int,
                 image_size: int = 32,
                 convert_to_image_with_max_value_255=True,
                 transform: Optional[ContrastiveTransformations] = None,
                 lazy_loading: bool = False
                 ):
        super().__init__()
        self.time_series_csv_paths_list = time_series_csv_paths
        self.labels = labels
        self.direction = direction
        self.image_size = image_size
        self.convert_to_image_with_max_value_255 = convert_to_image_with_max_value_255
        self.contrastive_transform: Optional[ContrastiveTransformations] = transform
        self.trim_time: int = flow_max_duration
        self.lazy_loading = lazy_loading

        assert len(self.labels) == len(self.time_series_csv_paths_list)
        self.targets_list: List[torch.Tensor] = [Tensor([self.labels_to_indices_dict[label]]).type(torch.LongTensor) for
                                                 label in self.labels]

        self.time_series_df_list: List[pd.DataFrame] = []
        if not self.lazy_loading:
            # This would load into RAM the all dataframes, which is faster but might cause OOM errors
            for time_series_csv_path in self.time_series_csv_paths_list:
                time_series_df = pd.read_csv(time_series_csv_path)
                self.time_series_df_list.append(time_series_df)

        logging.info(f"Total {len(self.time_series_csv_paths_list)} images")
        logging.info(f"pre-loaded into RAM {len(self.time_series_df_list)} dataframe of sessions"
                     f"(with up to {self.trim_time}sec)")
        logging.info(f"labels distribution:\n{pd.DataFrame(labels).value_counts()}")

    def __len__(self):
        return len(self.time_series_csv_paths_list)

    def __getitem__(self, index):
        time_series_df = self._load_time_series_df(index)
        images = self.create_images(time_series_df, self.direction, self.trim_time, self.image_size)
        target = self.targets_list[index]
        return images, target

    def get_statistical_features(self, index):
        # • Packet size statistics: mean, minimum, maximum, and standard deviation of packet sizes.
        # • Inter-arrival time statistics: mean, minimum, maximum, and standard deviation of time differences.
        # • Flow Bytes per second (BPS).
        # • Flow packets per second (PPS).
        time_series_df = self._load_time_series_df(index)
        pkt_times, pkt_sizes = get_times_and_sized(time_series_df, self.direction, self.trim_time)
        pkt_time_diffs = [t1 - t0 for t1, t0 in zip(pkt_times[1:], pkt_times[:-1])]

        mean_pkt_size = statistics.mean(pkt_sizes)
        max_pkt_size = max(pkt_sizes)
        min_pkt_size = min(pkt_sizes)
        std_pkt_size = statistics.stdev(pkt_sizes)

        mean_pkt_time_diffs = statistics.mean(pkt_time_diffs)
        max_pkt_time_diffs = max(pkt_time_diffs)
        min_pkt_time_diffs = min(pkt_time_diffs)
        std_pkt_time_diffs = statistics.stdev(pkt_time_diffs)

        bytes_per_second = sum(pkt_sizes) / self.trim_time
        pkts_per_second = len(pkt_sizes) / self.trim_time

        return np.array([mean_pkt_size, max_pkt_size, min_pkt_size, std_pkt_size,
                         mean_pkt_time_diffs, max_pkt_time_diffs, min_pkt_time_diffs, std_pkt_time_diffs,
                         bytes_per_second, pkts_per_second])

    def get_2d_image(self, index):
        """
        get first image of the pair in index
        """
        image = self.__getitem__(index)[0][0]
        image = image.squeeze().numpy()
        # if type(image) != Image.Image:
        #     image = Image.fromarray(np.array(image).squeeze().squeeze().astype(np.uint8), mode='L')
        return image

    def get_label(self, index):
        target = int(self.__getitem__(index)[1])
        return self.indices_to_labels_dict[target]

    def create_images(self,
                      df: pd.DataFrame,
                      direction: str,
                      trim_time: int,
                      image_size: int,
                      plot: bool = False):

        imgs_list = []
        imgs_list_size = 1
        if self.contrastive_transform:
            imgs_list_size = self.contrastive_transform.n_views
        for _ in range(imgs_list_size):
            if self.contrastive_transform:
                df_i = self.contrastive_transform.data_frame_transform(df)
            else:
                df_i = df

            pkt_times, pkt_sizes = get_times_and_sized(df_i, direction, trim_time)
            try:
                histogram = self.session_2d_histogram(pkt_times, pkt_sizes, trim_time, image_size, plot=plot)
            except IndexError as e:
                logging.error(f"{df.shape=}")
                logging.error(f"{df_i.shape=}")
                logging.error(f"{pkt_times.shape=}")
                logging.error(f"{pkt_sizes.shape=}")
                logging.error(f"{df.shape=}")
                logging.error(f"df\n{df.head(20)}")
                logging.error(f"df_i\n{df_i.head(20)}")
                logging.error(f"pkt_times={pkt_times}")
                logging.error(f"pkt_sizes={pkt_sizes}")
                raise e

            if self.convert_to_image_with_max_value_255:
                # doing this so that it is consistent with image conventions and return a PIL Image
                histogram[histogram > 255] = 255

            img_i = Image.fromarray(histogram.astype(np.uint8), mode='L')
            if self.contrastive_transform:
                # Todo, notice *255 improve results drastic.. figure out why and make it better than
                img_i = self.contrastive_transform.torch_vision_transform(img_i) * 255
            else:
                img_i = torch.from_numpy(np.array(img_i)).type('torch.DoubleTensor').unsqueeze(0).float()

            imgs_list.append(img_i)
        return imgs_list

    def session_2d_histogram(self, pkt_times: List[float], pkt_sizes: List[int], trim_time: int, image_size: int,
                             max_pkt_size=1500, plot: bool = False):

        pkt_times = np.asarray(pkt_times)
        pkt_sizes = np.asarray(pkt_sizes)

        pkt_sizes[pkt_sizes > max_pkt_size] = max_pkt_size  # Replace

        pkt_times_norm = ((pkt_times - pkt_times[0]) / trim_time) * image_size
        sizes_norm = (pkt_sizes / max_pkt_size) * image_size
        H, xedges, yedges = np.histogram2d(sizes_norm, pkt_times_norm,
                                           bins=(range(0, image_size + 1, 1), range(0, image_size + 1, 1)))

        if plot:
            plt.pcolormesh(xedges, yedges, H)
            plt.colorbar()
            plt.xlim(0, image_size)
            plt.ylim(0, image_size)
            plt.set_cmap('binary')
            plt.show()

        return H.astype(np.uint8)

    def _load_time_series_df(self, index):
        if not self.lazy_loading:
            time_series_df = self.time_series_df_list[index]
        else:
            time_series_df = pd.read_csv(self.time_series_csv_paths_list[index])
        return time_series_df

    def convert_to_numpy_arrays(self, use_augmentations_data=False):
        if use_augmentations_data:
            imgs_list_size = self.contrastive_transform.n_views
            logging.info(f"convert_to_numpy_arrays: use {imgs_list_size} augmentations per sample ")
        images_list = []
        targets_list = []
        for images, target in self:
            if use_augmentations_data:
                for i in range(imgs_list_size):
                    images_list.append(images[i].numpy())
                    targets_list.append(target.numpy())
            else:
                images_list.append(images[0].numpy())
                targets_list.append(target.numpy())
        np_X = np.concatenate(images_list).reshape(len(images_list), -1)
        np_y = np.concatenate(targets_list)
        return np_X, np_y

    def convert_to_statistical_features(self):
        stats_list = []
        targets_list = []
        for idx, (_, target) in enumerate(self):
            stats_list.append(self.get_statistical_features(idx))
            targets_list.append(target.numpy())
        np_X = np.concatenate(stats_list).reshape(len(stats_list), -1)
        np_y = np.concatenate(targets_list)
        return np_X, np_y

    def convert_to_rep_vec(self, conv_net):
        if next(conv_net.parameters()).is_cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = ""

        images_list = []
        targets_list = []
        conv_net.eval()
        with torch.no_grad():
            for images, target in self:
                if str(device) == 'cuda:0':
                    image = images[0].to(device)
                else:
                    image = images[0]

                h, z = conv_net(image.detach().unsqueeze(dim=0))
                images_list.append(h.cpu())
                targets_list.append(target.detach().numpy())
        conv_net.train()

        np_X = np.concatenate(images_list).reshape(len(images_list), -1)
        np_y = np.concatenate(targets_list)
        return np_X, np_y


class UCDavisQuicDataset(absFlowPicDataset):
    UP = "up"
    DOWN = "down"
    BOTH = "both"

    # {label: idx for idx, label in enumerate(self.labels_set)}
    labels_to_indices_dict = {
        'search': 0,
        'drive': 1,
        'youtube': 2,
        'music': 3,
        'doc': 4
    }

    #  {idx: label for idx, label in enumerate(self.labels_set)}
    indices_to_labels_dict = {
        0: 'search',
        1: 'drive',
        2: 'youtube',
        3: 'music',
        4: 'doc'
    }

    labels_ordered_list = ["search", "drive", "youtube", "music", "doc"]

    def __init__(self, dataset_path,
                 direction: str,
                 flow_max_duration: int,
                 image_size: int = 32,
                 convert_to_image_with_max_value_255=True,
                 split: str = 'train',
                 transform: Optional[ContrastiveTransformations] = None,
                 lazy_loading: bool = False,
                 small_train_size: Optional[int] = None,
                 seed=0
                 ):

        assert split in ['train', 'val', 'test', 'few-shot-train', 'all', 'small-train',
                         'small-train-random', 'test-random', 'test-human']
        if split == 'small-train' or split == 'small-train-random':
            assert small_train_size > 0
        dataset_path = Path(dataset_path)
        self.seed = seed

        index_df = pd.read_csv(dataset_path / 'index.csv')

        # Validate that with the chosen flow definition (direction+duration) we have enough packets sampled
        if flow_max_duration in [3, 15, 30, 60]:
            if flow_max_duration >= 15:
                count_interval = 15
            else:
                count_interval = 3

            if direction in ['up', 'down']:
                filter_cond = index_df[f'count_{direction}_pkt_in_first_{count_interval}_sec'] > \
                              self.MINIMUM_PKTS_FLOWPIC
            else:
                filter_cond = index_df[f'count_up_pkt_in_first_{count_interval}_sec'] + \
                              index_df[f'count_down_pkt_in_first_{count_interval}_sec'] > \
                              self.MINIMUM_PKTS_FLOWPIC
            index_df = index_df[filter_cond]

        index_train_df, index_val_df, index_few_shot_df, index_test_df = self.dynamic_split_data_set(index_df)

        labels = []
        time_series_csv_paths = []
        dataset_path = Path(dataset_path)

        if split == 'all':
            chosen_index_df = index_df
        elif split == 'train':
            chosen_index_df = index_train_df
        elif split == 'val':
            chosen_index_df = index_val_df
        elif split == 'test':
            chosen_index_df = index_test_df
        elif split == 'few-shot-train':
            chosen_index_df = index_few_shot_df
        elif split == 'small-train':
            chosen_index_df = self.create_sub_index(index_train_df, 0, small_train_size)
        elif split == 'small-train-random':
            chosen_index_df, _ = df_take_n_samples_per_class(index_train_df, 'category', n=small_train_size,
                                                             random_state=seed)
        elif split == 'test-random':
            _, not_train_df = df_take_n_samples_per_class(index_train_df, 'category', n=30,
                                                          random_state=seed)
            chosen_index_df, _ = df_take_n_samples_per_class(not_train_df, 'category', n=30,
                                                             random_state=seed)
        elif split == 'test-human':
            chosen_index_df, _ = df_take_n_samples_per_class(index_df[index_df['data_source'] == 'small_human'],
                                                             'category', n=15,
                                                             random_state=0)

        else:
            raise ValueError

        for index, row in chosen_index_df.iterrows():
            labels.append(row['category'])
            time_series_csv_paths.append(dataset_path / row['rel_path'])

        super().__init__(time_series_csv_paths=time_series_csv_paths,
                         labels=labels,
                         direction=direction,
                         flow_max_duration=flow_max_duration,
                         image_size=image_size,
                         convert_to_image_with_max_value_255=convert_to_image_with_max_value_255,
                         transform=transform,
                         lazy_loading=lazy_loading
                         )

    def create_sub_index(self, index_df, start, end, shuffle=False, seed=0):
        sub_index_df_list = []
        for label in self.labels_ordered_list:
            all_sample_with_label_df = index_df[index_df['category'] == label]
            if shuffle:
                all_sample_with_label_df = all_sample_with_label_df.sample(frac=1, random_state=seed).reset_index(
                    drop=True)

            sub_index_df_list.append(all_sample_with_label_df.iloc[start:end])
        return pd.concat(sub_index_df_list)

    def dynamic_split_data_set(self, index_df):
        # split base on ucdavis github:
        # https://github.com/shrezaei/MultitaskTrafficClassification/blob/master/dataProcessQuic.py
        np.random.seed(10)
        rand_index = np.array(range(len(index_df)))
        np.random.shuffle(rand_index)
        index_df = index_df.copy().iloc[rand_index]
        test_size = 30
        val_size = 30
        tfs_size = 30
        train_df_list = []
        test_df_list = []
        val_df_list = []
        tfs_df_list = []

        for label in self.labels_ordered_list:
            all_sample_with_label_df = index_df[index_df['category'] == label]
            test_df_list.append(all_sample_with_label_df.iloc[-test_size:])
            val_df_list.append(all_sample_with_label_df.iloc[-(test_size + val_size):-test_size])
            tfs_df_list.append(
                all_sample_with_label_df.iloc[-(test_size + val_size + tfs_size):-(test_size + val_size)])
            train_df_list.append(all_sample_with_label_df.iloc[:-(test_size + val_size + tfs_size)])

        index_train_df = pd.concat(train_df_list)
        index_test_df = pd.concat(test_df_list)
        index_val_df = pd.concat(val_df_list)
        index_few_shot_df = pd.concat(tfs_df_list)
        return index_train_df, index_val_df, index_few_shot_df, index_test_df


def create_hyper_params_dict(

        # Running environment
        ds_path: Union[str, Path] = 'data/ucdavis_quic',
        save_model_path: Optional[str] = None,
        num_workers: int = -1,
        seed: int = 10,

        # train-test-split
        split='small-train',
        small_train_size: Optional[int] = 50,
        partial_dataset_for_development_only: bool = False,

        # image creation
        flow_direction: str = 'up',
        flow_max_duration: int = 15,
        img_size: int = 1500,

        # trainer and model parameters
        max_epochs: int = 10,
        monitor_metric: str = 'val_aug_acc_in_top5',
        monitor_mode: str = 'max',
        early_stop_patience: int = 3,
        batch_size: int = 32,
        eval_on_each_train_step: bool = False,

        lightning_auto_finder: bool = False,
        lr: float = 0.001,
        temperature: float = 0.07,
        weight_decay: float = 1e-4,

        representation_size: Optional[int] = None,
        similarity_vector_size: int = int(120 / 4),
        head_type: str = 'linear',

        # Augmentations
        pkt_time_min_mult_factor: int or float = 1,
        pkt_time_max_mult_factor: int or float = 1,
        pkt_time_min_move_const: int or float = 0,
        pkt_time_max_move_const: int or float = 0,
        cutout_time_window_size: int or float = 0,

        color_jitter_apply_prob: int or float = 0,
        color_jitter_brightness: int or float = 0,
        color_jitter_contrast: int or float = 0,
        color_jitter_saturation: int or float = 0,
        color_jitter_hue: int or float = 0,

        flip_time_axis_apply_prob: int or float = 0,

        rotate_img_apply_prob: int or float = 0,
        rotate_img_min_angle: int or float = 0,
        rotate_img_max_angle: int or float = 0,

        **kwargs):
    params = locals()

    params['aug_net'] = f"mult:{pkt_time_min_mult_factor}_{pkt_time_max_mult_factor},\n" \
                        f"move:{pkt_time_min_move_const}_{pkt_time_max_move_const},\n" \
                        f"cutout:{cutout_time_window_size}"
    params['aug_cv'] = f"jitter:{color_jitter_apply_prob}_" \
                       f"{color_jitter_brightness}_{color_jitter_contrast}_" \
                       f"{color_jitter_saturation}_{color_jitter_hue},\n" \
                       f"flip:{flip_time_axis_apply_prob},\n" \
                       f"rotate:{rotate_img_apply_prob}_{rotate_img_min_angle}_{rotate_img_max_angle}"
    return params


def ucdavis_train_ds(hp, n_views=10):
    train_transform = ContrastiveTransformations(hp, n_views=n_views)
    print(train_transform)
    return UCDavisQuicDataset(
        dataset_path=hp['ds_path'],
        direction=hp['flow_direction'],
        flow_max_duration=hp['flow_max_duration'],
        image_size=hp['img_size'],
        convert_to_image_with_max_value_255=True,
        split=hp['split'],
        transform=train_transform,
        lazy_loading=False,
        small_train_size=hp['small_train_size'],
        seed=hp['seed'])


def load_ds_for_vis(hp):
    ds_for_visualization = ucdavis_train_ds(hp)
    return ds_for_visualization


def render_image(flow_pic_image, class_name, duration):
    MTU = 1500
    image_size = flow_pic_image.shape[0]
    fig = plt.figure()
    plt.imshow(flow_pic_image, cmap=plt.get_cmap('gray_r'), origin='lower')
    plt.title(class_name, fontsize=18)

    x_ticks = [0, image_size // 3, 2 * image_size // 3, image_size]
    x_vals = [0, duration // 3, 2 * duration // 3, duration]

    y_ticks = [0, image_size // 3, 2 * image_size // 3, image_size]
    y_vals = [0, MTU // 3, 2 * MTU // 3, MTU]

    plt.xticks(x_ticks, x_vals)
    plt.yticks(y_ticks, y_vals)

    st.pyplot(fig)


def extract_img_from_ds(ds, image_index):
    flow_pic_images, label = ds[image_index]
    flow_pic_image = flow_pic_images[0].squeeze().numpy()
    class_name = ds.indices_to_labels_dict[int(label)]
    return flow_pic_image, class_name


def plot_images_and_labels(images_list, labels_list, n_rows, n_cols, flow_max_duration, show=True):
    assert len(images_list) == len(labels_list)
    assert type(images_list) == list
    assert type(labels_list) == list

    total_images = n_rows * n_cols

    figsize = [6, 8]  # figure size, inches

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    fig.suptitle(f"FlowPic {flow_max_duration}sec {images_list[0].shape[0]}x{images_list[0].shape[0]}")
    plt.setp(ax, xticks=[], yticks=[])

    for idx in range(min(total_images, len(images_list))):
        image = images_list[idx]
        label = labels_list[idx]
        row_id = idx // n_cols
        col_id = idx % n_cols

        ax[row_id, col_id].imshow(image, origin='lower', cmap=plt.get_cmap('gray_r'))
        ax[row_id, col_id].set_title(f"{label}", fontsize=6)
    if show:
        plt.show()
    return fig


def create_image_and_labels_list(dataset, total_images):
    # TODO: change to random choosing NxN images
    images_list = []
    labels_list = []
    map_label_to_idx = defaultdict(list)
    for idx, (_, y) in enumerate(dataset):
        map_label_to_idx[int(y)].append(idx)

    max_range = min([len(x) for x in map_label_to_idx.values()])
    sorted_indices = []
    for i in range(int(max_range)):
        for label in map_label_to_idx.keys():
            sorted_indices.append(map_label_to_idx[label][i])

    for idx, data in enumerate(dataset):
        image = dataset.get_2d_image(sorted_indices[idx])
        label = dataset.get_label(sorted_indices[idx])
        if idx < total_images:
            images_list.append(image)
            labels_list.append(label)
        else:
            break
    return images_list, labels_list


######################################################################
if __name__ == '__main__':
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("info")

    # GLOBALS
    SPLIT = 'test'
    SMALL_TRAIN_SIZE = 100
    DS_NAME = 'ucdavis_quic'
    SHOW_MANY_IMAGES = False
    DATA_FOLDER_PATH = Path('data')
    DS_ZIP_FILENMAE = Path('ucdavis_quic.zip')
    DS_PATH = DATA_FOLDER_PATH / DS_ZIP_FILENMAE.stem
    DS_ZIP_PATH = DATA_FOLDER_PATH / DS_ZIP_FILENMAE

    # ###### Extract dataset ######
    logging.info(f"Extracting {DS_ZIP_PATH} to {DATA_FOLDER_PATH}")
    if not DS_PATH.exists():
        with zipfile.ZipFile(DS_ZIP_PATH, 'r') as unzip_obj:
            try:
                unzip_obj.extractall(DATA_FOLDER_PATH)
                logging.info(f"finish Extracting")
            except FileExistsError as e:
                logging.warning(f"{e}")
    assert DS_PATH.exists()

    # Choose iew images or image
    how_many_to_see = st.selectbox('Images View', ('single', '5_per_class'))
    if how_many_to_see == '5_per_class':
        SHOW_MANY_IMAGES = True

    # Choose Split

    # Choose images configuration
    img_size = st.select_slider("Image Size", [32, 64, 1500], 32)
    flow_max_duration = st.select_slider("Block Duration [sec]", [1, 3, 5, 15, 30, 60, 90], 15)
    flow_direction = st.selectbox('Flow Direction', ('both', 'up', 'down'))
    hp1 = create_hyper_params_dict(ds_path=DS_PATH,
                                   img_size=img_size,
                                   flow_direction=flow_direction,
                                   flow_max_duration=flow_max_duration,
                                   split=SPLIT,
                                   small_train_size=SMALL_TRAIN_SIZE,
                                   )

    ds1 = load_ds_for_vis(hp1)
    img_index = st.slider("image_index", 0, len(ds1), 0)

    # Choose augmentations configuration
    angle = st.sidebar.slider("Rotate Angle", -90, 90, 0)
    factor = st.sidebar.slider("RTT Factor", 0.0, 10.0, 1.0)
    const = st.sidebar.slider("Delay [sec]", -60.0, 0.0, 0.0)
    cutout_time_window_size = st.sidebar.slider("Packet Loss window [sec]", 0.0, 10.0, 0.0)
    flip_time_axis_apply_prob = st.sidebar.slider("Flip prob", 0, 1, 0)
    color_jitter_apply_prob = 1
    color_jitter_brightness = st.sidebar.slider("Jitter brightness", 0.0, 1.0, 0.0)
    color_jitter_contrast = st.sidebar.slider("Jitter contrast", 0.0, 1.0, 0.0)
    color_jitter_saturation = st.sidebar.slider("Jitter Saturation", 0.0, 1.0, 0.0)
    color_jitter_hue = st.sidebar.slider("Jitter Hue", 0.0, 1.0, 0.0)

    hp = create_hyper_params_dict(ds_path=DS_PATH,
                                  img_size=img_size,
                                  split=SPLIT,
                                  small_train_size=SMALL_TRAIN_SIZE,
                                  flow_max_duration=flow_max_duration,
                                  flow_direction=flow_direction,
                                  pkt_time_min_mult_factor=factor,
                                  pkt_time_max_mult_factor=factor,
                                  pkt_time_min_move_const=const,
                                  pkt_time_max_move_const=const,
                                  color_jitter_apply_prob=1,
                                  color_jitter_brightness=color_jitter_brightness,
                                  color_jitter_contrast=color_jitter_contrast,
                                  color_jitter_saturation=color_jitter_saturation,
                                  color_jitter_hue=color_jitter_hue,
                                  cutout_time_window_size=cutout_time_window_size,
                                  flip_time_axis_apply_prob=flip_time_axis_apply_prob,
                                  rotate_img_apply_prob=1,
                                  rotate_img_min_angle=angle,
                                  rotate_img_max_angle=angle,
                                  )
    logging.info(f"hp: {hp}")
    st.text(hp['aug_net'])
    st.text(hp['aug_cv'])
    col1, col2 = st.columns(2)

    with col1:
        # image1
        st.title("Original FlowPic")
        flow_pic_image, class_name = extract_img_from_ds(ds1, img_index)
        render_image(flow_pic_image, class_name, flow_max_duration)

    with col2:
        # image2
        st.title("Augmented Pic")
        ds2 = load_ds_for_vis(hp)
        flow_pic_image, class_name = extract_img_from_ds(ds2, img_index)
        render_image(flow_pic_image, class_name, flow_max_duration)

    if SHOW_MANY_IMAGES:
        col1, col2 = st.columns(2)
        n_rows, n_cols = 5, 5
        with col1:
            st.title("Original FlowPics")
            images_list, labels_list = create_image_and_labels_list(ds1, n_rows * n_cols)
            fig = plot_images_and_labels(images_list, labels_list, n_rows, n_cols, flow_max_duration, show=False)
            st.pyplot(fig)

        with col2:
            st.title("Augmented Pics")
            images_list, labels_list = create_image_and_labels_list(ds2, n_rows * n_cols)
            fig = plot_images_and_labels(images_list, labels_list, n_rows, n_cols, flow_max_duration, show=False)
            st.pyplot(fig)
