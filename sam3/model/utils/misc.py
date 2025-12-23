# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os, platform
from collections import OrderedDict, defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Protocol, runtime_checkable
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F


def _is_named_tuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


@runtime_checkable
class _CopyableData(Protocol):
    def to(self, device: torch.device, *args: Any, **kwargs: Any):
        """Copy data to the specified device"""
        ...


def copy_data_to_device(data, device: torch.device, *args: Any, **kwargs: Any):
    """Function that recursively copies data to a torch.device.

    Args:
        data: The data to copy to device
        device: The device to which the data should be copied
        args: positional arguments that will be passed to the `to` call
        kwargs: keyword arguments that will be passed to the `to` call

    Returns:
        The data on the correct device
    """

    if _is_named_tuple(data):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key, default=None):
        if key not in self.cache:
            return default
        # Move the key to the end to show that it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, key):
        return self.get(key)

    def put(self, key, value):
        # Insert the item or update the existing one
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # If the cache exceeds the capacity, pop the first (least recently used) item
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __contains__(self, key):
        return key in self.cache

    def pop(self, key, default=None):
        return self.cache.pop(key, default)

    def clear(self):
        self.cache.clear()

    def items(self):
        """Return a dynamic view of (key, value) pairs, like dict.items()."""
        return self.cache.items()

class TorchCodecDecoder:
    """
    A wrapper to support GPU device and num_threads in TorchCodec decoder,
    which are not supported by `torchcodec.decoders.SimpleVideoDecoder` yet.
    """

    def __init__(self, source, dimension_order="NCHW", device="cpu", num_threads=1):
        # ensure ffmpeg binaries needed by torchcodec are on path
        import ffmpeg as _ffmpeg
        _ffmpeg.add_to_path()

        from torchcodec import _core as core

        self._source = source  # hold a reference to the source to prevent it from GC
        if isinstance(source, str):
            self._decoder = core.create_from_file(source, "exact")
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source, "exact")
        else:
            raise TypeError(f"Unknown source type: {type(source)}.")
        assert dimension_order in ("NCHW", "NHWC")

        device_string = str(device)
        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(
            self._decoder,
            dimension_order=dimension_order,
            device="cpu",
            num_threads=(1 if "cuda" in device_string else num_threads),
        )
        video_metadata = core.get_container_metadata(self._decoder)
        best_stream_index = video_metadata.best_video_stream_index
        assert best_stream_index is not None
        self.metadata = video_metadata.streams[best_stream_index]
        assert self.metadata.num_frames_from_content is not None
        self._num_frames = self.metadata.num_frames_from_content

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key: int):
        from torchcodec import _core as core

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )
        frame_data, *_ = core.get_frame_at_index(
            self._decoder,
            frame_index=key,
        )
        return frame_data


class VideoFileLoaderWithTorchCodec:
    def __init__(
        self,
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        gpu_acceleration=True,
        gpu_device=None,
        cache_size=100,
        separate_prompts=None,
    ):
        # Check and possibly infer the output device (and also get its GPU id when applicable)
        assert gpu_device is None or gpu_device.type == "cuda"
        gpu_id = (
            gpu_device.index
            if gpu_device is not None and gpu_device.index is not None
            else torch.cuda.current_device()
        )
        if offload_video_to_cpu:
            out_device = torch.device("cpu")
        else:
            out_device = torch.device("cuda") if gpu_device is None else gpu_device
        self.out_device = out_device
        self.gpu_acceleration = gpu_acceleration
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        if not isinstance(img_mean, torch.Tensor):
            img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self.img_mean = img_mean
        if not isinstance(img_std, torch.Tensor):
            img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self.img_std = img_std

        if gpu_acceleration:
            self.img_mean = self.img_mean.to(f"cuda:{self.gpu_id}")
            self.img_std = self.img_std.to(f"cuda:{self.gpu_id}")
            decoder_option = {"device": f"cuda:{self.gpu_id}"} if platform.system() == "Linux" else {} # not on Linux? Upload later as no cuda-enabled versions of torchcodec are available
        else:
            self.img_mean = self.img_mean.cpu()
            self.img_std = self.img_std.cpu()
            decoder_option = {"num_threads": 1}  # use a single thread to save memory

        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.video_stream = TorchCodecDecoder(video_path, **decoder_option)

        # `num_frames_from_content` is the true number of frames in the video content
        # from the scan operation (rather than from the metadata, which could be wrong)
        self.num_frames = self.video_stream.metadata.num_frames_from_content
        self.video_height = self.video_stream.metadata.height
        self.video_width = self.video_stream.metadata.width

        # if we have extra frames for prompts, set up loading for those too
        self.extra_frames = None
        if separate_prompts is not None:
            self.extra_frames = [fr for fr in separate_prompts]
            self.num_extra_frames = len(self.extra_frames)
            self.num_frames += self.num_extra_frames
        img_paths = [video_path]
        if self.extra_frames is not None:
            self.img_paths = self.extra_frames+img_paths
        else:
            self.img_paths = img_paths

        # Create an LRU cache for frames
        self.cache_size = cache_size
        self.frame_cache = LRUCache(capacity=self.cache_size)
        # warm up: get first frame
        self.get_frame(0)

    @torch.inference_mode()
    def get_frame(self, idx):
        """Fetch a frame using the LRU cache or load it if it's not cached."""
        # Check if frame is in cache
        cached_frame = self.frame_cache.get(idx)
        if cached_frame is not None:
            return cached_frame

        # Load the frame if it's not in cache
        frame = self._load_frame(idx)

        # Add the frame to the cache
        self.frame_cache.put(idx, frame)
        return frame

    def __getitem__(self, idx):
        return self.get_frame(idx)

    def _load_frame(self, idx):
        if self.extra_frames is not None:
            if idx<self.num_extra_frames:
                img = Image.open(self.extra_frames[idx]).convert("RGB")
                self.video_width  = img.width
                self.video_height = img.height
                return self._transform_frame(img)
            else:
                idx = idx-self.num_extra_frames
        frame = self.video_stream[idx].to(self.out_device)  # ensure on correct device
        return self._transform_frame(frame)

    def _transform_frame(self, frame):
        if not isinstance(frame, torch.Tensor):
            frame = torch.tensor(np.array(frame), dtype=torch.float32).permute(2, 0, 1).to(self.out_device)
        else:
            frame = frame.float()  # convert to float32 before interpolation
        frame_resized = F.interpolate(
            frame[None, :],
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )[0]
        # float16 precision should be sufficient for image tensor storage
        frame_resized = frame_resized.half()  # uint8 -> float16
        frame_resized /= 255
        frame_resized -= self.img_mean
        frame_resized /= self.img_std
        if self.offload_video_to_cpu:
            frame_resized = frame_resized.cpu()
        elif frame_resized.device != self.out_device:
            frame_resized = frame_resized.to(device=self.out_device, non_blocking=True)
        return frame_resized

    def __len__(self):
        return self.num_frames
