import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from PIL import Image

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))

            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class VideoInpaintingDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        video_dir: str = "RGB_480",
        mask_dir: str = "MASK_480",
        gt_dir: str = "GT_480",
        num_frames: int = 100,  # Original sequence length
        frame_stride: int = 2,
        image_size: int = 480,
        center_crop: bool = True,
        normalize: bool = True,
    ):
        """Dataset for video inpainting training.
        
        Args:
            data_root: Root directory containing sequence folders
            video_dir: Name of subdirectory containing input frames (RGB)
            mask_dir: Name of subdirectory containing mask frames
            gt_dir: Name of subdirectory containing ground truth frames
            num_frames: Number of frames per sequence
            frame_stride: Stride between sampled frames
            image_size: Target video height/width
            center_crop: Whether to center crop images
            normalize: Whether to normalize images to [-1, 1]
        """
        self.data_root = Path(data_root)
        self.video_dir = video_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = (image_size, image_size)
        self.center_crop = center_crop
        self.normalize = normalize
        
        # Calculate model's native dimensions
        self.model_height = 60  # Native model height
        self.model_width = 90   # Native model width
        self.model_frames = 49  # Native model frames
        
        # Calculate scaling factors
        self.spatial_scale = (image_size / self.model_height, image_size / self.model_width)
        self.temporal_scale = num_frames / self.model_frames
        
        # Get video sequences (folders containing frame sequences)
        self.video_sequences = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        # Calculate effective sequence length
        self.effective_length = self.num_frames * self.frame_stride
        
        # Create transforms for RGB and GT frames
        transforms_list = [
            transforms.Resize(self.image_size, antialias=True),
        ]
        if center_crop:
            transforms_list.append(transforms.CenterCrop(self.image_size))
        if normalize:
            transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = transforms.Compose(transforms_list)
        
        # Create mask transforms (no normalization)
        mask_transforms = [
            transforms.Resize(self.image_size, antialias=True),
        ]
        if center_crop:
            mask_transforms.append(transforms.CenterCrop(self.image_size))
        self.mask_transform = transforms.Compose(mask_transforms)
        
        # Log dataset information
        logger.info(f"Dataset initialized with:")
        logger.info(f"  Data root: {data_root}")
        logger.info(f"  Number of sequences: {len(self.video_sequences)}")
        logger.info(f"  Frames per sequence: {num_frames}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Spatial scaling: {self.spatial_scale}")
        logger.info(f"  Temporal scaling: {self.temporal_scale}")
        
        # Validate sequences
        self._validate_sequences()
    
    def _validate_sequences(self):
        """Validate sequences and filter out invalid ones."""
        valid_sequences = []
        for seq_dir in self.video_sequences:
            # Check required directories exist
            rgb_dir = seq_dir / self.video_dir
            mask_dir = seq_dir / self.mask_dir
            gt_dir = seq_dir / self.gt_dir
            
            if not all(d.exists() and d.is_dir() for d in [rgb_dir, mask_dir, gt_dir]):
                logger.warning(f"Skipping sequence {seq_dir}: missing required directories")
                continue
            
            # Check minimum number of frames in each directory
            min_frames = 100
            rgb_frames = len(list(rgb_dir.glob("frame_*.png")))
            mask_frames = len(list(mask_dir.glob("frame_*.png")))
            gt_frames = len(list(gt_dir.glob("frame_*.png")))
            
            if any(count < min_frames for count in [rgb_frames, mask_frames, gt_frames]):
                logger.warning(
                    f"Skipping sequence {seq_dir}: insufficient frames "
                    f"(RGB: {rgb_frames}, Mask: {mask_frames}, GT: {gt_frames}, "
                    f"minimum required: {min_frames})"
                )
                continue
            
            # Check all required frames exist
            frames_missing = False
            for frame_idx in range(1, self.num_frames + 1):
                frame_name = f"frame_{frame_idx:05d}.png"
                if not all((d / frame_name).exists() for d in [rgb_dir, mask_dir, gt_dir]):
                    logger.warning(f"Skipping sequence {seq_dir}: missing frame {frame_name}")
                    frames_missing = True
                    break
            
            if frames_missing:
                continue
                
            valid_sequences.append(seq_dir)
        
        # Update video_sequences with only valid ones
        self.video_sequences = valid_sequences
        logger.info(f"Found {len(self.video_sequences)} valid sequences out of {len(valid_sequences)} total")
    
    def __len__(self):
        """Return number of sequences in dataset."""
        return len(self.video_sequences)
    
    def _load_frame(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single RGB frame."""
        img = Image.open(path).convert("RGB")
        img = TT.ToTensor()(img)
        img = self.transform(img)
        return img
    
    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single mask frame."""
        # Load mask as grayscale
        mask = Image.open(path).convert("L")
        
        # Convert to tensor
        mask = TT.ToTensor()(mask)
        
        # Ensure values are in [0, 1] range
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # Strict binary thresholding
        mask = (mask > 0.5).float()
        
        # Apply any additional transforms
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            # Re-threshold after transforms to ensure binary
            mask = (mask > 0.5).float()
        
        # Ensure single channel
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask[0:1]
        
        # Final validation
        if not torch.all(torch.logical_or(mask == 0, mask == 1)):
            raise ValueError(f"Mask from {path} contains non-binary values after processing")
        
        return mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a video sequence and corresponding masks.
        
        Returns:
            Dict containing:
                - rgb: Input frames [C, T, H, W]
                - mask: Binary masks [1, T, H, W]
                - gt: Ground truth frames [C, T, H, W]
        """
        # Get sequence folder
        sequence_folder = self.video_sequences[idx]
        
        # Get paths for RGB, mask, and GT
        rgb_dir = sequence_folder / self.video_dir
        mask_dir = sequence_folder / self.mask_dir
        gt_dir = sequence_folder / self.gt_dir
        
        # Load frames
        rgb_frames = []
        mask_frames = []
        gt_frames = []
        
        for frame_idx in range(1, self.num_frames + 1):
            frame_name = f"frame_{frame_idx:05d}.png"
            
            # Load RGB frame
            rgb_frame = self._load_frame(rgb_dir / frame_name)
            rgb_frames.append(rgb_frame)
            
            # Load mask frame
            mask_frame = self._load_mask(mask_dir / frame_name)
            mask_frames.append(mask_frame)
            
            # Load ground truth frame
            gt_frame = self._load_frame(gt_dir / frame_name)
            gt_frames.append(gt_frame)
        
        # Stack frames along temporal dimension
        rgb_tensor = torch.stack(rgb_frames, dim=1)    # [C, T, H, W]
        mask_tensor = torch.stack(mask_frames, dim=1)  # [1, T, H, W]
        gt_tensor = torch.stack(gt_frames, dim=1)      # [C, T, H, W]
        
        return {
            "rgb": rgb_tensor,
            "mask": mask_tensor,
            "gt": gt_tensor
        }


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []
