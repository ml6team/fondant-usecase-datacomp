"""Pipeline used to filter the dataset of the Datacomp competition.

This pipeline implements the T-MARS paper: https://arxiv.org/abs/2307.03132.
"""
import logging
from pathlib import Path
import pyarrow as pa
from fondant.pipeline import Pipeline, Resources

logger = logging.getLogger(__name__)

BASE_PATH = "./fondant-artifacts"
N_ROWS_TO_LOAD = 10  # Set to None to load all rows
IMAGE_SIZE = 256

# Create data directory if it doesn't exist
Path(BASE_PATH).mkdir(parents=True, exist_ok=True)

# Initialize pipeline and client
pipeline = Pipeline(
    name="datacomp-filtering-pipeline",
    description="A pipeline for filtering the Datacomp dataset",
    base_path=BASE_PATH,
)

dataset_from_hf_hub = pipeline.read(
    "load_from_hf_hub",
    arguments={
        "dataset_name": "nielsr/datacomp-small-with-text-embeddings",
        "n_rows_to_load": N_ROWS_TO_LOAD,
    },
    produces={
        "url": pa.string(),
        "original_width": pa.int64(),
        "original_height": pa.int64(),
        "face_bboxes": pa.list_(pa.list_(pa.float64())),
        "sha256": pa.string(),
        "text": pa.string(),
        "uid": pa.string(),
        "clip_b32_similarity_score": pa.float32(),
        "clip_l14_similarity_score": pa.float32(),
        "clip_l14_text_embedding": pa.list_(pa.float64())
    }
)

images = dataset_from_hf_hub.apply(
    "download_images",
    consumes={
        "image_url": "url"
    },
    arguments={
        "retries": 2,
        "min_image_size": 0,
    },
)

resized_images = images.apply(
     "components/resize_images",
     arguments={
         "resize_width": IMAGE_SIZE,
         "resize_height": IMAGE_SIZE,
     }
 )

detected_text = resized_images.apply(
    "components/detect_text",
    arguments={
        "batch_size": 8,
        "image_size": IMAGE_SIZE,
    },
    resources=Resources(accelerator_name="GPU", accelerator_number=1)
)

mask_images = detected_text.apply(
    "components/mask_images",
)

embedded_images = mask_images.apply(
    "embed_images",
    arguments={
        "batch_size": 8,
    },
    resources=Resources(accelerator_name="GPU", accelerator_number=1)
)

images_with_clip_score = embedded_images.apply(
    "components/add_clip_score",
    consumes={
        "text_embedding": "clip_l14_text_embedding"
    }
)

filtered_clip_score_op = images_with_clip_score.apply(
    "components/filter_clip_score",
    arguments={
        "threshold_score": 0.19
    }
)
