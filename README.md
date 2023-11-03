# DataComp pipeline

## Introduction

[DataComp](https://www.datacomp.ai/) is a competition organized by the University of Washington and
others to come up with the best possible image-text dataset to train a fixed CLIP model. Hence, it's
an ideal use case for Fondant, as we can leverage reusable components to filter large, noisy
image-text datasets.

In this example, we build a pipeline for filtering the dataset using the T-Mars data filtering
approach. For more information on T-Mars, check out
the [official paper](https://arxiv.org/pdf/2307.03132.pdf).

## Pipeline Overview

The image below shows the entire pipeline and its workflow. Note that this workflow is currently
adapted to the interior design domain, but can be easily adapted to other domains by changing the
prompt generation component.

![Image](docs/art/pipelines/interior_design/controlnet-interior-design-pipeline.png)

There are 5 components in total, these are:

1. [**Load from hf hub**](components/generate_prompts): The pipeline begins by loading the initial
   datacomp data which we hosted on the Hugginface hub.

2. [**Download images**](https://github.com/ml6team/fondant/tree/main/components/download_images):
   This component downloads the actual images based on the URLs retrieved by the previous component.
   It takes in the URLs as input and returns the actual images.

3. [**Resize images**](https://github.com/ml6team/fondant/tree/main/components/resize_images): This
   component resizes the images to a fixed size. It takes in the images as input and returns the
   resized images.

4. [**Detect text**](components/detect_text): This component detects text in the images using
   ann [mmocr model](https://github.com/locuslab/T-MARS/tree/main/dataset2metadata/text_detection).
   It takes in the images as input and returns the bounding boxes of the detected text.

5. [**Mask images**](components/mask_images): This component masks the detected text in the images.
   It takes in the images and the bounding boxes as input and returns the masked images.

6. [**Add clip score**](components/add_clip_score): This component adds a CLIP score to the images.
   The clip score is estimated as the dot product between the CLIP embeddings of the masked images
   and the original image captions.

7. [**Filter clip score**](components/filter_clip_score): This component filters the images based on
   their CLIP score. It takes in the images and the CLIP scores as input and returns the filtered
   indexes.

## Install and Run

### Usage

**Prerequisite:**

- Ensure Python version 3.8 to 3.10 is installed on your system.
- Install and configure Docker on your system.
- Ensure that you have a GPU for running the GPU-based component of the pipeline.

Follow these steps to get started and running the Fondant pipeline on your local machine.

1. **Setup your environment:** Clone this repository to your local machine using the following
   command:

```shell
git clone https://github.com/ml6team/fondant-usecase-controlnet.git
```

or use SSH instead:

```shell
git clone git@github.com:ml6team/fondant-usecase-controlnet.git
```

Afterwards, you can install all needed requirements:

```shell
pip install -r requirements.txt
```

You can confirm that Fondant has been installed correctly on your system by executing the following
command:

```shell
fondant --help
```

2**Run the pipeline:** Please navigate to the root directory of this repository and perform the
following:

```shell
fondant run local pipeline.py
```

The pipeline will be compiled into a `docker-compose.yaml` file and subsequently executed.

Fondant provides various runners to execute the pipeline in different environments. If you intend to
run the pipeline in a production environment, you can utilize, for example,
the [Vertex AI runner](https://fondant.ai/en/latest/pipeline/#vertex-runner).