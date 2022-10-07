# Collection sayakpaul/swin/1

Collection of Swin Transformers.


<!-- task: video-feature-extraction -->

## Overview

This collection contains different Video Swin Transformer [1] models. The original model weights are provided from [2]. There were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting steps are available in [3].


## About the models

These models can to used to extract features from videos. These models are accompanied by
Colab Notebooks for demonstration purposes and fine-tuning steps for action-recognition task and video-classification. 

The table below provides a performance summary (ImageNet-1k validation set):

| model_name                                     |   pre-train dataset |   fine-tune dataset   |   acc@1(%) |  acc@5(%) |
|:----------------------------------------------:|:-------------------:|:---------------------:|:----------:|----------:|
| swin_tiny_patch244_window877_kinetics400_1k    |    ImageNet-1K      | Kinetics 400(1k       |       78.8 |      93.6 |
| swin_small_patch244_window877_kinetics400_1k   |    ImageNet-1K      | Kinetics 400(1k)      |       80.6 |      94.5 |
| swin_base_patch244_window877_kinetics400_1k    |    ImageNet-1K      | Kinetics 400(1k)      |       80.6 |      96.6 |
| swin_base_patch244_window877_kinetics400_22k   |    ImageNet-12K     | Kinetics 400(1k)      |       82.7 |      95.5 |
| swin_base_patch244_window877_kinetics600_1k    |    ImageNet-1K      | Kinetics 600(1k)      |       84.0 |      96.5 |
| swin_base_patch244_window1677_sthv2            |    Kinetics 400     | Something-Something V2|       69.6 |      92.7 |


The `base` and `large` models were first pre-trained on the ImageNet-22k dataset and then fine-tuned
on the ImageNet-1k dataset.

[This directory](https://github.com/sayakpaul/swin-transformers-tf/tree/main/in1k-eval) provides details
on how these numbers were generated. Original scores for all the models except for the `s3` ones were
gathered from [here](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md). Scores
for the `s3` model were gathered from [here](https://github.com/microsoft/Cream/tree/main/AutoFormerV2#model-zoo).




### Feature extractors

* [swin_base_patch4_window12_384_fe](https://tfhub.dev/sayakpaul/swin_base_patch4_window12_384_fe)
* [swin_base_patch4_window7_224_fe](https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224_fe)
* [swin_large_patch4_window12_384_fe](https://tfhub.dev/sayakpaul/swin_large_patch4_window12_384_fe)
* [swin_large_patch4_window7_224_fe](https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224_fe)
* [swin_small_patch4_window7_224_fe](https://tfhub.dev/sayakpaul/swin_small_patch4_window7_224_fe)
* [swin_tiny_patch4_window7_224_fe](https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224_fe)
* [swin_base_patch4_window12_384_in22k_fe](https://tfhub.dev/sayakpaul/swin_base_patch4_window12_384_in22k_fe)
* [swin_base_patch4_window7_224_in22k_fe](https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224_in22k_fe)
* [swin_large_patch4_window12_384_in22k_fe](https://tfhub.dev/sayakpaul/swin_large_patch4_window12_384_in22k_fe)
* [swin_large_patch4_window7_224_in22k_fe](https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224_in22k_fe)
* [swin_s3_tiny_224_fe](https://tfhub.dev/sayakpaul/swin_s3_tiny_224_fe)
* [swin_s3_small_224_fe](https://tfhub.dev/sayakpaul/swin_s3_small_224_fe)
* [swin_s3_base_224_fe](https://tfhub.dev/sayakpaul/swin_s3_base_224_fe)'

## Notes

All the models output attention weights from each of the transformer blocks. The [classification Colab Notebook](https://colab.research.google.com/github/sayakpaul/swin-transformers-tf/blob/main/notebooks/classification.ipynb) shows how to fetch the attention weights for a given prediction image. 


## References

[1] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows Liu et al.](https://arxiv.org/abs/2103.14030)

[2] [Searching the Search Space of Vision Transformer by Chen et al.](https://arxiv.org/abs/2111.14725)

[3] [Swin Transformers GitHub](https://github.com/microsoft/Swin-Transformer)

[4] [AutoFormerV2 GitHub](https://github.com/silent-chen/AutoFormerV2-model-zoo)

[5] [Swin-TF GitHub](https://github.com/sayakpaul/swin-transformers-tf)



## References
[1] [Video Swin Transformer Ze et al.](https://arxiv.org/abs/2106.13230)
[2] [Video Swin Transformers GitHub](https://github.com/SwinTransformer/Video-Swin-Transformerr)
[3] [GSOC-22-Video-Swin-Transformers GitHub](https://github.com/shoaib6174/GSOC-22-Video-Swin-Transformers)

## Acknowledgements
* [Google Summer of Code 2022](https://summerofcode.withgoogle.com/)
* [Luiz GUStavo Martins](https://www.linkedin.com/in/luiz-gustavo-martins-64ab5891/)
* [Sayak Paul](https://www.linkedin.com/in/sayak-paul/)
