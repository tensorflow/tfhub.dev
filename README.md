# TensorFlow Hub has moved to [Kaggle Models](https://kaggle.com/models)

Starting November 15th 2023, links to [tfhub.dev](https://tfhub.dev) redirect to
their counterparts on Kaggle Models. `tensorflow_hub` will continue to support
downloading models that were initially uploaded to tfhub.dev via e.g.
`hub.load("https://tfhub.dev/<publisher>/<model>/<version>")`. Although no migration or
code rewrites are explicitly required, we recommend replacing tfhub.dev links
with their Kaggle Models counterparts to improve code health and debuggability.
See FAQs [here](https://kaggle.com/tfhub-dev-faqs).

As of March 18, 2024, unmigrated model assets (see list below) were deleted and retrieval is no longer possible. These unmigrated model assets include:

-   [inaturalist/vision/embedder/inaturalist_V2](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/inaturalist/models/vision/embedder/inaturalist_V2)
-   [nvidia/unet/industrial/class_1](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_1)
-   [nvidia/unet/industrial/class_2](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_2)
-   [nvidia/unet/industrial/class_3](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_3)
-   [nvidia/unet/industrial/class_4](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_4)
-   [nvidia/unet/industrial/class_5](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_5)
-   [nvidia/unet/industrial/class_6](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_6)
-   [nvidia/unet/industrial/class_7](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_7)
-   [nvidia/unet/industrial/class_8](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_8)
-   [nvidia/unet/industrial/class_9](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_9)
-   [nvidia/unet/industrial/class_10](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/nvidia/models/unet/industrial/class_10)
-   [silero/silero-stt/de](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/silero/models/silero-stt/de)
-   [silero/silero-stt/en](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/silero/models/silero-stt/en)
-   [silero/silero-stt/es](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/silero/models/silero-stt/es)
-   [svampeatlas/vision/classifier/fungi_mobile_V1](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/svampeatlas/models/vision/classifier/fungi_mobile_V1)
-   [svampeatlas/vision/embedder/fungi_V2](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/svampeatlas/models/vision/embedder/fungi_V2)

Thank you for using tfhub.dev over the years and see you at Kaggle Models!
