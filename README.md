**Warning: unmigrated tfhub.dev model artifacts will be deleted on March 18,
2024.**

As of November 15th 2023, most [tfhub.dev](https://tfhub.dev) URLs and model
handles are now redirecting to their migrated/equivalent counterpart on Kaggle
Models.

On March 18, 2024, all unmigrated model assets previously surfaced on tfhub.dev
will be deleted – after this date, `hub.load` and `hub.KerasLayer` calls to
these tfhub.dev handles will fail permanently. See the list of unmigrated model
assets here:

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

**If you are an owner of an unmigrated model, please get in touch with us at
kaggle-models@google.com if you'd like to migrate your model. If you take no
action, your model(s) will be deleted on March 18, 2024 and not retrievable
(either by you or other users).**

For models with a Kaggle Models copy, there will be no impact on the
availability/functionality of models that were copied from tfhub.dev –
`tensorflow_hub` will continue to support downloading models that were initially
uploaded to tfhub.dev via
e.g. `hub.load("https://tfhub.dev/<publisher>/<model>/<version>")`. To see if a
tfhub.dev model has been migrated, enter the model handle in your URL bar – if
the redirect is successful, it has already been migrated, otherwise it is an
unmigrated model and will be subject to deletion.

Although no migration or code rewrites are explicitly required, we recommend
replacing tfhub.dev links with their Kaggle Models counterparts to improve code
health and debuggability.

See FAQs [here](https://kaggle.com/tfhub-dev-faqs).
