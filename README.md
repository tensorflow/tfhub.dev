# TensorFlow Hub has moved to [Kaggle Models](https://kaggle.com/models)

Starting November 15th 2023, links to [tfhub.dev](https://tfhub.dev) redirect to
their counterparts on Kaggle Models. `tensorflow_hub` will continue to support
downloading models that were initially uploaded to tfhub.dev via e.g.
`hub.load("https://tfhub.dev/<publisher>/<model>/<version>")`. Although no migration or
code rewrites are explicitly required, we recommend replacing tfhub.dev links
with their Kaggle Models counterparts to improve code health and debuggability.
See FAQs [here](https://kaggle.com/tfhub-dev-faqs).