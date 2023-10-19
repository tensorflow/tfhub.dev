# TensorFlow Hub is moving to [Kaggle Models](https://kaggle.com/models)

Starting November 15th, links to [tfhub.dev](https://tfhub.dev) will redirect to
their counterparts on Kaggle Models. `tensorflow_hub` will continue to support
downloading models that were initially uploaded to tfhub.dev via e.g.
`hub.load("https://tfhub.dev/<publisher>/<model>")`. Although no migration or
code rewrites are explicitly required, we recommend replacing tfhub.dev links
with their Kaggle Models counterparts before November 15th to improve code
health and debuggability.