# Note 1.1 MNIST widget

`mnist-widget.html` contains the drawing interface, inference code, and all trained
weights. It makes no network requests. The first code cell in Note 1.1 reads this
file into a sandboxed `srcdoc` iframe; rerun that cell after changing the widget
and save its HTML output before building the site. This works from either the
notes root or the chapter directory. No model libraries are needed by students.

The model is [ONNX Model Zoo MNIST-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist),
trained on MNIST using CNTK. The upstream model is MIT licensed (see
`mnist-model-LICENSE.txt`). Original model SHA-256:
`5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd`.

The six `Parameter*` initializers were extracted using `onnx.numpy_helper.to_array`,
converted to little-endian float32, and base64 encoded without quantization. The
JavaScript reproduces the original NCHW graph: 5x5 SAME convolution (8 channels),
bias/ReLU, 2x2 pooling; 5x5 SAME convolution (16 channels), bias/ReLU, 3x3 pooling;
flatten 16x4x4, dense layer to 10 logits, and stable softmax.

The drawing displays black ink on white; preprocessing inverts it to the model's
white-on-black input, scaled to [0, 1]. The drawing canvas is actually
28x28 pixels, enlarged with nearest-neighbor display. Ink is cropped, shrunk if
needed to fit within 20x20 pixels, and centered by its center of mass. The empty
canvas deliberately has no prediction. Displayed percentages use largest-remainder
rounding to sum to 100.0%.

Run `node scripts/test_mnist_widget.cjs`. Fixtures contain the first official MNIST
test example of each digit and reference probabilities computed by ONNX Runtime
1.30.0 from the original model. The browser implementation also classified 984 of
the first 1,000 official test images correctly before drawing preprocessing.
