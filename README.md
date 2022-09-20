# METER

This is the pytorch implementation of our manuscript "METER: Multi-task Efficient Transformer for No-Reference Image Quality Assessment".

**Abstract**

Image quality assessment (IQA) is a fundamental yet challenging task in computer vision. IQA methods based on convolutional neural networks typically employ deeply-stacked convolutions to learn local image features pertinent to image quality. However, non-local information is usually neglected, making it difficult to assess image quality in a holistic manner. As an effective alternative to capture long-range dependencies, transformers can extract both local and non-local information based on a self-attention mechanism, but are computationally expensive. Moreover, existing methods are typically not tailored for dealing with multiple types of distortions that may occur for example in image acquisition, compression, and transmission. As a remedy, we introduce in this paper an end-to-end multi-task efficient transformer (METER) for NR-IQA, consisting of a muti-scale semantic feature extraction (MSFE) backbone module, a distortion type identification (DTI) module, and an adaptive quality prediction (AQP) module. METER identifies the distortion type using the DTI module to facilitate extraction of distortion-specific features via the MSFE module. METER scores image quality in an adaptive manner by adjusting the weights and biases of adaptive fully-connected (AFC) layers in the AQP module, increasing generalizability to images captured in different natural environments. Experimental results on five public datasets show that METER significantly outperforms existing methods and achieves near-perfect accuracy.

**Dependencies**

--python  3.6

--pytorch 1.7.0

--torchvision 0.8.0

--scipy 1.5.4

--pliiow 8.4.0

--timm 0.4.9

--openpyxl 3.0.7

**Usages**

**Testing batch images**

To run the test_batch, please put trained model in 'model' folder, then run:

        python test_batch.py

You will get a quality score and a distortion type, and a higher score indicates better image quality.

**Training & Testing on IQA Databases**

First put the pretrained weight (https://drive.google.com/file/d/1W0mvaqjFVlEXSZynEJhRRg7LlsCr2AV9/view?usp=sharing) in 'pretrained' folder. Training and testing our model on the specified dataset.

        python train_test_IQA.py

**Some available options:**

--dataset: Training and testing dataset, support datasets: livec | koniq-10k | bid | live | csiq.

--train_patch_num: Sampled image patch number per training image.

--test_patch_num: Sampled image patch number per testing image.

--batch_size: Batch size.

**Acknowledgement**

The code used in this research is inspired by HyperIQA (https://github.com/SSL92/hyperIQA) and ResT (https://github.com/wofmanaf/ResT).

**Contact**

If you have any questions, please contact us (dlmu.p.l.zhu@gmail.com)
