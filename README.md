# METER

This is the pytorch implementation of our manuscript "METER: Multi-task Efficient Transformer for No-Reference Image Quality Assessment".

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

To run the test_batch, please put pre-trained model in 'model' folder, then run:

        python test_batch.py

You will get a quality score and a distortion type, and a higher score indicates better image quality.

**Training & Testing on IQA Databases**

Training and testing our model on the specified dataset.

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
