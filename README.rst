=================================================================================
MICCAI 2017 Endoscopic Vision Challenge Angiodysplasia Detection and Localization
=================================================================================

Here we present our wining solution and its further development for `MICCAI 2017 Endoscopic Vision Challenge Angiodysplasia Detection and Localization`_. It addresses binary segmentation problem, where every pixel in image is labeled as an angiodysplasia lesions or background. Then, we analyze connected component of each predicted mask. Based on the analysis we developed a classifier that predict angiodysplasia lesions (binary variable) and a detector for their localization (center of a component).

.. contents::

Team members
------------
`Alexey Shvets`_, `Vladimir Iglovikov`_, `Alexander Rakhlin`_, `Alexandr A. Kalinin`_

Citation
----------

If you find this work useful for your publications, please consider citing::

    @article{shvets2018angiodysplasia,
    title={Angiodysplasia Detection and Localization Using Deep Convolutional Neural Networks},
    author={Shvets, Alexey and Iglovikov, Vladimir and Rakhlin, Alexander and Kalinin, Alexandr A.},
    journal={arXiv preprint arXiv:arXiv:1804.08024},
    year={2018}
    }

Overview
--------
Angiodysplasias are degenerative lesions of previously healthy blood vessels, in which the bowel wall have microvascular abnormalities. These lesions are the most common source of small bowel bleeding in patients older than 50 years, and cause approximately 8% of all gastrointestinal bleeding episodes. Gold-standard examination for angiodysplasia detection and localization in the small bowel is performed using Wireless Capsule Endoscopy (WCE). Last generation of this pill-like device is able to acquire more than 60 000 images with a resolution of approximately 520*520 pixels. According to the latest state-of-the art, only 69% of angiodysplasias are detected by gastroenterologist experts during the reading of WCE videos, and blood indicator software (provided by WCE provider like Given Imaging), in the presence of angiodysplasias, presents sensitivity and specificity values of only 41% and 67%, respectively.

.. figure:: https://habrastorage.org/webt/if/5p/tj/if5ptjnbzeswfgqderpcww0sstm.jpeg

Data
----
The dataset consists of 1200 color images obtained with WCE. The images are in 24-bit PNG format, with 576 |times| 576 pixel resolution. The dataset is split into two equal parts, 600 images for training and 600 for evaluation. Each subset is composed of 300 images with apparent AD and 300 without any pathology. The training subset is annotated by human expert and contains 300 binary masks in JPEG format of the same 576 |times| 576 pixel resolution. White pixels in the masks correspond to lesion localization.

.. figure:: https://hsto.org/webt/nq/3v/wf/nq3vwfqtoutrzmnbzmrnyligwym.png
    :scale: 30 %

    First row corresponds to images without pathology, the second row to images with several AD lesions in every image, and the last row contains masks that correspond to the pathology images from the second row.

|
|
|

.. figure:: https://habrastorage.org/webt/t3/p6/yy/t3p6yykecrvr9mim7fqgevodgu4.png
    :scale: 45 %

    Most images contain 1 lesion. Distribution of AD lesion areas reaches maximum of 12,000 pixels and has median 1,648 pixels.


Method
------
We evaluate 4 different deep architectures for segmentation: `U-Net`_ (Ronneberger et al., 2015; Iglovikov et al., 2017a), 2 modifications of `TernausNet`_ (Iglovikov and Shvets, 2018), and `AlbuNet34`_, a modifications of `LinkNet`_ (Chaurasia and Culurciello, 2017; Shvets et al., 2018). As an improvement over standard `U-Net`_, we use similar networks with pre-trained encoders. `TernausNet`_ (Iglovikov and Shvets, 2018) is a U-Net-like architecture that uses relatively simple pre-trained VGG11 or VGG16 (Simonyan and Zisserman, 2014) networks as an encoder. VGG11 consists of seven convolutional layers, each followed by a ReLU activation function, and ve max polling operations, each reducing feature map by 2. All convolutional layers have 3 |times| 3 kernels. TernausNet16 has a similar structure and uses VGG16 network as an encoder

.. figure:: https://hsto.org/webt/vz/ok/wt/vzokwtntgqe6lb-g2oyhzj0qcyo.png
    :scale: 72 %

.. figure:: https://hsto.org/webt/vs/by/8y/vsby8yt4bj_6n3pqdqlf2tb8r9a.png
    :scale: 72 %

Training
--------

We use Jaccard index (Intersection Over Union) as the evaluation metric. It can be interpreted as a similarity measure between a finite number of sets. For two sets A and B, it can be defined as following:

.. raw:: html

    <figure>
        <img src="https://hsto.org/webt/r2/bt/ck/r2btckzdgnau9-kwsrzvryezclk.gif" align="center"/>
    </figure>

Since an image consists of pixels, the expression can be adapted for discrete objects in the following way:

.. figure:: https://habrastorage.org/webt/_8/wc/j1/_8wcj1to6ahxfsmb8s3nrxumqjy.gif
    :align: center

where |y| and |y_hat| are a binary value (label) and a predicted probability for the pixel |i|, respectively.

Since image segmentation task can also be considered as a pixel classification problem, we additionally use common classification loss functions, denoted as H. For a binary segmentation problem H is a binary cross entropy, while for a multi-class segmentation problem H is a categorical cross entropy.

.. figure:: https://habrastorage.org/webt/tf/d0/kn/tfd0kn2l612do_wmlc6zp5rdgdw.gif
    :align: center

As an output of a model, we obtain an image, in which each pixel value corresponds to a probability of belonging to the area of interest or a class. The size of the output image matches the input image size. For binary segmentation, we use 0.3 as a threshold value (chosen using validation dataset) to binarize pixel probabilities. All pixel values below the specied threshold are set to 0, while all values above the threshold are set to 255 to produce final prediction mask.

Following the segmentation step, we perform postprocessing in order to nd the coordinates of angiodysplasia lesions in the image. In the postprocessing step we use OpenCV implementation of connected component labeling function `connectedComponentsWithStats`. This function returns the number of connected components, their sizes (areas), and centroid coordinates of the corresponding connected component. In our detector we use another threshold to neglect all clusters with the size smaller than 300 pixels. Therefore, in order to establish the presence of the lesions, the number of found components should be higher than 0, otherwise the image corresponds to a normal condition. Then, for localization of angiodysplasia lesions we return centroid coordinates of all connected components.

Results
-------

The quantitative comparison of our models' performance is presented in the Table 1. For the segmentation task the best results is achieved by `AlbuNet34`_ providing IoU = 0.754 and Dice = 0.850. When compared by the inference time, `AlbuNet34`_ is also the fastest model due to the light encoder. In the segmentation task this network takes around 20ms

.. figure:: https://hsto.org/webt/mw/yj/-l/mwyj-l6ddk6xz-ykydduixzhrdk.png
    :scale: 60 %

    Prediction of our detector on the validation image. The left picture is original image, the central is ground truth mask, and the right is predicted mask. Green dots correspond to centroid coordinates that define localization of the angiodysplasia.

|
|
|

.. table:: Table 1. Segmentation results per task. Intersection over Union, Dice coefficient and inference time, ms.

    ============= ========= ========= ==================
    Model         IOU, %    Dice, %   Inference time, ms
    ============= ========= ========= ==================
    U-Net         73.18     83.06     21
    TernausNet-11 74.94     84.43     51
    TernausNet-16 73.83     83.05     60
    AlbuNet34     75.35     84.98     30
    ============= ========= ========= ==================

Pre-trained weights for all model of all segmentation tasks can be found on `google drive`_

Dependencies
------------

* Python 3.6
* PyTorch 0.3.1
* TorchVision 0.1.9
* numpy 1.14.0
* opencv-python 3.3.0.10
* tqdm 4.19.4

These dependencies can be installed by running::

    pip install -r requirements.txt


How to run
----------
The dataset is organized in the folloing way::
::

    ├── data
    │   ├── test
    │   └── train
    │       ├── angyodysplasia
    │       │   ├── images
    │       │   └── masks
    │       └── normal
    │           ├── images
    │           └── masks
    │       .......................

The training dataset contains 2 sets of images, one with angyodysplasia and second without it. For training we used only the images with angyodysplasia, which were split in 5 folds.

1. Training

The main file that is used to train all models -  ``train.py``. Running ``python train.py --help`` will return set of all possible input parameters.
To train all models we used the folloing bash script (batch size was chosen depending on how many samples fit into the GPU RAM, limit was adjusted accordingly to keep the same number of updates for every network)::

    #!/bin/bash

    for i in 0 1 2 3
    do
       python train.py --device-ids 0,1,2,3 --limit 10000 --batch-size 12 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --jaccard-weight 0.3 --model UNet11
       python train.py --device-ids 0,1,2,3 --limit 10000 --batch-size 12 --fold $i --workers 12 --lr 0.00001 --n-epochs 15 --jaccard-weight 0.3 --model UNet11
    done

2. Mask generation.

The main file to generate masks is ``generate_masks.py``. Running ``python generate_masks.py --help`` will return set of all possible input parameters. Example::

    python generate_masks.py --output_path predictions/UNet16 --model_type UNet16 --model_path data/models/UNet16 --fold -1 --batch-size 4

3. Evaluation.

The evaluation is different for a binary and multi-class segmentation:

[a] In the case of binary segmentation it calculates jaccard (dice) per image / per video and then the predictions are avaraged.

[b] In the case of multi-class segmentation it calculates jaccard (dice) for every class independently then avaraged them for each image and then for every video::

    python evaluate.py --target_path predictions/UNet16 --train_path data/train/angyodysplasia/masks

4. Further Improvements.

Our results can be improved further by few percentages using simple rules such as additional augmentation of train images and train the model for longer time. In addition, the cyclic learning rate or cosine annealing could be also applied. To do it one can use our pre-trained weights as initialization. To improve test prediction TTA technique could be used as well as averaging prediction from all folds.

Demo Example
------------
You can start working with our models using the demonstration example: `Demo.ipynb`_

..  _`Demo.ipynb`: Demo.ipynb
.. _`Alexander Rakhlin`: https://www.linkedin.com/in/alrakhlin/
.. _`Alexey Shvets`: https://www.linkedin.com/in/shvetsiya/
.. _`Vladimir Iglovikov`: https://www.linkedin.com/in/iglovikov/
.. _`Alexandr A. Kalinin`: https://alxndrkalinin.github.io/
.. _`MICCAI 2017 Endoscopic Vision SubChallenge Angiodysplasia Detection and Localization`: https://endovissub2017-giana.grand-challenge.org/angiodysplasia-etisdb/
.. _`TernausNet`: https://arxiv.org/abs/1801.05746
.. _`U-Net`: https://arxiv.org/abs/1505.04597
.. _`AlbuNet34`: https://arxiv.org/abs/1803.01207
.. _`LinkNet`: https://arxiv.org/abs/1707.03718
.. _`google drive`: https://drive.google.com/drive/folders/1V_bLBTzsl_Z8Ln9Iq8gjcFDxodfiHxul

.. |br| raw:: html

   <br />

.. |plusmn| raw:: html

   &plusmn

.. |times| raw:: html

   &times

.. |micro| raw:: html

   &microm

.. |y| image:: https://hsto.org/webt/jm/sn/i0/jmsni0y8mao8vnaij8a4eyuoqmu.gif
.. |y_hat| image:: https://hsto.org/webt/xf/j2/a4/xfj2a4obgqhdzneysar5_us5pgk.gif
.. |i| image:: https://hsto.org/webt/87/cc/ca/87cccaz4gjp2lgyeip17utljvvi.gif
