# Bonus Work 1
This submission consists of changes made to the repository with respect to performing a deploy with Tensorflow Serving

Once you clone the repo, run the training code which is the file "myTFDistributedTrainer.py"
I have attached the snippets of the input/command used to run this file and the expected output. The snippets are "myDistributedTrainer_input.PNG" and "myDistributedTrainer_output.PNG"

Once you have trained the model, do the inference and this can be done by using "myTFInference.py"
The snippets associated with this is "myTFInference_execution.PNG" showing the complete output.

So, here, the dataset I am working on is fashionMNIST. For the inference, I have passed an image which is a tshirt and the URL I have passed is https://cdn.shopify.com/s/files/1/1748/4357/products/merry-new-year-unisex-t-shirt.jpg?v=1667874550 . It depicts the provided image as a shirt with a 91.72 accuracy.

Now, the next step is exporting this entire flow into a lite model.
Here, the output shows that the image provided is a t-shirt.
A snippet representing this output is also attached and the files are "exportfile_input.PNG" and "exportfile_output.PNG"

A plot between Training and Validation interms of Accuracy and Loss are plotted and an image is attached to take a look at it. The file is "accuracy_vs_loss.PNG"

# TensorFlow Serving
Reference used: https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

Here, the model that we have saved above, we create requests now using tensorflow-model-server which is essentially utilized for deploying the trained models.
I am attaching a Google Colab notebook which has all the steps followed and also the expected output snippets are attached.
The model is saved in the outputs folder. I have uploaded this folder to my google drive.
My fetching the folder from the drive, I have run the necessary commands on the google colab notebook.
The end goal is it displays what is the expected and the obtained output.

# MultiModalClassifier
This is a project repo for multi-modal deep learning classifier with popular models from Tensorflow and Pytorch. The goal of these baseline models is to provide a template to build on and can be a starting point for any new ideas, applications. If you want to learn basics of ML and DL, please refer this repo: https://github.com/lkk688/DeepDataMiningLearning.

# Package setup
Install this project in development mode
```bash
(venv38) MyRepo/MultiModalClassifier$ python setup.py develop
```
After the installation, the package "MultimodalClassifier==0.0.1" is installed in your virtual environment. You can check the import
```bash
>>> import TFClassifier
>>> import TFClassifier.Datasetutil
>>> import TFClassifier.Datasetutil.Visutil
```

If you went to uninstall the package, perform the following step
```bash
(venv38) lkk@cmpeengr276-All-Series:~/Developer/MyRepo/MultiModalClassifier$ python setup.py develop --uninstall
```

# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# Tensorflow Lite
* Tensorflow lite guide [link](https://www.tensorflow.org/lite/guide)
* [exportTFlite](\TFClassifier\exportTFlite.py) file exports model to TFlite format.
  * testtfliteexport function exports the float format TFlite model
  * tflitequanexport function exports the TFlite model with post-training quantization, the model size can be reduced by
![image](https://user-images.githubusercontent.com/6676586/126202680-e2e53942-7951-418c-a461-99fd88d2c33e.png)
  * The converted quantized model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
* To ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), we can enforce full integer quantization for all ops including the input and output, add the following code into function tflitequanintexport
```bash
converter_int8.inference_input_type = tf.int8  # or tf.uint8
converter_int8.inference_output_type = tf.int8  # or tf.uint8
```
  * The check of the floating model during inference will show false
```bash
floating_model = input_details[0]['dtype'] == np.float32
```
  * When preparing the image data for the int8 model, we need to conver the uint8 (0-255) image data to int8 (-128-127) via loadimageint function
  
# TensorRT inference
Check this [Colab](https://colab.research.google.com/drive/1aCbuLCWEuEpTVFDxA20xKPFW75FiZgK-?usp=sharing) (require SJSU google account) link to learn TensorRT inference for Tensorflow models.
Check these links for TensorRT inference for Pytorch models: 
* https://github.com/NVIDIA-AI-IOT/torch2trt
* https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
* https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/


[def]: C:\fall2022\255\bonus_final\myDistributedTrainer_input.PNG