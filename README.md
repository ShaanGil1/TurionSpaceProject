# Installation
- Install conda from: https://www.anaconda.com/ <br>
- Install cuda from: https://developer.nvidia.com/cuda-11-7-0-download-archive (Code will still run w/o gpu but will be significantly slower) <br>Note: You need to have cuda 11.7 running and not the most recent version since pytorch does not support that yet.<br>
- For Nvidia GPU run the following line on command prompt to check cuda version <br>
```
nvcc -V
```
- Extract all files into a folder <br>
- In the root of the folder run the following line using conda command line<br>
```
conda env create -f environment.yaml
```
- Activate the environment using the following command: <br>
```
conda activate pytorchcnn
```
- Run the model.ipynb <br>
- Run all cells and model should run <br>
- Modify any hyperparams in the hyperparams cell<br>
- Note: Dataset only needs to be downloaded once no need to re-run that cell after the first time, (might need to create empty folder called "dataset") <br>
## Computer Specs
- Whole file can run under 4 minutes (not including dataset download) on my device using a NVIDIA GeForce RTX 3070, 8GB dedicated RAM
# Considerations and Justifactions for Choices made
## Dataset
- The chosen Dataset was CalTech101, the main reason it was chosen is that the pictures have varying background clutter and lighting (as described by the motivation)
- Additionally, another big reason is because of time and space for the project, Caltech101 is a fairly small dataset with only 9000 images in comparison to the hundreds of thousands in LSUN and COCO
- The dataset has 101 categories with both black/white images and colored images (This was dealt with by converting B&W to 3 channel tensors)
- Images were converted to be the size of 256x256 (seems like a reasonable option given the image sizes varied)
- A Gaussian Blur was also applied to all of the images as described by Y. Gal could be a good idea to increase performance
## Model
- The model is used as a simple CNN that uses convolutions, batch norms, max pooling, and dropouts layers.  The activation function used was ReLU for quick computations.
- The model follows the standard CNN convention of using convolution layers, followed by max pooling and some activation function (ReLU in this case).  And finally condensing into fully connected layers that have activation/dropouts between them.
- Dropout layers were applied as the papers mentioned was a good way to address uncertainty
## Hyper-Parameters
- Cross-Entropy Loss was selected for this model as it had the best results and is standard a good choice for image classification
- Optimizer chosen was Adam, it quickly convergence and yielded the best results, however, did seem to overfit the model
- Batch_size chosen was 64, search space of 16, 32, 64, 128 was checked and 64 has the best results, not by much however
- LR selected through many experimentations, search space from 1e-5 to 1e-2 was checked in increments of 1e-5, 5e-4, 1e-4 and so on. 1e-4 was best
- Weight Decay was added later to apply L2 regularization to remove some overfitting. Weight decay was checked from 1-e4 to 1-e3.
- Epochs selected are likely too many models converge much sooner than 30 epochs but due to using an LR scheduler I increased it to see if small improved could be made.
- LR scheduler is to help the model converge quicker the values chooses were selected by looking at the accuracy/loss curve (see where the model plateaued) and the decay value was selected and tested from search space of .1, .2, .3, .4, .5.
## Metrics used/Results
- Accuracy/Loss was used as the metric for evaluation.  The final test accuracy was: 74.48% and the final train accuracy was: 95.94%
- This suggests the model was overfitted by a lot due to the gap loss/accuracy from the train and test set
- The loss was also significantly lower in the train set
- Applying Gaussian Blur slightly increased performance (1-2 points)
- Adding regularization (weight decay) also slightly increased performance (1-2 points)
# Ways to improve
- There are many ways to improve this model, one simple way would be to use a heavier dataset such as COCO/LSUN
- Another big way would be to convert this into object detection instead of simple classification and make a bounding box around the object so it can track this object (as per the note). Making an object detection model solves the problem in a much better way by actually tracking the object.
- Use more data augmentation so that the model can learn sparse backgrounds/variable lighting conditions. Augmentations that lower the intensity of colors would be a good example to look into
- Another simple way to improve this would be to use transfer learning or make a deeper CNN, the ones used in this project were light and simple and made to run in a short period of time
- Use more ways to combat overfitting such as more dropout layers, more regularization, and actually use a validation set.
- Use Monte Carlo dropout (as per paper), using different dropouts at test time and comparing them at the end. Or also using a BNN.  Using CE Loss also does this in a way by generating probabilities and soft-maxing them.
- Use ensemble learning, by chaining multiple models to increase performance.
- Given more time and computing power there are a lot of ways to improve this design.
