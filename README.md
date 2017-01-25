## Project: Behavioral Cloning

### Overview

The goal of this project is to teach a Convolutional Neural Network (CNN) to drive a car in a simulator. The simulator is provided by Udacity, it's based on Unity Engine, and uses real game physics to create a close approximation ro real driving.

![The Self-Driving Car Simulator](https://raw.githubusercontent.com/mpuig/CarND-Behavioral-Cloning/master/_static/simulator.png)


When entering the simulator two options are given: Training Mode and Autonomous Mode.

![Udacity Simulator](https://raw.githubusercontent.com/mpuig/CarND-Behavioral-Cloning/master/_static/entering.png)

In order to start collecting training data, we need to do the following:

1. Enter Training Mode in the simulator.
2. Start driving the car to get a feel for the controls.
3. When we're ready, hit the record button in the top right to start recording.
4. Continue driving for a few laps or till we feel like we have enough data.
5. Hit the record button in the top right again to stop recording.

If everything went correctly, we should see the following in the directory we selected:

1. `IMG` folder - it contains all the frames of our driving.
2. `driving_log.csv`- each row in this sheet correlates the images with the steering angle, throttle brake and speed of our car. We'll mainly be using the steering angle.

![Contents of driving_log.csv](https://raw.githubusercontent.com/mpuig/CarND-Behavioral-Cloning/master/_static/dataset.png)

Now that we have training data, it's time to build and train a neural network!

We'll use [Keras](http://keras.io) to train a network to do the following:

1. Take in an image from the center camera of the car as input to our neural network.
2. Output a new steering angle for the car.
3. Save our model architectura as `model.json`and weights as `model.h5`.


#### Validating our network

We can validate our model by launching the simulator and entering autonomous mode.

The car just sit there until our Python server connects to it and provides it steering angles. To start the server, just run `python drive.py model.json`.

Once the model is up and running in `drive.py`, we should see the car move around the track!


### Multiple camera views and data augmentation

If we drive and record normal laps around the track, it might not be enough to train the model to drive properly.

Here’s the problem: if the training data is all focused on driving down the middle of the road, the model won’t ever learn what to do if it gets off to the side of the road. And probably when we run the model to predict steering angles, things won’t go perfectly and the car will wander off to the side of the road at some point.

To solve the recovery issue, the simulator provides images of three cameras into the car: center, left and right.

What we'll do is to map recovery paths from each camera. For example, if we train the model to associate a given image from the center camera with a left turn, then we could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn. And we could train the model to associate the corresponding image from the right camera with an even harder left turn.

In that way, we can simulate our vehicle being in different positions, somewhat further off the center line. To read more about this approach, see this [paper by NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) or this post by [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.rk62yvsgs)

We also use brightness augmentation, changing the brightness of images to simulate day and night conditions. To do it, we generate images with different brightness by first converting images to HSV, scaling up or down the V channel and converting back to the RGB channel.

Another transformation we use is to flip images randomly, and change the sign of the predicted steering angle, to simulate driving in the opposite direction.


***

### The project

To do the project, I created a python package [model](https://github.com/mpuig/CarND-Behavioral-Cloning/tree/master/model) which wraps all the model functionallities, and allows to use different network architectures easily. A jupyther notebook [Data Exploration & Tests](https://github.com/mpuig/CarND-Behavioral-Cloning/blob/master/CarND-Behavioral-Cloning.ipynb) is used to test the package functions,  to visualize the results, and to execute the training processes. For example, to train the Nvidia model, it's three lines of code:

```
   from model.nvidia import NvidiaModel
   nvidia = NvidiaModel()
   nvidia.train_model(epochs=5, batch_size=256)
```

The `train_model` method, launches the keras training process, and it saves the results of each epoch to the directory `out`, using the pattern `model_{name}_{epoch}.json` for the model and `model_{name}_{epoch}.h5` for the weights. The training process uses the keras `fit_generator` method. This method uses a batch-by-batch python generator to generate data. The generator is run in parallel to the model, for efficiency, and it allows to do real-time data augmentation on images on CPU in parallel to training the model on GPU.

The `fit_generator` uses keras callbacks to save the model files, and to stop the training if the validation loss doesn't improve for 3 consecutive epochs.

To launch the training process, and also to evaluate the step-by-step functions, I've used the python notebook [CarND-Behavioral-Cloning.ipynb](https://github.com/mpuig/CarND-Behavioral-Cloning/blob/master/CarND-Behavioral-Cloning.ipynb). The results can be consulted there.

After several tests, the best results has been achived using a `samples_per_epoch` value of batch_size*200 (equivalent to 256*200, by default) and `epochs=15`


When the training process is done, the simulation can be launched using the command:

```
   python drive.py ./out/filename.json
```

The `drive.py` has been adapted to recognize the name of the model from the filename, and depending on the model, it applies a different preprocessing to the images from the simulator.

Using the modular architecture, three different network architectures has been implemented to test how they perform with the simulator:

1. NVIDIA: based on [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
2. Custom: based on NVIDIA, changing image sizes and a few other parameters.
3. Carputer: based on [Carputer](https://github.com/otaviogood/carputer)

After a lot of tests, the best results are obtained using the NVIDIA architecture. In fact, the other models weren't able to finish the simulation.


### Run

1. [Download the sample dataset for track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

2. Clone the project and install the required packages (based on Anaconda).


  ```
     git clone https://github.com/mpuig/CarND-Behavioral-Cloning
     cd CarND-Behavioral-Cloning
     conda env create -f environment.yml
     source activate CarND-Behavioral-Cloning
  ```

3. Install Tensorflow following the instructions from this link https://www.tensorflow.org/get_started/os_setup

  ```
     # Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
     # Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
     $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
     pip install --ignore-installed --upgrade $TF_BINARY_URL

  ```
     # Mac OS X, CPU only, Python 3.4 or 3.5:
     export PIP_REQUIRE_VIRTUALENV=false
     export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
     pip install --ignore-installed --upgrade $TF_BINARY_URL
  ```

4. Install Keras


  ```
     pip install keras
  ```

5. Launch the jupyter notebook


  ```
     jupyter notebook CarND-Behavioral-Cloning.ipynb
  ```

6. Go to  http://localhost:8888/notebooks/CarND-Behavioral-Cloning.ipynb in your browser and run all the cells. Everything should execute without error.


7. Launch the simulator and run autonomous driving with the command:


  ```
     python drive.py ./out/model_nvidia_09-0.03.json
  ```

***

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Keras](http://keras.io)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)
