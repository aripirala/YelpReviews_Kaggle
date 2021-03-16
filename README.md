# YelpReviews_Kaggle
![Yelp Reviews]

## Project Description
In this we attempt to build a simple sentiment classifier with 2 classes *positive* and *negative*. Most of the code is based on the book by Delip Rao and Brian McMahan - Natural Language Processing

## Folder Structure
- **src**: All the code resides in this folder
  -  *configs.py*: contains all the configs related to the project
  -  *utils.py*: contains all the helper functions
  -  *dataset.py*: contains the methods that help generate pytorch dataset that is needed for the deep learning model
  -  *model.py*: contains all pytorch models we would be trying in the project
  -  *train.py*: entry point which uploads the configs, creates a dataset/dataloader, builds a model and finally saves all the artifacts in the experiment folder
- **input**: All the data such as csv, tfrecords or any flat files resides in this folder. You need to create this folder before you run the code
- **experiment**: Any experiment you are creating should be set in the src/config.py. Then when you run train.py it automatically saves all the artifacts created during the model building in that folder
