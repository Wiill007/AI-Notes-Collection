# Production test
## ***CHECK OUT aiquizzes.com***

## Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
If the bear is in circumstances not seen during our data, it's most sure that the model will fail. Examples of this will be if the bear is far from camera, partially obstructed or in a posture not seen in the dataset.

## Where do text models currently have a major deficiency?
In generating *correct* text because they tend to have major hallucinations.

## What are possible negative societal implications of text generation models?
This is dangerous because we can create some content that sounds really compelling and accurate while it's completely the other way around. As a consequence, these models can be used to spread misinformation.

## In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
Generally, if these mistakes can be harmful the model shouldn't be even used in the first place unless we have an accuracy over 99%. Fundamentally, we'd need to deploy early models to test in a controlled environment, make use of the feedback from human data and improve it to get to a better mark.

## What kind of tabular data is deep learning particularly good at?
Natural language (book titles, reviews, etc.), and high-cardinality categorical columns (i.e., something that contains a large number of discrete choices, such as zip code or product ID)

## What's a key downside of directly using a deep learning model for recommendation systems?
It doesn't show useful recommendations but rather recommendations the user might like.

## What are the steps of the Drivetrain Approach?
1. Considering your objective.
2. Think actions to complete said objective.
3. What data do we have or can acquire that can help with it.

## How do the steps of the Drivetrain Approach map to a recommendation system?
1. The objective is to leverage the sales made by our company.
2. Well of course recommending items the user would not buy without the recommendations.
3. New data must be collected to generate recommendations that will cause new sales. This will require conducting many randomized experiments in order to collect data about a wide range of recommendations for a wide range of customers. 

## Create an image recognition model using data you curate, and deploy it on the web.

## What is DataLoaders class?
**DataLoaders** is a thin class that just stores whatever **DataLoader** objects you pass to it, and makes them available as **train and valid**.

## What four things do we need to tell fastai to create DataLoaders?
- What kinds of data we are working with
- How to get the list of items
- How to label these items
- How to create the validation set

## What does the splitter parameter to DataBlock do?
Split validation and training sets randomly (well, using a given seed)

## How do we ensure a random split always gives the same validation set?
Fixing the seed value to a specific number

## What letters are often used to signify the independent and dependent variables?
x and y

## What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?


## What is data augmentation? Why is it needed?
Modify the data we already have in ways that make the neural network learn intrinsic features of the data we have. In case of images this would be the use of rotations and crops.

## What is the difference between item_tfms and batch_tfms?
item_tfms will resize our images so that they are the same size while batch_tfms will be used in order to data augmentate using rotation, flipping, perspective warping, brightness changes and contrast changes.

## What is a confusion matrix?
Matrix that shows the amount of times a model performs predictions correctly, wrongly and which is which, meaning it shows the predicted value and the number of times it was correct and the number of times it was other classes.

## What does export save?
export.pkl which is needed for deployment.

## What is it called when we use a model for getting predictions, instead of training?
Inference.

## What are IPython widgets?
GUI components that bring together Javascript and Python in a web browser.

## When might you want to use CPU for deployment? When might GPU be better?
CPU for the starting. GPU might be better if it really is worth it after the work gets popular enough.

## What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
The steps to incorporate all libraries required for the model to work directly on client software are more and harder because they do not always support PyTorch and fastai layers we may use.

## What are three examples of problems that could occur when rolling out a bear warning system in practice?
1. Data might not be adequated to video.
2. Data might not have seen night.
3. Data might not have seen low quality images (camera compressions)
4. Ensuring results are returned fast enough to be useful in practice
5. Recognizing bears in positions that are rarely seen in photos that people post online (for example from behind, partially covered by bushes, or when a long way away from the camera)

## What is "out-of-domain data"?
That is to say, there may be data that our model sees in production which is very different to what it saw during training. There isn't really a complete technical solution to this problem; instead, we have to be careful about our approach to rolling out the technology.

## What is "domain shift"?
The type of data that our model sees changes over time.

## What are the three steps in the deployment process?
1. Manual process
2. Limited scope deployment
3. Gradual expansion
