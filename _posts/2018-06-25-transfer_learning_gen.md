---
toc: true
layout: post
description: Leveraging transfer learning to improve generalization capabilities of deep learning models 
categories: [transfer-learning, nlp, deep-learning]
title:  Using Transfer Learning to Introduce Generalization in Models
image: images/transfer_learning_gen/cover.jpg
---
The following article first appeared on [Intel Developer Zone](https://software.intel.com/en-us/articles/part-1-using-transfer-learning-to-introduce-generalization-in-models).

# Abstract
Researchers often try to capture as much information as they can, either by using existing architectures, creating new ones, going deeper, or employing different training methods. This paper compares different ideas and methods that are used heavily in Machine Learning to determine what works best. These methods are prevalent in various domains of Machine Learning, such as Computer Vision and Natural Language Processing (NLP).

# Transfer Learning is the Key

Throughout our work, we have tried to bring generalization into context, because that’s what matters in the end. Any model should be robust and able to work outside your research environment. When a model lacks generalization, very often we try to train the model on datasets it has never encountered … and that’s when things start to get much more complex. Each dataset comes with its own added features which we have to adjust to accommodate our model.

One common way to do so is to transfer learning from one domain to another.

Given a specific task in a particular domain, for which we need labelled images for the same task and domain, we train our model on that dataset. In practice, the dataset is usually the largest in that domain so that we can leverage the features extracted effectively. In computer vision, it’s mostly Imagenet, which has 1,000 classes and more than 1 million images. When training your network upon it, it’s bound to extract [features](https://arxiv.org/abs/1311.2901) that are difficult to obtain otherwise. Initial layers usually capture small, fine details, and as we go deeper, ConvNets try to capture task-specific details; this makes ConvNets fantastic feature extractors.

Normally we let ConvNet capture features by training it on a larger dataset and then modify. Fully connected layers in the end can do whatever we require for carrying out classification, and we can add a combination of linear layers. This makes it easy to transfer the knowledge of our network to carry out another task.

# Transfer Learning in Natural Language Processing

A recent paper, Universal LM for Text Classification,3 showed how to apply transfer learning to Natural Language Processing. This method has not been applied widely in this field. We can use pretrained models and not embeddings that have been trained on WikiText 103. Embeddings are word representations that allow words with similar meaning to have similar representation. If you visualize their embeddings, they would appear close to one another. It’s basically a fixed representation, so their scope is limited in some ways. But, creating a language model that has learned to capture semantic relationships within languages is bound to work better on newer datasets, as evidenced by results from the paper. So far, it has been tested on Language Modeling tasks and the results are impressive. This applies to Seq2Seq learning as well in instances where length of inputs and outputs is variable. This can be expanded further to many other tasks in NLP. Read more: [Introducing state of the art text classification with universal language models](http://nlp.fast.ai/).
![lm_stages](https://software.intel.com/sites/default/files/managed/9d/76/transfer-learning-to-introduce-generalization-in-models-fig2-sm.jpg)

# Learning without Forgetting

Another paper, [Learning without Forgetting](https://arxiv.org/abs/1606.09282), provides context for what’s been done earlier to make our network remember what it was trained on earlier, and how it can made to remember new data without forgetting earlier learning. The paper discussed the researchers’ methods compared with other prevalent, widely used methods such as transfer learning, joint training, feature extraction, and fine tuning. And, they tried to capture differences in how learning is carried out.

For example, fine tuning is an effective way to extend the learning of neural networks. Using fine tuning, we usually train our model on a larger dataset – let’s say ResNet-50 trained on Imagenet trained on ImageNet. A pretrained ResNet5 has 25.6 Million parameters. [ResNet](https://arxiv.org/abs/1512.03385) let you go deeper without incrementing the number of parameters over counterparts. The number of parameters is so great that you can expect to use the model to fit any other dataset in a very efficient manner: you simply load the model, remove the fully connected layers which are task specific, freeze the model, add linear layers as per your own needs, and train it on your own dataset. It’s that simple and very effective. The trained model has so many capabilities and reduced our workload by a huge factor; we recommend using fine tuning wherever you can.

## What We’ve Actually Been Doing: Curve Fitting

Judea Pearl recently published a [paper](https://arxiv.org/abs/1801.04016) in which he states that although we have gained a strong grasp of probability, we still can’t do cause-and-effect reasoning. Instead, basically what we’ve doing is curve fitting. So many different domains can be unlocked with do-calculus and causal modelling.
![causal_hierarchy](/images/transfer_learning_gen/causal_hierarchy.png)

Returning to where we were, we implemented learning without forgetting to measure how well the model does compared to other discussed methods in some computer vision tasks. They define three types of parameters: θs, θ o, and θn. θs are the shared set of parameters, while θ o is a parameter the model has trained on previous tasks (with a different dataset). Θn is a parameter the model will have when trained on another dataset.

# How to Perform Training

First, we used ResNet-50 (authors used 5 conv layers + 2 FC layers of AlexNet) instead of stated architecture with pretrained weights. The purpose behind pretrained weights is that our model will be used in domain adaptation and will see increased use of fine tuning. It’s necessary that the convolutional layers have extracted rich features that will help in many computer vision tasks, preferably on ImageNet, which has 26.5 million parameters. If you want to go deep, consider using other ResNet variants like ResNet-101. After that, our model must be trained using the architecture as prescribed in the paper: 
![lwf_arch](https://software.intel.com/sites/default/files/managed/9d/76/transfer-learning-to-introduce-generalization-in-models-fig3.png)

The model in between is ResNet-50 as per our implementation. We removed the last two layers and added two FC (fully connected) layers. We dealt with FC layers in a different manner appropriate to our task, but it can be modified for each use case. Add multiple FC layers depending on how many tasks you plan to perform.

After creating the architecture, it’s necessary to freeze the second FC layer. This is done to ensure that the first FC layer can perform better on this task when the model is learned on another task with a significantly lower learning rate.

This method solves a big challenge: after training, the older dataset is no longer required, whereas other methods of training do still require it.
![lwf_features](/images/transfer_learning_gen/lwf_features.png)

This is a big challenge: to make incremental learning more natural, dependence on older datasets must be removed. After training the model we are required to freeze the base architecture (in our case it implies ResNet-50) and the first FC layer with only the second FC layer turned on. We have to train the model with this arrangement.

## The Rationale for this Training Approach
The base model (ResNet in our case) earlier had fine-tuned weights. Convolutional layers do an excellent job of feature extraction. As we fine-tune the base model, we are updating the weights as per the dataset we’re using. When we freeze the base model and train with another FC layer turned on, it implies that we have gone task specific, but we don’t want go much deeper into that task. By training the base model on a particular task and re-training it, the model will capture the weights required to perform well on the default dataset. If we want to perform domain adaptation, earlier and middle layers should be very good at feature extraction and bring generalization into context rather than making it task-specific.

![lwf_method](https://software.intel.com/sites/default/files/managed/ff/be/transfer-learning-to-introduce-generalization-in-models-fig4.png)

After performing the training, we must join train all the layers. This implies turning on both FC layers of the base model and training them to converge.

Use any loss function your task requires. The authors used modified cross entropy (knowledge distillation loss), which proved to work well for encouraging the outputs of one network to approximate the outputs of another.

![loss_function](https://software.intel.com/sites/default/files/managed/9d/76/transfer-learning-to-introduce-generalization-in-models-fig5.png)

# Observations

This method seems to work well when the number of tasks is kept to a minimum (in our case, two). It may outperform fine-tuning for new tasks because the base model is not being retrained repeatedly, only the FC layers. Performance is similar to joint training when new tasks are being added. But, this method is bound to work poorly on older tasks as new tasks are added.

This is because same convolutional layers are being used when we are freezing them, which means they are using the same feature extractor. We don’t expect them to outperform on all above-mentioned training tasks just by dealing with FC layers.
![task_specific_diagram](https://software.intel.com/sites/default/files/managed/9d/76/transfer-learning-to-introduce-generalization-in-models-fig6.png)

You can add more task-specific layers to introduce more generalization. But, as you go deep, you will make the model task-specific. This method addresses the problem of adapting to different domains of computer vision without relying on older datasets that were used in earlier training. It can be regarded as a hybrid of knowledge distillation and fine-tuning training methods.

This is an incremental step toward bringing generalization to neural networks, but we still lack ways to achieve full generalization, wherein we can expect to make our networks learn just like we do. We still have a long way to go, but research is in progress.