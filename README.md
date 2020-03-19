# Adversarial Examples on KNN (and its neural network friends)
This repo contains code for two very related papers:
1. (Deprecated) Defending Against Adversarial Examples with K-Nearest Neighbor   
https://arxiv.org/abs/1906.09525
2. Minimum-Norm Adversarial Examples on KNN and KNN-Based Models  
https://arxiv.org/abs/2003.06559

## Defending Against Adversarial Examples with K-Nearest Neighbor

### Notice
This code is **DEPRECATED** because we found that the
empirical results reported are **INACCURATE**. Specifically, we developed a
stronger attack (the second paper, version 2) that manages to find adversarial examples
with smaller L2 perturbation than originally reported according to our [first
version of the attack](https://arxiv.org/abs/1903.08333). The bottom line is
our method does not offer a significant improvement over _Adversarial Training_
(Madry et al.) except a possible increase on clean accuracy. Please see
_Minimum-Norm Adversarial Examples on KNN and KNN-Based Models_ for the attack
desceiption.

### Abstract
Robustness is an increasingly important property of machine learning models as they become more and more prevalent. We propose a defense against adversarial examples based on a k-nearest neighbor (kNN) on the intermediate activation of neural networks. Our scheme surpasses state-of-the-art defenses on MNIST and CIFAR-10 against l2-perturbation by a significant margin. With our models, the mean perturbation norm required to fool our MNIST model is 3.07 and 2.30 on CIFAR-10. Additionally, we propose a simple certifiable lower bound on the l2-norm of the adversarial perturbation using a more specific version of our scheme, a 1-NN on representations learned by a Lipschitz network. Our model provides a nontrivial average lower bound of the perturbation norm, comparable to other schemes on MNIST with similar clean accuracy.

### Model Weights
- MNIST
  - Basic CNN: [mnist_basic.h5](saved_models/mnist/mnist_basic.h5)
  - L2 Adversarial Training (l2-Adv): [mnist_at.h5](saved_models/mnist/mnist_at.h5)
  - Soft Nearest Neighbor Loss (SNNL): [mnist_snnl.h5](saved_models/mnist/mnist_snnl.h5)
  - Hidden Mixup: [mnist_hidden_mixup.h5](saved_models/mnist/mnist_hidden_mixup.h5)
  - Input Mixup: [mnist_input_mixup.h5](saved_models/mnist/mnist_input_mixup.h5)
  - VAE: [mnist_vae.h5](saved_models/mnist/mnist_vae.h5)
  - Autoencoder (AE): [mnist_ae.h5](saved_models/mnist/mnist_ae.h5)
  - L2 Adversarially trained Autoencoder (l2-Adv-AE): [mnist_ae_at.h5](saved_models/mnist/mnist_ae_at.h5)
  - Rotation Prediction: [mnist_rot.h5](saved_models/mnist/mnist_rot.h5)
  - L2 Adversarially trained Rotation Prediction (l2-Adv-Rot): [mnist_rot_at.h5](saved_models/mnist/mnist_rot_at.h5)
- CIFAR-10
  - Basic ResNet20 (ResNet): [cifar10_basic_rn.h5](saved_models/cifar10/cifar10_basic_rn.h5)
  - L2 Adversarially Trained ResNet20 (l2-Adv ResNet): [cifar10_at_rn.h5](saved_models/cifar10/cifar10_at_rn.h5)

## Minimum-Norm Adversarial Examples on KNN and KNN-Based Models

### Abstract
We study the robustness against adversarial examples of kNN classifiers and classifiers that combine kNN with neural networks. The main difficulty lies in the fact that finding an optimal attack on kNN is intractable for typical datasets. In this work, we propose a gradient-based attack on kNN and kNN-based defenses, inspired by the previous work by Sitawarin & Wagner [1]. We demonstrate that our attack outperforms their method on all of the models we tested with only a minimal increase in the computation time. The attack also beats the state-of-the-art attack [2] on kNN when k > 1 using less than 1% of its running time. We hope that this attack can be used as a new baseline for evaluating the robustness of kNN and its variants.

### Related Files
- Attack implementation: `lib/dknn_attack_v2.py` [[link]](lib/dknn_attack_v2.py)
- Base Deep kNN model: `lib/dknn.py` [[link]](lib/dknn.py)
- Dubey et al. model and attack: `lib/knn_defense.py` [[link]](lib/knn_defense.py)

Note that kNN and all kNN-based models we evaluated (except for Dubey et al.)  
can be represented by `DKNNL2` class. Please see `attack_demo.ipynb` for an
example of the attack usage, and feel free to leave any question/suggestion by
opening an issue.

## Authors
Chawin Sitawarin (chawins@eecs.berkeley.edu)  
David Wagner (daw@cs.berkeley.edu)
