# Defending Against Adversarial Examples with K-Nearest Neighbor   
https://arxiv.org/abs/1906.09525

### Model Weights
- MNIST
  - Basic CNN: [mnist_basic.h5](saved_models/mnist/mnist_basic.h5)
  - L2 Adversarial Training (l2-Adv): [mnist_at.h5](saved_models/mnist/mnist_at.h5)
  - Soft Nearest Neighbor Loss (SNNL): [mnist_snnl.h5](saved_models/mnist/mnist_snnl.h5)
  - Hidden Mixup: [mnist_hidden_mixup.h5](saved_models/mnist/mnist_hidden_mixup.h5)
  - Input Mixup: [mnist_input_mixup.h5](saved_models/mnist/mnist_input_mixup.h5)
  - VAE: [mnist_vae.h5](saved_models/mnist/mnist_vae.h5)
  - Autoencoder (AE): [mnist_ae.h5](saved_models/mnist/mnist_ae.h5)
  - L2 Adversarially trained Autoencoder (l2-Adv-AE): [mnist_basic.h5](saved_models/mnist/mnist_ae_at.h5)
  - Rotation Prediction: [mnist_basic.h5](saved_models/mnist/mnist_rot.h5)
  - L2 Adversarially trained Rotation Prediction (l2-Adv-Rot): [mnist_basic.h5](saved_models/mnist/mnist_rot_at.h5)
- CIFAR-10
  - Basic ResNet20 (ResNet): [cifar10_basic_rn.h5](saved_models/cifar10/cifar10_basic_rn.h5)
  - L2 Adversarially Trained ResNet20 (l2-Adv ResNet): [cifar10_at_rn.h5](saved_models/cifar10/cifar10_at_rn.h5)

Chawin Sitawarin (chawins@berkeley.edu)  
David Wagner (daw@cs.berkeley.edu)
