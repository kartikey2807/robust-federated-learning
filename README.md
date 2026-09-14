## Byzantine-Robust Federated Learning with WCGAN

<div align="justify">Federated learning is a way to jointly train multiple ML models without exposing the underlying data. It comprises multiple clients and a single server. The flow looks as follows: the server shares a global model with a client subset (based on activity or network connection), clients train the model on their local data and send back updated weights, and then the server aggregates these updates to produce a new global model. Federated learning (FL) is preferred when data privacy and network latency are a concern. <i>But what if one or more clients are malicious and send poisoned weights?</i> This would compromise the aggregation process, and all clients would suffer as a result. To prevent this, robust aggregation methods have been investigated. These include trimmed mean, median, and <a href="https://arxiv.org/pdf/1703.02757">KRUM</a>. However, these methods either discard useful information or are susceptible to heterogeneous data distributions. <a href="https://arxiv.org/pdf/2503.20884">Usama et al.</a> suggested using a GAN model on the server with a modified discriminator. The discriminator is the global classifier itself, and we expect the trained GAN to generate samples that are representative of the underlying client dataset. During aggregation, clients are evaluated against this synthetic dataset and filtered out if their performance deviates significantly from the rest. However, in practice, the synthetic data generated did not resemble the underlying client data when we trained it on the MNIST and CIFAR-10 datasets, limiting model explainability. This research problem aims to fix this by changing the underlying model and applying distributed learning. Below is the training pipeline.
</div>
<br>

<img src="./images/pipeline.png">

<div align="justify">
Here we see that after one round of federated training, the generator can generate decent 'fake' samples. The clients are evaluated against these fake samples, and based on their classification accuracy, we can identify malicious ones.<i> It should be noted that this setup needs the client distributions to be IID.</i> The benign clients will then be aggregated. What we change is the model architecture in <i>GAN training</i>. The figure below depicts the applied distributed WCGAN.
</div>
<br>

<img src="./images/setup.png">

<div align="justify">
We have classifiers, which are the 'usual' FL training models, multiple discriminators, one on each client and one on the server, and a single generator model (also on the server side). The classifiers follow the vanilla FL training flow. The client discriminators are trained using the local dataset, and the gradient for the first component of the WGAN loss is computed. These gradients are also sent to the server, where they are robustly aggregated using KRUM. <i>Since KRUM is being applied to the discriminator updates and not the client models, we are not discarding any information.</i> On the server, the 'fake' samples are generated from Gaussian noise and conditioned on class labels (e.g., digits for MNIST). We pass these samples through the discriminator and the global classifier. The former makes sure the samples are close to client distributions, and the latter makes the generator conditional so we can control what we want to generate. Once updated, the discriminator and global model are sent back to clients. During averaging, the learned generator can now produce samples that are interpretable and can be used to classify clients. We inject noise in the gradient sharing step to prevent a man-in-the-middle attack and ensure differential privacy.
</div>
<br>

***How to run this setup***
```
python pipeline_train.py
```

I evaluated this setup on MNIST data, and it achieved about 80.91% accuracy (with DP).
