

## Introduction

*torchadver* is a Pytorch tool box for generating adversarial images. The basic adversarial attack are implemented. Such as [FSGM](https://arxiv.org/abs/1412.6572), [I-FGSM](https://arxiv.org/abs/1607.02533), [MI-FGSM](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html), [M-DI-FGSM](https://arxiv.org/abs/1803.06978), [C&W](https://ieeexplore.ieee.org/abstract/document/7958570) .etc.






## Installation



## How to Use

The brief attack process is shown below. More detailed process introduction you can refer to [`./examples/toturial.py`](https://github.com/nebula-beta/torchadver/blob/master/examples/toturial.py).
### Generate adversarial images by satisfy L2 norm

**Non-targeted attack**
```python
from torchadver.attacker.iterative_gradient_attack import FGM_L2, I_FGM_L2, MI_FGM_L2, M_DI_FGM_L2


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# images normalized by mean and std
images, labels = ...
model = ...

# use mean and std to determine effective range of pixel of image in channels.
attacker = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(),
				  mean=mean, std=std, 
				  max_norm=4.0, # L2 norm bound
				  random_init=True)

# for non-targeted attack
adv_images = attacker.attack(images, labels) # or adv_images = attacker.attack(images)
```

**Targeted attack**
```python
from torchadver.attacker.iterative_gradient_attack import FGM_L2, I_FGM_L2, MI_FGM_L2, M_DI_FGM_L2


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# images normalized by mean and std
images, labels = ...
model = ...
targeted_labels = ...

# use mean and std to determine effective range of pixel of image in channels.
attacker = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(),
				  mean=mean, std=std, 
				  max_norm=4.0, # L2 norm bound
				  random_init=True)

# for non-targeted attack
adv_images = attacker.attack(images, targeted_labels)
```

### Generate adversarial images by satisfy Linf norm


**Non-targeted attack**
```python
from torchadver.attacker.iterative_gradient_attack import FGM_LInf, I_FGM_LInf, MI_FGM_LInf, M_DI_FGM_LInf


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# images normalized by mean and std
images, labels = ...
model = ...

# use mean and std to determine effective range of pixel of image in channels.
attacker = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(),
				 mean=mean, std=std,
				 max_norm=0.1, # Linf norm bound
				 random_init=True)

# for non-targeted attack
adv_images = attacker.attack(images, labels) # or adv_images = attacker.attack(images)
```

**Targeted attack**
```python
from torchadver.attacker.iterative_gradient_attack import FGM_LInf, I_FGM_LInf, MI_FGM_LInf, M_DI_FGM_LInf


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# images normalized by mean and std
images, labels = ...
model = ...
targeted_labels = ...

# use mean and std to determine effective range of pixel of image in channels.
attacker = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(),
				 mean=mean, std=std,
				 max_norm=0.1, # Linf norm bound
				 random_init=True, targeted=True)

# for non-targeted attack
adv_images = attacker.attack(images, targeted_labels)
```

## Citations

More information about adversarial attack about deep learning, refer to [awesome-adversarial-deep-learning](https://github.com/nebula-beta/awesome-adversarial-deep-learning).




