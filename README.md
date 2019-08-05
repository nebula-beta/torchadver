

## Introduction

torchadver is a Pytorch tool box for generating adversarial images.









## Installation



## How to Use

### Generate adversarial images by satisfy L2 norm
```
from torchadver.attacker.iterative_gradient_attack import FGM_L2, I_FGM_L2, MI_FGM_L2, M_DI_FGM_L2


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

images, labels = ...
model = ...

attacker = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)

# for non-targeted attack
adv_images = attacker.attack(images, labels) # or adv_images = attacker.attack(images)
```


### Generate adversarial images by satisfy linf norm


```
from torchadver.attacker.iterative_gradient_attack import FGM_LInf, I_FGM_LInf, MI_FGM_LInf, M_DI_FGM_LInf


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

images, labels = ...
model = ...
targeted_labels = ...

attacker = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True, targeted=True)

# for non-targeted attack
adv_images = attacker.attack(images, targeted_labels)
```


## Citations





