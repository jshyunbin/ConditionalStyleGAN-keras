# Conditional StyleGAN
CSGAN(Conditional StyleGAN) implementation with Keras. Used CelebA datasets(64 x 64 x 3) and 5 classes for training.

## Prerequisites

- tensorflow-gpu: 1.14.0
- Keras: 2.3.1
- tensorboard: 1.14.0
- h5py: 2.10.0
- numpy: 1.18.1

## How to use

**You need to prepare your own dataset in the dataset/ directory.**

```
python model.py --model=CSGAN --gpu=2 --name=csgan-test --load_model=3000
```

There are 8 flags for training/validating parameters. 

| Flags      | type    |                                                                                           | default |
|------------|---------|-------------------------------------------------------------------------------------------|---------|
| load_model | integer | Iteration number of the model you wish to load. '-1' for training a new model             | -1      |
| validate   | boolean | Set to 'True' for validating a model and 'False' for training.                            | False   |
| glasses    | boolean | When set 'True', generates 'glasses' validate images.(Only works when validate is 'True') | False   |
| male       | boolean | When set 'True', generates 'male' validate images. (Only works when validate is 'True')   | False   |
| model      | enum    | Select from ['CSGAN', 'ACGAN', 'CGAN'].                                                   | CSGAN   |
| gpu        | integer | The gpu number for training.                                                              | 0       |
| name       | string  | The name of your model. Will be used for saving model and validate images.                | None    |
| batch_size | integer | The batch size for training the model.                                                    | 64      |

