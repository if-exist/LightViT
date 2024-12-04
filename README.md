## Requirement

```torch>=1.7.0; torchvision>=0.8.0; pyyaml; timm==0.6.13;``

dataset: Tiny ImageNet

## Train

```torchrun --nproc_per_node=4 train.py -c configs/config.yaml --model lightvit --experiment LightViT```

## Benchmark

```python3 benchmark_onnx.py --model lightvit --input-size 3 224 224 --benchmark_cpu```

## Acknowledgement

Our code base is partly built with [SwiftFormer](https://github.com/Amshaker/SwiftFormer), [EfficientMod](https://github.com/ma-xu/EfficientMod/tree/main), [image_classification_sota](https://github.com/hunto/image_classification_sota) and [EfficientFormer](https://github.com/snap-research/EfficientFormer/tree/main).

Thanks for the great implementations!

