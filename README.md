# SplitSR
Unofficial implementation of [SplitSR: An End-to-End Approach to Super-Resolution on Mobile Devices](https://arxiv.org/abs/2101.07996)

![a) SplitSRBlock, b) SplitSR ](assets/network.png)

## Keys from the Paper
- Split convolution splits input by alpha ratio along depth channel.
- The conv-processed part is concatenated at the end.
- By the second key point, every channel would be processed after 1/α blocks.


## Progress
- Splitted Convolution Block is implemented.
