
# ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning
<!-- schematic: -->
<div style="text-align: center;">
  <img src="sample_figs/schematic.png" alt="Architecture Diagram" width="98%" style="display: inline-block;" />
</div>
<!-- links: -->
<hr>
<div align="center">
  <a href="https://reinflow.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Visit-Website-007bff?style=for-the-badge" />
  </a>
  <a href="https://drive.google.com/drive/folders/11LZmP5Fi_aBTwZXKC4yPzSuXr2o5OoE9?usp=drive_link" target="_blank">
    <img alt="Checkpoints" src="https://img.shields.io/badge/Download-Checkpoints-blue?style=for-the-badge" />
  </a>
  <a href="https://arxiv.org/abs/2505.22094" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.22094-b31b1b?style=for-the-badge" />
  </a>
</div>
<!-- mini table of contents: -->
<p align="center">
  <a href="#1-installation">Installation</a> |
  <a href="#2-quick-start-reproduce-our-results">Quick Start</a> |
  <a href="#3-implementation-details">Implementation Details</a> |
  <a href="#4-adding-your-own-dataset-or-environment">Add Dataset/Environment</a> <br>
  <a href="#5-debug-aid-and-known-issues">Debug & Known Issues</a> |
  <a href="#6-license">License</a> |
  <a href="#7-citation">Citation</a> |
  <a href="#8-acknowledgement">Acknowledgement</a>
</p>


## :star: Comming Soon
- A comprehensive webpage
- WandB project
- Supporting Fine-tuning [Mean Flow](https://arxiv.org/abs/2505.13447) with online RL

## :rocket:  Installation
Please follow the steps in [installation/reinflow-setup.md](./installation/reinflow-setup.md).

## :rocket: Quick Start: Reproduce Our Results
To fully reproduce our experiments, please refer to [ReproduceExps.md](docs/ReproduceExps.md). 
To download our training data and reproduce the plots in the paper, please refer to [ReproduceFigs.md](docs/ReproduceFigs.md).

## :rocket: Implementation Details
Please refer to [Implement.md](docs/Implement.md) for descriptions of key hyperparameters of FQL, DPPO, and ReinFlow.

## :rocket: Adding Your Own Dataset or Environment
Please refer to [Custom.md](docs/Custom.md).

## :rocket: Debug Aid and Known Issues
Please refer to [KnownIssues.md](docs/KnownIssues.md) to see how to resolve errors you encounter.

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).

## Citation
If you find our work inspiring your own ideas, please consider to cite our [arXiv paper](https://arxiv.org/abs/2505.22094). 

## Acknowledgement
This repository was developed from multiple open-source projects. Major references include:  
- [TorchCFM, Tong et al.](https://github.com/atong01/conditional-flow-matching): Conditional flow-matching repository.  
- [Shortcut Models, Francs et al.](https://github.com/kvfrans/shortcut-models): One-step Diffusion via Shortcut Models. 
- [DPPO, Ren et al.](https://github.com/irom-princeton/dppo): DPPO official implementation.  

For more references, please refer to [Acknowledgement.md](docs/Acknowledgement.md).