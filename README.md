<h1 align='center'>Rectified Flow Matching</h1>

Cutting-edge and feature-rich implementation of Rectified Flows from [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flows](https://arxiv.org/abs/2209.03003) in `jax` and `equinox`.

Features:
- Deterministic and stochastic sampling of the associated ODE and SDE respectively,
- array-typed to-the-teeth for dependable execution with `jaxtyping` and `beartype`.

To implement:
- [ ] Guidance by score of conditioning
- [x] Mixed precision
- [x] EMA
- [x] AdaLayerNorm
- [x] Stochastic sampling
- [x] ODE Sampling
- [x] Likelihoods
- [ ] DiT
- [ ] Hyperparameter/model saving

```bibtex
@misc{liu2022flowstraightfastlearning,
      title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow}, 
      author={Xingchao Liu and Chengyue Gong and Qiang Liu},
      year={2022},
      eprint={2209.03003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2209.03003}, 
}
```

```bibtex
    @misc{lipman2023flowmatchinggenerativemodeling,
        title={Flow Matching for Generative Modeling}, 
        author={Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le},
        year={2023},
        eprint={2210.02747},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2210.02747}, 
    }
```

```bibtex
@misc{yang2024consistencyflowmatchingdefining,
      title={Consistency Flow Matching: Defining Straight Flows with Velocity Consistency}, 
      author={Ling Yang and Zixiang Zhang and Zhilong Zhang and Xingchao Liu and Minkai Xu and Wentao Zhang and Chenlin Meng and Stefano Ermon and Bin Cui},
      year={2024},
      eprint={2407.02398},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.02398}, 
}
```