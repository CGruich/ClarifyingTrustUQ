![](markdown/UQ_Materials_Discovery_Sample.png).

**Clarifying Trust of Materials Property Predictions using Neural Networks with Distribution-specific Uncertainty Quantification**
___
Uncertainty quantification (UQ) refers to some measure of variability, often as a result of imperfect or incomplete information. In the context of AI models, the predictive uncertainty could arise from an imperfectly designed model. In the context of materials predictions, the uncertainty could arise from the materials dataset itself, such as dataset sparsity. Often, different sources of uncertainty are significant and at play in AI-guided materials screening strategies. **UQ methods quantify the uncertainty arising from these sources.**

**Many advanced materials discovery strategies are enhanced by trustworthy UQ.** For example, if the uncertainty intervals of materials predictions behave similarly to a confidence interval, then the uncertainty can be used to infer the accuracy of millions of materials predictions without having to explicitly simulate millions of materials or synthesize them in the lab—this is uncertainty-enhanced high-throughput screening.

Neural networks are a class of AI model that often outperform other model classes on diverse, sparse, challenging materials datasets, **yet the uncertainty of neural networks, their associated materials tasks, and datasets are not well-understood and not well-adopted—thereby limiting their use in advanced materials discovery strategies.**

Uncertainty arises from different sources in AI-guided screening efforts. Moreover, independent of the mathematics, the predictive uncertainty is amenable to different interpretations. UQ methods can wildly differ in how they quantify the uncertainty of materials predictions. **This work addresses two key questions:**

1. **If uncertainty is a general concept that is amenable to interpretation, then what is a common framework to assess trustworthy uncertainty quantification across applications?**
   
2. **Which UQ methods are suggestible as practical, tractable, and trustworthy options for advanced materials discovery strategies that represent a larger multi-component software effort?**
___

