# Awesome Uncertainty Quantification

A paper list with short personal categorization/review of uncertainty quantification in ML/DL.

> Everything originated from the *apeiron*.                                 --- Anaximander

##
1. [On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty](http://bayesiandeeplearning.org/2021/papers/28.pdf)

   *Joost van Amersfoort, Lewis Smith, Andrew Jesson, Oscar Key, Yarin Gal*

   `NeurIPS 2021`

   `epistemic`  `GP`

   ...*propose to constrain DKL's feature extractor to approximately preserve distances through a bi-Lipschitz constraint, resulting in a feature space favorable to DKL*...



2.  [Evidential Deep Learning to Quantify Classification Uncertainty](https://dl.acm.org/doi/pdf/10.5555/3327144.3327239)

    *Murat Sensoy, Lance M. Kaplan, Melih Kandemir*

    `NeurIPS 2018`  \[[CODE](https://github.com/dougbrion/pytorch-classification-uncertainty)\]
   
    `epistemic`  `evidential learning`  `non-Bayesian`

    *...By placing a Dirichlet distribution on the class probabilities, we treat predictions of a neural net as subjective opinions and learn the function that collects the evidence leading to these opinions by a deterministic neural net from data...*

     **Why this matters**: This paper is the first to assemble evidential learning with deep learning on classification task. Many terms in the paper sounds really philosophical: experience (observation) -> evidence -> belief -> opinion. Btw, as epistemic uncertainty is known as "uncertainty on uncertainty", this paper shows the conjugate prior (Dirichlet distribution) of a multinomial likelihood distribution.



3. [Deep Evidential Regression](https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf)

   *Alexander Amini, Wilko Schwarting, Ava P. Soleimany, Daniela Rus*

   `NeurIPS 2022` \[[CODE](https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf)\]

   `aleatoric`  `epistemic`  `evidential learning`  `non-Bayesian`

   *...We accomplish this by placing evidential priors over the original Gaussian likelihood function and training the NN to infer the hyperparameters of the evidential distribution...*

   **Why this matters**: Again, this is the first to study the regression (continuous) counterpart on deep evidential learning. The Gaussian conjugate prior is the Normal Inverse-Gamma distribution. Some modification, e.g., evidence regularizer,  is adapted although the main framework is the same as the classification one. Btw, this paper is more readable from my personal view.

   

4. [Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods](https://link.springer.com/content/pdf/10.1007/s10994-021-05946-3.pdf)
   
   *Eyke Hüllermeier, Willem Waegeman*

   `Machine Learning 2021`

   `survey`  `aleatoric`  `epistemic`

   *...conventional approaches to probabilistic modeling, which are essentially based on capturing knowledge in terms of a single probabilistic distribution, fail to distinguish two inherently different sources of uncertainty, which are often referred to as aleatoric and epistemic uncertainty...*

   **Why this matters:** This survey about the two types of uncertainty is really well-written, readable and newbie-friendly. The examples about the discrimination of two uncertainties are helpful.



5. [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://arxiv.org/pdf/2011.06225.pdf)
   
   *Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu e, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U. Rajendra Acharya, Vladimir Makarenkov, Saeid Nahavandi*

   `Information Fusion 2021`

   `survey`



6. [What uncertainties do we need in Bayesian deep learning for computer vision](https://arxiv.org/pdf/1703.04977)

   *Alex Kendall, Yarin Gal*

   `NeurIPS 2017` [[CODE](https://github.com/pmorerio/dl-uncertainty)] [[BLOG](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/uncertainty_bdl.html)]

   `Bayesian`  `epistemic`  `aleatoric`  `MC dropout`

   *...we present a Bayesian deep learning framework combining input-dependent aleatoric uncertainty together with epistemic uncertainty...*

   **Why this matters:** This paper proposes a learned loss attenuation to model heteroscedastic (aleatoric) uncertainty. The interpretation of the loss is insightful.



7. [Dropout as a Bayesian approximation: representing model uncertainty in deep learning](http://proceedings.mlr.press/v48/gal16.pdf)

   *Yarin Gal, Zoubin Ghahramani*

   `ICML 2016` [[CODE](https://github.com/yaringal/DropoutUncertaintyExps)]  [[BLOG](https://ahmdtaha.medium.com/dropout-as-a-bayesian-approximation-representing-model-uncertainty-in-deep-learning-7a2e49e64a15)]  [[Appendix](http://proceedings.mlr.press/v48/gal16-supp.pdf)]

   `Bayesian`  `epistemic   MC dropout`

    ...we develop a new theoretical framework casting dropout training in deep neural networks (NNs) as approximate Bayesian inference in deep Gaussian processes...

   **Why this matters:** As the title shows, the paper formalizes dropout as a Bayesian approximation theoretically. In other words, it combines the efficiency of DL (practically) and the stochasticity of BNN (theoretically). It provides a Bayesian view on an otherwise trick on DL the same way as a probabilistic perspective on NN. 
   
  
8. [Quantifying Uncertainties in Natural Language Processing Tasks](https://ojs.aaai.org/index.php/AAAI/article/view/4719/4597)

   *Yijun Xiao and William Yang Wang*

   `AAAI 2019` 

   `epistemic and aleatoric uncertainty`  `NLP`

    ...we focus on exploring the ben-efits of quantifying both model and data uncertainties in thecontext of various natural language processing (NLP) tasks...

   **Why this matters:** The paper summarizes UQ on some NLP tasks to give some insights on language uncertainty, which is different from mainstream CV tasks. The summary of data/model uncertainty is so elegent and beautiful - Var(y) = Var(E\[y|x\])+E\[Var(y|x)\] = Um + Ud.
