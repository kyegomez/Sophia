# Research Analysis: Sophia Paper's Training Strategy




## Architecture

    Model: Autoregressive models on openwebtext
    context length = 1024
    model type: decoder only transformer
    model sizes: 125m (small), 355M(medium), and 770m(large)


    datasets:
        openwebtext(Gokasalan Cohen)


    Baselines
        Adam with decoupled weight decay AdamW(Hutter 2017)
        Lion


## Algorithmic Pseudocode 

Initialize the model (GPT-2) with the desired number of parameters (small, medium, or large).

Load the OpenWebText dataset.

Set the context length to 1024.

Set the batch size to 480.

Use a cosine learning rate schedule with the final learning rate equal to 0.05 times the peak learning 
rate.

Apply gradient clipping with a threshold of 1.0.

Use a fixed 2k steps of learning rate warm-up.

Train the model using the Sophia optimizer with the chosen Hessian estimator (Sophia-H or Sophia-G) and hyperparameters.

Train the model for 100K, 200K, or 400K steps.

Evaluate the model using log perplexity on OpenWebText and in-context learning results on SuperGLUE.
