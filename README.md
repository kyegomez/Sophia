# Sophia Optimizer

Cut Model Training Cost by 50%? with this all-new simple plug in and play Optimizer: Sophia

[Sophia: A Scalable Stochastic Second-order Optimizer for
Language Model Pre-training](https://arxiv.org/pdf/2305.14342.pdf)

Now with training file ready in experiments folder! 🔥🔥🔥

Sophia is an second order clipped stochastic optimization algorithm that uses an inexpensive stochastic estimate of the diagonal of the Hessian as an pre-conditioner and a clipping mechanism to control the worst case update size. It achieves better performance than adam in terms of validation pre-traing loss, total compute, and wall-clock time. By cutting model training cost in half, Sophia can help save millions if not billions of dollars in computational resources.


## Benefits

Sophia achievs the same validation pre training loss with 50% fewer number of steps than Adam

50% less total compute and 50% less wall-clock time

Seamless integration into existing training pipelines -- plug in and play!

No special requirments on model architecture or computing infrastructure

Supports both Hutchinson and Gauss-Newton-Bartlett Hessian Estimators



## 👥 Share With Your Friends
If you find Sophia useful, please share this GitHub repository with your friends and colleagues. Let's cut the cost of AI training together!

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FSophia&text=Check%20out%20Sophia%20Optimizer%20-%20a%20second-order%20clipped%20stochastic%20optimization%20algorithm%20that%20cuts%20model%20training%20cost%20in%20half!%20%23DeepLearning%20%23AI%20%23Optimization)

[Share on Linkedin](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FSophia&title=Sophia%20Optimizer&summary=Check%20out%20Sophia%20Optimizer%20-%20a%20second-order%20clipped%20stochastic%20optimization%20algorithm%20that%20cuts%20model%20training%20cost%20in%20half!%20%23DeepLearning%20%23AI%20%23Optimization)

# 🌐 Agora: AI Researchers Advancing Humanity
Sophia is backed by Agora, a community of AI researchers dedicated to advancing humanity and solving some of the world's biggest problems. Join us in making a difference!

[Join our discord and write your mark on the history books](https://discord.gg/qUtxnK2NMf)



# Usage

Download with pip ```pip install Sophia-Optimizer``` 

or 

download with git ```git clone https://github.com/kyegomez/Sophia.git ```

```python 
import torch 
from torch import nn
from Sophia import SophiaG # or Sophia2 for more speed and inplace ops rather than updating params 


#or decoupled
#from Sophia import DecoupledSophia #plug in and play, personalize, improved maintainability


#define your model

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#init model loss function and input data
model = MyModel()
loss_function = nn.CrossEntropy()
input_data = ... #input data

#init the optimizer
optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)

#decoupled
#optimizer = DecoupledSophia(model.parameters(), lr=1e-3, betas=(0.9, 0.999), rho=0.04, weight_decay=1e-1, }   estimator="Hutchinson")

#training loop
for epoch in range(epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
```

## Training:
To run training use git clone method and navigate to experiments folder and if not then do the following:

```python
from Sophia import DecoupledSophia, trainer


#train model
trainer.train()


#eval the model
eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")

```


# Algorithmic pseudocode:

```

Initialize parameters: θ1, learning rate {ηt}, hyperparameters λ, β1, β2, ϵ, and estimator choice Estimator ∈ {Hutchinson, Gauss-Newton-Bartlett}
Set m0 = 0, v0 = 0, h1−k = 0
For t = 1 to T do
    Compute minibatch loss Lt(θt)
    Compute gt = ∇Lt(θt)
    mt = β1mt−1 + (1 − β1)gt
    If t mod k = 1 then
        Compute hˆt = Estimator(θt)
        ht = β2ht−k + (1 − β2)hˆt
    Else
        ht = ht−1
    θt = θt − ηtλθt (weight decay)
    θt+1 = θt − ηt · clip(mt/ max{ht, ϵ}, ρ)

```

# Pytorch Implementation

```python

import torch 

class Sophia(torch.optim.Optimizer):
    def __init__(self, model, input_data, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, k=10, estimator="Hutchinson", rho=1):
        self.model = model
        self.input_data = input_data
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, estimator=estimator, rho=rho)
        super(Sophia, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")
                
                state = self.state[p]

                #state init
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['h'] = torch.zeros_like(p.data)

                m, h = state['m'], state['h']
                beta1, beta2 = group['betas']
                state['step'] += 1


                if group['weight_decay'] != 0:
                    grad = grad.add(group["weight_decau"], p.data)

                #update biased first moment estimate
                m.mul_(beta1).add_(1 - beta1, grad)

                #update hessian estimate
                if state['step'] % group['k'] == 1:
                    if group['estimator'] == "Hutchinson":
                        hessian_estimate = self.hutchinson(p, grad)
                    elif group['estimator'] == "Gauss-Newton-Bartlett":
                        hessian_estimate = self.gauss_newton_bartlett(p, grad)
                    else:
                        raise ValueError("Invalid estimator choice")
                    h.mul_(beta2).add_(1 - beta2, hessian_estimate)

                #update params
                p.data.add_(-group['lr'] * group['weight_decay'], p.data)
                p.data.addcdiv_(-group['lr'], m, h.add(group['eps']).clamp(max=group['rho']))

        return loss
    
    def hutchinson(self, p, grad):
        u = torch.randn_like(grad)
        hessian_vector_product = torch.autograd.grad(grad.dot(u), p, retain_graph=True)[0]
        return u * hessian_vector_product
    
    def gauss_newton_bartlett(self, p, grad):
        B = len(self.input_data)
        logits = [self.model(xb) for xb in self.input_data]
        y_hats = [torch.softmax(logit, dim=0) for logit in logits]
        g_hat = torch.autograd.grad(sum([self.loss_function(logit, y_hat) for logit, y_hat in zip(logits, y_hats)]) / B, p, retain_graph=True)[0]
        return B * g_hat * g_hat
    
        
```

# Roadmap
The following roadmap outlines the future development plans for the Sophia optimizer. The roadmap is divided into three stages: short-term, mid-term, and long-term goals.


## Short-term Goals

Ready to train plug in and play file with your own model or Andromeda

Performance improvements: Investigate and implement potential performance improvements to further reduce training time and computational resources -> Decoupled Sophia + heavy metric logging + Implement in Triton and or Jax?

Additional Hessian estimators: Research and implement other Hessian estimators to provide more options for users.

Hyperparameter tuning: Develop a set of recommended hyperparameters for various use cases and model architectures.

# Mid-term Goals
Integration with Andromeda model: Train the Andromeda model using the Sophia optimizer and compare its performance with other optimizers.

Sophia optimizer variants: Explore and develop variants of the Sophia optimizer tailored for specific tasks, such as computer vision, multi-modality AI, and natural language processing, and reinforcement learning.

Distributed training: Implement support for distributed training to enable users to train large-scale models using Sophia across multiple devices and nodes.

Automatic hyperparameter tuning: Develop an automatic hyperparameter tuning module to help users find the best hyperparameters for their specific use case.

# Long-term Goals
Training multiple models in parallel: Develop a framework for training multiple models concurrently with different optimizers, allowing users to test and compare the performance of various optimizers, including Sophia, on their specific tasks.

Sophia optimizer for other domains: Adapt the Sophia optimizer for other domains, such as optimization in reinforcement learning, Bayesian optimization, and evolutionary algorithms.


By following this roadmap, we aim to make the Sophia optimizer a powerful and versatile tool for the deep learning community, enabling users to train their models more efficiently and effectively.


# Epoch 1 - Decoupled Sophia

The DecoupledSophia optimizer offers several benefits that can help accelerate training speed and improve the flexibility of the optimization process:

Modularity: Decoupling the Hessian estimation from the main optimizer allows users to easily plug in different Hessian estimators without modifying the core optimizer code. This modularity makes it easier to experiment with various Hessian estimation techniques and find the best one for a specific task.

Ease of experimentation: With a decoupled architecture, researchers and practitioners can develop and test new Hessian estimators independently from the optimizer. This can lead to faster innovation and the discovery of more efficient Hessian estimation methods, which can further accelerate training speed.

Customization: Users can create custom Hessian estimators tailored to their specific use cases or model architectures. This customization can potentially lead to better optimization performance and faster training times.

Improved maintainability: Separating the Hessian estimation from the optimizer makes the codebase easier to maintain and understand. This can lead to faster bug fixes and improvements in the optimizer's performance.

By offering these benefits, DecoupledSophia can help users accelerate training speed and improve the overall optimization process. The modular design allows for easy experimentation with different Hessian estimators, which can lead to the discovery of more efficient techniques and ultimately faster training times.


# Epoch - 2
Implement the training strategy as closely as possible with transformers library!

# Load and preprocess the OpenWebText dataset
class CFG:
    SEQ_LEN: int = 1024
    NUM_CPU: int = multiprocessing.cpu_count()
    TOKENIZER: str = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(CFG.TOKENIZER)
dataset = load_dataset("openwebtext")

def tokenize_function(example):
    return tokenizer(example["text"] + tokenizer.eos_token)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=CFG.NUM_CPU,
    remove_columns=["text"],
)

block_size = CFG.SEQ_LEN

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

train_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=CFG.NUM_CPU,
)

# Initialize the GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2", n_ctx=1024)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# Choose a Hessian estimator
hessian_estimator = HutchinsonEstimator()

# Initialize the DecoupledSophia optimizer
optimizer = DecoupledSophia(model.parameters(), hessian_estimator, lr=1e-3)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=480,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    gradient_accumulation_steps=1,
    gradient_clipping=1.0,
    learning_rate_scheduler_type="cosine",
    warmup_steps=2000,
    report_to="none",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=train_dataset,
    optimizers=(optimizer, None),
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")