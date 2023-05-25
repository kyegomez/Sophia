import torch
from torch.optim import Optimizer
from abc import ABC, abstractmethod


class HessianEstimator(ABC):
    @abstractmethod
    def estimate(self, p, grad):
        pass


class HutchinsonEstimator(HessianEstimator):
    def estimate(self, p, grad):
        u = torch.randn_like(grad)
        grad_dot_u = torch.sum(grad * u)
        hessian_vector_product = torch.autograd.grad(grad_dot_u, p, retain_graph=True)[0]
        return u * hessian_vector_product
    

class GaussNewtonBartlettEstimator(HessianEstimator):
    def __init__(self, model, input_data, loss_function):
        self.model = model
        self.input_data = input_data
        self.loss_function = loss_function
    
    def estimate(self, p, grad):
        B = len(self.input_data)
        logits = [self.model(xb) for xb in self.input_data]
        y_hats = [torch.softmax(logit, dim=0) for logit in logits]
        g_hat = torch.autograd.grad(sum([self.loss_function(logit, y_hat) for logit, y_hat in zip(logits, y_hats)]) / B, p, retain_graph=True)[0]
        return B * g_hat * g_hat
    

class DecoupledSophia(Optimizer):
    def __init__(self, params, hessian_estimator, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, k=10, rho=1):
        self.hessian_estimator = hessian_estimator
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, rho=rho)
        super(DecoupledSophia, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            if closure is not None:
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('DecoupledSophia does not support sparse gradients')
            
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
                grad = grad.add(group['weight_decay'], p.data)


            #update biased first moment estimate
            m.mul_(beta1).add_(1 - beta1, grad)

            #update hessian estomate
            if state['step'] % group['k'] == 1:
                hessian_estimator = self.hessian_estimator.estimate(p, grad)
                h.mul_(beta2).add_(1 - beta2, hessian_estimator)

            #update params
            p.data.add_(-group['lr'] * group['weight_decay'], p.data)
            p.data.addcdiv_(-group['lr'], m, h.add(group['eps']).clamp(max=group['rho']))

        return loss

    
    
