import numpy as np
import theano
import theano.tensor as T
import math

# from MALA import RWM


# This only works with 1D data
class Variational_inference(object):

    def __init__(self,fn, mu_init, var_init, learning_rate, no_of_sample,no_of_epochs):

        self.f = fn
        self.mu_init = mu_init
        self.var_init = var_init
        self.learning_rate = learning_rate
        self.n = no_of_sample
        self.epoch = no_of_epochs


    def grad(self, mu, var):

        w = T.vector('w')
        x = T.scalar('x')
        log_density = -T.log(T.sqrt(2*math.pi*w[1])) - ((x-w[0])**2)/(2*w[1])
        log_density_grad = T.grad(log_density, w)

        sum =0
        for i in range(self.n):
            sample = mu+ var**.5* np.random.randn()
            sum =sum+np.log(self.f(sample))*log_density_grad.eval({w:[mu, var],x:sample})

        mc_avg = sum/self.n

        mu_grad = -mc_avg[0]
        var_grad = -(1/(2*var)) - mc_avg[1]

        return mu_grad,var_grad

    def get_param(self):

        mu = self.mu_init
        var = self.var_init


        mu_arr = []
        var_arr = []

        for i in range(self.epoch):
            mu -=self.learning_rate*self.grad(mu, var)[0]
            var -=self.learning_rate*self.grad(mu, var)[1]

            mu_arr.append(mu)
            var_arr.append(var)

        return mu_arr, var_arr



#This is after reparametrization of variance i.e. taking \sigma^2 = log(1+e^s)
#Also added the momentum support
class Variational_inference_new(object):

    """
    This estimates a density with N(mu, sigma^2) using variational inference
    """

    def __init__(self,fn, mu_init, var_init, learning_rate, no_of_sample,no_of_epochs, momentum=.9):

        self.f = fn
        self.mu_init = mu_init
        self.var_init = var_init
        self.learning_rate = learning_rate
        self.n = no_of_sample
        self.epoch = no_of_epochs
        self.momentum = momentum


    def grad(self, mu, s):

        w = T.vector('w')
        var = T.log(1 + T.exp(w[1]))

        x = T.scalar('x')


        log_density = -T.log(T.sqrt(2*math.pi*var)) - ((x-w[0])**2)/(2*var)
        log_density_grad = T.grad(log_density, w)

        sum =0
        for i in range(self.n):
            sample = mu+ (math.log(1+math.exp(s))**.5)* np.random.randn()
            sum =sum+np.log(self.f(sample))*log_density_grad.eval({w:[mu,s], x:sample})

        mc_avg = sum/self.n

        mu_grad = -mc_avg[0]
        s_grad = (-.5* math.exp(s)) /(math.log(1+math.exp(s))*(1+math.exp(s))) - mc_avg[1]

        return mu_grad,s_grad

    def get_param(self):

        mu = self.mu_init
        s = math.log(math.exp(self.var_init)-1)


        mu_arr = []
        var_arr = []

        mu_update = [0]
        s_update = [0]

        for i in range(self.epoch):

            update1 = self.momentum*mu_update[i] + (1-self.momentum)*self.grad(mu, s)[0]
            update2 = self.momentum * mu_update[i] + (1 - self.momentum) * self.grad(mu, s)[1]

            mu = mu - self.learning_rate * update1
            s = s - self.learning_rate * update2

            mu_update.append(update1)
            s_update.append(update2)

            # mu = mu - self.learning_rate*self.grad(mu, s)[0]
            # s = s - self.learning_rate*self.grad(mu, s)[1]

            mu_arr.append(mu)
            var_arr.append(math.log(1+math.exp(s)))

            #We want to print the loss after every 5 epochs
            if i%5 ==0:
                sum = 0
                for k in range(self.n):
                    sample = mu + (math.log(1 + math.exp(s)) ** .5) * np.random.randn()
                    sum = sum + np.log(self.f(sample))

                loss = sum/self.n
                print("Loss for epoch:{} is \t {:.4f}".format(i, loss))




        return mu_arr, var_arr



if __name__ == '__main__':

    fn1 = lambda x: np.exp(-x ** 2 / (1.1 + np.sin(x)))
    vi1 = Variational_inference_new(fn1, 1, .1, .01, 1000, 200)

    rwm = RWM(fn1,10000,0,.1)

    mu, var = vi1.get_param()












