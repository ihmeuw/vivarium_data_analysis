import numpy as np
import scipy.stats as scs

class MixtureContinuous():
    """
    Class to encode a mixture distribution, modeled after the distributions in scipy.stats.
    So far this is just a stub - nothing has been tested.

    Ideally, this class should be converted to a subclass of scipy.stats.rv_continuous.
    We should also have a separate MixtureDiscrete subclass of scipy.stats.rv_discrete.
    
    I'm not sure what would be the best way to handle a general mixture distribution
    with both a continuous part and discrete part - perhaps  that should be its own
    class, e.g. MixedDistribution. Hmm, note that an arbitrary 1-dimensional distribution
    is a mixture of 3 distributions - its discrete, absolutely continuous, and
    singular continuous parts - so a "mixed distribution" class could in theory
    model any distribution when combined with other classes.
    """
    def __init__(self, components, weights=None):
        n = len(components)
        if weights is None:
            weights = np.ones(n) / n
        if len(weights) != n:
            raise ValueError("must have same number of weights as components")
        self.components = components
        self.weights = np.array(weights)
        
    def pdf(self, x):
        return sum(w*c.pdf(x) for w,c in zip(self.weights, self.components))
    
    def cdf(self, x):
        return sum(w*c.cdf(x) for w,c in zip(self.weights, self.components))
    
    def mean(self):
        return sum(w*c.mean() for w,c in zip(self.weights, self.components))
    
    def component_means(self):
        return np.array([c.mean() for c in self.components])
    
    def component_vars(self):
        return np.array([c.var() for c in self.components])
    
    def var(self):
        mean_of_vars = self.weights.dot(self.component_vars())
        var_of_means = self.weights.dot((self.component_means() - self.mean())**2)
        return mean_of_vars + var_of_means
    
    def rvs(self, size=1):
        rvs = [c.rvs() for c in np.random.choice(self.components, p=self.weights, size=size).flatten()]
        return rvs[0] if size == 1 else np.array(rvs).reshape(size)
    