# Normalized BSSRDF

An implementation of the Normalized BSSRDF via Python port of [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute).

Tricks applied in the implementation includes:

- Importance Sampling of Normalized BSSDRF via the analytical form of the inverse CDF(yes, analytical).

```python
from math import *
dmfp = 1.0 # diffuse mean free path
def sample_radius(u):
    q = 4. * (u - 1.)
    x = pow(-0.5 * q + sqrt(0.25 * q * q + 1.), 1 / 3) - pow(0.5 * q + sqrt(0.25 * q * q + 1.), 1 / 3)
    return -3. * log(x) * dmfp
```

- Multiple Importance Sampling(light && cosine)

The result looks like below:

![Normalized BSSRDF](https://github.com/LeonKang130/NormalizedBSSRDF/blob/main/result-teaser.png)
