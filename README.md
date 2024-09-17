# coralreflections
Deriving bottom reflectance and untangling the contribution from benthic endmembers in shallow-water reef systems

`coralreflections` uses physical models of the water column to parameterise spectra obtained by [NASA's COral Reef Airborne Observatory (CORAL) mission](https://science.nasa.gov/mission/coral/). Led by Dr Eric Hochberg, the mission obtained aerial hyperspectral imagery of thousands of kilometers of reef habitat around the world with the goal of producing high-resolution maps of benthic cover. 

Retrieving benthic cover requires retrieval and characterisation of seafloor reflectance, $R_{b(\lambda)}$. From [Maritorena et al. (1994)](https://aslopubs.onlinelibrary.wiley.com/doi/abs/10.4319/lo.1994.39.7.1689):

$$R(O^-, H)\_{(\lambda)} = \frac{bb_{(\lambda)}}{2 K_{(\lambda)}} + \left( R_{b(\lambda)} - \frac{bb_{(\lambda)}}{2 K_{(\lambda)}}\right) e^{-2K_{(\lambda)} H}$$

where $R(O^-, H)\_{(\lambda)}$ is the remote sensing retrieval from just below the water surface (having been corrected for interference in the atmosphere and at the air-water interface); $bb_{(\lambda)}$ is the backscatter coefficient; $K_{(\lambda)}$ is the attenuation coefficient; and $H$ is the depth.

In order to best characeterise $R_{b(\lambda)}$, spectra were fitted to $R(O^-, H)\_{(\lambda)}$ by varying $bb_{(\lambda)}$, $K_{(\lambda)}$, and $H$ in order to minimise some objective function. The potential contribution of $R_{b(\lambda)}$ to the retrieved spectra was modelled through combining spectra from an endmenber library containing thousands of in-situ spectra from relevant benthic types.

The parameter space of coefficients was constrained using domain expertise. Various objective functions were trialled, and custom functions created in order to emphasise the importance of different spectral regions. A range of methods to characterise the endmember contributions were tested.

