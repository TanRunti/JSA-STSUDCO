# STS_UDCO
This is the source code of the paper: "UAV-Empowered Dependency-Aware Computation Offloading in Device-Edge-Cloud Collaborative Computing: An Improved Actor-Critic DRL Approach". A brief introduction of this work is as follows:
> Unmanned aerial vehicle (UAV)-empowered mobile edge computing (MEC) has become a research hotspot, ad-dressing challenges posed by the pressure of cloud computing and the limited service scope of MEC. 
However, the constrained computing resources of UAVs and the data dependence on actual tasks hinder the implementation of efficient computational offloading (CO) in reality.
> In this context, a device-edge-cloud collaborative compu-ting model was first developed to provide complementary offloading services.
> It considers stochastic movement and channel obstacles, representing the dependency relationships as a directed acyclic graph.
> An optimization problem is formulated to jointly optimize system costs (i.e., delay and energy consumption) and UAV endurance while considering resource and task-dependent constraints.
> Second, a saturated training SAC-based UDCO algo-rithm (STS-UDCO) is designed. STS-UDCO learns the entropy and value of the CO policy through to efficiently approximate the optimal solution.
> An adaptive saturation training rule (ASTR) proposed in STS-UDCO can dy-namically controls the update frequency of the critic based on the current fitted state, aiming to enhance training stability.
> Finally, extensive experiments demonstrate that STS-UDCO achieves better convergence and stability while also reducing the system total cost and convergence speed by at least 11.83% and 39.10% compared to other advanced algorithms.

This work was submitted to Journal of Systems Architecture.
We note that a shorter version of this paper was accepted at the IEEE ICPADS conference in 2023. Click [here](https://doi.org/10.1109/ICPADS60453.2023.00312) for our conference paper online.

## Dependencies
* Python 3.9.x

* Pytorch 1.12.x

## File Structure
* `STS_UDCO.py` # The STS_UDCO algorithm
* `Environment.py` # The UAV-empowered D-E-C collaborative computing environment model
* `Observation_Normal.py` # The observation normalization mechanism
* `Usernum_record.py` # The user count recorder

## usage
To start our project files: run `STS_UDCO.py`
