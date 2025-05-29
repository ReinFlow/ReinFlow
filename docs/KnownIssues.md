



## Known Issues and Fixes to Common Bugs


### ``ValueError: XML Error: top-level default class 'main' cannot be renamed''
* This indicates your dm_control package is incompatible with mujoco's version. Try `pip install dm_control==1.0.16 mujoco==3.1.6`. 


### IsaacGym simulation for furniture-bench
* IsaacGym simulation can become unstable at times and lead to NaN observations in Furniture-Bench. The current env wrapper does not handle NaN observations.