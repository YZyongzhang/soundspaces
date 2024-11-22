# soundspaces
## install
1. follow the https://github.com/facebookresearch/sound-spaces step by step to install
2. Edit habitat/tasks/rearrange/rearrange_sim.py file and remove the 36th line where FetchRobot is imported.
3. when you install habitat-sim , may have some problems ,you can follow steps under to sovle it
    1. use `pip install -r environment.txt`
    ```
    after , you can install habitat-sim dependence and other that the project need
    ```
4. if you want to run this project , you should to install `sam` (sagment anything)
    1. you can follow the https://github.com/facebookresearch/segment-anything 
    to install it , and do not forget download the chackpoint.
5. finally , you finnash all the pro-task , and the most import things is to edit the haibit-sim package and more information you can see the github issue about the error `sensor key error ` ,you can download the `simulator.py` in my github `https://github.com/YZyongzhang/soundspaces.git`