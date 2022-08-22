# N-Body simulator

Contains both a sequential and a parallel CUDA implementation of the Barnes-Hut algorithm. CUDA implementation is based on the implementation by James Sandham( https://bitbucket.org/jsandham/nbodycuda/src/master/) and the paper "An Efficient CUDA Implementation of the Tree-Based Barnes Hut N-Body Algorithm" by Martin Burtscher and Keshav Pingali.

The project can be configured by changing the values of the parameters in the 'Params' structure(h/Params.h). You can run the program with command like arguments like '-num' and then the number of bodies you want to run the simulation with.

The project also contains a visualizer that displays the bodies and the quad-tree. For running the program with visualization either set the value of Params.visualize or run the program with command line argument '-v'. For displaying the tree either set value of Params.display_tree or run program with '-tree'.


Simulation with 100k bodies:
![100k](https://github.com/veljkozz/NBodySim/blob/main/imgs/Barnes-Hut-100k-77mb.gif)


Simulation with quad-tree visualization
![quad-tree](https://github.com/veljkozz/NBodySim/blob/main/imgs/Barnes-Hut-visualize-treek.gif)



 
 
