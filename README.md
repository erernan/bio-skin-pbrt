Rendering and Training Code of "A Biophysical-based Skin Model for Heterogeneous Volume Rendering"
===============

### This repository holds the code of our paper "A Biophysical-based Skin Model for Heterogeneous Volume Rendering", which will be published on Computational Visual Media ([CVMJ](https://link.springer.com/journal/41095)) in 2024. 
### [paper link](https://drive.google.com/file/d/1pQV_Qx7x39Ob8w0mm5myReWi4JD5540N/view?usp=sharing)

| Rendering results under different env maps |
|:-----------:|
|<img src="paper/env_result_new.png" width="500px">|

Building the code
--------------
### Rendering
The rendering code is based on [PBRT-V3](https://github.com/mmp/pbrt-v3). Please follow their instructions to build PBRT. After building, you can run the following command to render the test scene (a cube with the skin texture). 
```bash
$ pbrt -nthreads 64 test_scene/cube/cube_tc.pbrt
```
If rendering fails, please check the relative paths in the .pbrt file. Our code is not that well organized. There is some redundant code that has not been cleaned up. I may re-organize the code when I am available.

### Training the network
In the `bio_network` folder, I provide the training code of the neural network described in the paper. Unfortunately, our training data is lost. Therefore, if you need to train the network, please generate training data according to the description in the paper. This is not difficult. All training data can be generated using only the provided cube scene. All you need to do is change various parameters of the participate medium and rendering.


Example scenes
--------------
Due to copyright policy, I cannot provide the 3D models in our paper. But I still provide a simple cube scene in the `test_scene` folder to test our method. The test scene includes the interior and exterior geometry, the dermis and epidermis textures, and the .pbrt files. I also provide my rendering results (under relatively high spp) in the `cube/results` folder. The two geometries are obtained by scaling along the normal with Blender. Please refer to the paper for detailed description. Bio-textures are generated through the network. Therefore, for new albedo textures, you have to train the above mentioned neural network.

Contact
--------------
If you have any questions, please email me: wqnina1995@gmail.com.

