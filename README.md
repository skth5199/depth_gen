# Three Depth estimation end to end approaches focusing on complex situations while reducing computational complexity

Three monocular depth estimation approaches were developed:
1. Pix2pix model
2. U-net with DenseNet encoder
3. U-net with MobileNetv2 encoder

Datasets used for training and testing:
1. KITTI
2. NYU

The architectures of these approaches is illustrated in the dissertation document.

A new metric was engineered for the fair and more apposite comparison of the approaches, which factored in the computational complexity of the approaches. It consists of two components.
The computational complexity (CC) is calculated as follows:
<p align="center">
![image](https://user-images.githubusercontent.com/26760537/164729416-f2be262a-6c2b-4f4e-8f86-bd3099516386.png)
  </p>

where n is the total number of approaches being considered

The final value of the metric is calculated as follows:
![image](https://user-images.githubusercontent.com/26760537/164729478-e1c18c58-4104-4432-a0ed-c8c9078e3b3c.png)

It was determined that the Pix2pix model performed the best with an SSIM and Engineered metric scores of over 0.95. The testing process involved both, indoor and 
outdoor scenes. The U-nets with Densenet and MobileNetv2 encoders scored reltively less on these metrics. However, the generated depth maps were further evaluated and  explained through inspection to even out the differences caused by various factors such as colour encodings, between the ground truth maps and the depth maps generated by the approaches. Here, it was determined that the U-net with MobileNetv2 also generated accurate depth maps. Its score was low solely due to the aforementioned differences. 
Finally, a Graphical user interface was developed to facilitate the hybridised and appropriate use of the engineered models.
