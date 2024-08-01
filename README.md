# Impact of Loss Functions and Model Feature Selections on CNN-Based Image Style Transfer Similarity Metrics
Hannah Pavlovich and Richard Lepkowicz

[The final report for this project can be accessed here
]([url](https://github.com/hannahpav/CNN-Style-Transfer-Similarity-Scores/blob/main/Final-Paper-CNN-Style-Transfer.pdf))

## Abstract
This study investigates the effectiveness of neural style transfer techniques, focusing on enhancing the preservation of artistic styles and content, particularly in architectural images. We begin by replicating Gatys et al.'s method using the VGG-19 model and then explore various extensions to improve style transfer from architectural styles to cityscapes and portraits. Our modifications include testing alternative loss functions—such as perceptual, total variation, and Wasserstein losses—to capture nuanced differences in style and content preservation. We also adjust CNN layer selections to assess how varying levels of feature abstraction affect transfer results to evaluate their impact on image quality.

Our results reveal that while the VGG-19 model provides a balanced integration of texture and content, the Perceptual Loss Method, though numerically promising, often produces simplistic overlays rather than effective style integration, especially with architectural images. The study concludes that high-level feature models generally offer a good balance but may struggle with complex imagery. Current evaluation metrics, such as cosine similarity, improve style-to-content ratios but may not fully capture qualitative aspects of style transfer. Further refinement in methods and metrics is needed to enhance style transfer for complex and varied imagery.

## Files
The style transfer file 
