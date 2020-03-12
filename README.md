# magnet-slosh
MultiSLOSH using Magnet Loss for Metric Learning

This code uses Magnet Loss, a method for metric learning, to cluster images of frequency power spectra. Magnet Loss is described in the following paper: [Metric Learning With Adaptive Density Discrimination](https://research.fb.com/wp-content/uploads/2016/05/metric-learning-with-adaptive-density-discrimination.pdf). The implementation in this repo borrows heavily from the [MagnetLoss-PyTorch](https://github.com/vithursant/MagnetLoss-PyTorch) repo.

To run, input the following in the terminal (add -h to see a list of input arguments):

**python magnet_loss_slosh.py**  

The data for this experiment is stored in /data/ is the kepler q9 stellar variability catalog curated by [TASOC](https://tasoc.dk/). The labels are as follows:  
0. APERIODIC
1. CONSTANT
2. CONTACT_ROT
3. DSCT_BCEP
4. ECLIPSE
5. GDOR_SPB
6. RRLYR_CEPHEID
7. SOLARLIKE  
  
By default this repo is set up to load and save data to their corresponding folders.  
/saved_models/ contains the trained models.  
/results/ contains images of the t-SNE-mapped projections.   
/embed/ contains saved Numpy arrays of the training/validation low-dimensional embeddings.  

This code was tested using Pytorch 1.1.0.

![Sample Embedding](/sample/sample.png)
Format: ![Alt Text](url)

