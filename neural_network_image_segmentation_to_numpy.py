import fast
import numpy as np 
import matplotlib.pyplot as plt

#fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT) # Uncomment to show debug info
fast.downloadTestDataIfNotExists() # This will download the test data needed to run the example

streamer = fast.ImageFileStreamer.New()
streamer.setFilenameFormat(fast.Config.getTestDataPath() + 'US/JugularVein/US-2D_#.mhd')
streamer.enableLooping()

segmentationNetwork = fast.SegmentationNetwork.New()
segmentationNetwork.setInputConnection(streamer.getOutputPort())
segmentationNetwork.setInferenceEngine('OpenVINO')
segmentationNetwork.setScaleFactor(1/255)
segmentationNetwork.load(fast.Config.getTestDataPath() +
    'NeuralNetworkModels/jugular_vein_segmentation.xml') 

result = segmentationNetwork.getOutputPort()
segmentationNetwork.update()
res = np.asarray(result.getNextImage())

plt.imshow(res)
plt.show()