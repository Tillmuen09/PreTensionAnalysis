# PreTensionAnalysis
## How to use
Start the PreTension Analysis Software by running the Pfostengui.py file with your python interpreter.  
If necessary, install required packages.  
The following GUI should apper:  
![This is an image](https://github.com/Tillmuen09/PreTensionAnalysis/blob/main/Screenshot.png)
By clicking browse, upload two different images of chambers, from which you want to determine a change of post distance.  
Using to slider below each image, a main analysis axis (white line) can be set. The software will search for the minimal distances between bottom and top post within a certain range around the main axis. This range is controlled by the "Search range [px]" input field in the top left.  
In the top center input field, insert the pixel size of the image in um. In the top right, the spring constant of the posts has to be set.  
After pressing the "Analyse" Button, the difference between both images will displayed as "Deflection". Additionally, the force difference aka PreTension is calculated using the spring constant and displayed as "Force".

## Further information
Available uppon reasonable request
