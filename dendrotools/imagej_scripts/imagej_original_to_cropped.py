import os

from ij import IJ
from ij.plugin.frame import RoiManager
from ij.gui import WaitForUserDialog, Toolbar


#remove all the previous ROIS
path = getArgument()
IJ.open(path)
imp = IJ.getImage()
img_name = os.path.split(os.path.basename(path))[0]
rm = RoiManager.getInstance()
if not rm:
	rm = RoiManager()
rm.runCommand("reset")

#ask the user to define a selection and get the bounds of the selection
IJ.setTool(Toolbar.RECTANGLE)
WaitForUserDialog("Select the area,then click OK.").show()
roi = imp.getRoi()
imp.setRoi(roi)
imp.crop()
print(roi)

IJ.saveAsTiff(imp, os.path.join(os.path.dirname(path), "{}_cropped".format(img_name))) #path.replace(img_name, "{}_cropped".format(img_name)))

imp.close()
# IJ.log(imp)
