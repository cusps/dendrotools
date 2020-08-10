from ij import IJ
from ij.plugin.frame import RoiManager
from ij.gui import WaitForUserDialog, Toolbar


path = getArgument()
IJ.open(path)
data = {}

imp = IJ.getImage()
IJ.setTool(Toolbar.LINE)
WaitForUserDialog("Select the area,then click OK.").show()
imp = IJ.getImage()
roi = imp.getRoi()
if roi.getPoints().npoints == 2:
	with open("I:\Research\dendrotools\dendrotools\imagej_scripts\data.txt", "w") as f:
		f.write('{{"x1":{},\n'.format(roi.getPoints().xpoints[0]))
		f.write('"y1":{},\n'.format(roi.getPoints().ypoints[0]))
		f.write('"x2":{},\n'.format(roi.getPoints().xpoints[1]))
		f.write('"y2":{}}}\n'.format(roi.getPoints().ypoints[1]))
