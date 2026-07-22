import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from arm_assets_v02 import baxter_reduced

package_dirs = ["./"] 
robot = baxter_reduced("baxter_description/urdf/baxter_fixed.urdf", package_dirs)

print(robot.getPos())
viz = MeshcatVisualizer(robot.model, robot.collModel, robot.visualModel)
viz.initViewer(open=False)
viz.loadViewerModel()
q0 = pin.neutral(robot.model)
viz.displayVisuals(True)
while True:
    viz.display(q0)