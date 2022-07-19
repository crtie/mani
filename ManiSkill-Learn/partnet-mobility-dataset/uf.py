from urdfpy import URDF
robot = URDF.load('A2_single.urdf')
# for link in robot.links:
#     print(link.name)
# print(URDF.link_fk(robot))
robot.show()