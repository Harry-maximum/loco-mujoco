from .atlas import Atlas
from .talos import Talos
from .unitreeH1 import UnitreeH1
from .unitreeG1 import UnitreeG1
from .gr1t1 import GR1T1
from .humanoids import HumanoidTorque, HumanoidMuscle, HumanoidTorque4Ages, HumanoidMuscle4Ages


# register environments in mushroom
Atlas.register()
Talos.register()
UnitreeH1.register()
UnitreeG1.register()
GR1T1.register()
HumanoidTorque.register()
HumanoidMuscle.register()
HumanoidTorque4Ages.register()
HumanoidMuscle4Ages.register()


from gymnasium import register

# register gymnasium wrapper environment
register("LocoMujoco",
         entry_point="loco_mujoco.environments.gymnasium:GymnasiumWrapper"
         )
