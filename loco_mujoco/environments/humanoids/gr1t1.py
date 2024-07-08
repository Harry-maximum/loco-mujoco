from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.utils import check_validity_task_mode_dataset
from loco_mujoco.environments import ValidTaskConf


class GR1T1(BaseRobotHumanoid):
    """
    Mujoco simulation of the Gr1T1 robot. Optionally, the Gr1T1 can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run", "carry"],
                                     data_types=["real", "perfect"],
                                     non_combinable=[("carry", None, "perfect")])

    def __init__(self, disable_arms=True, disable_back_joint=True, hold_weight=False,
                 weight_mass=None, **kwargs):
        """
        Constructor.

        """

        if hold_weight:
            assert disable_arms is True, "If you want gr1t1 to carry a weight, please disable the arms. " \
                                         "They will be kept fixed."

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "gr1t1" / "gr1t1.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["floor"]),
                            ("r_foot", ["r_foot"]),
                            ("l_foot", ["l_foot"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint
        self._hold_weight = hold_weight
        self._weight_mass = weight_mass
        self._valid_weights = [0.1, 1.0, 5.0, 10.0]

        if disable_arms or hold_weight:
            xml_handle = mjcf.from_path(xml_path)

            if disable_arms or disable_back_joint:
                joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
                obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
                observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
                action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

                xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                          motors_to_remove, equ_constr_to_remove)
            if disable_arms and not hold_weight:
                xml_handle = self._reorient_arms(xml_handle)

            xml_handles = []
            if hold_weight and weight_mass is not None:
                color_red = np.array([1.0, 0.0, 0.0, 1.0])
                xml_handle = self._add_weight(xml_handle, weight_mass, color_red)
                xml_handles.append(xml_handle)
            elif hold_weight and weight_mass is None:
                for i, w in enumerate(self._valid_weights):
                    color = self._get_box_color(i)
                    current_xml_handle = deepcopy(xml_handle)
                    current_xml_handle = self._add_weight(current_xml_handle, w, color)
                    xml_handles.append(current_xml_handle)
            else:
                xml_handles.append(xml_handle)

        else:
            xml_handles = mjcf.from_path(xml_path)

        super().__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate([self._get_collision_force("floor", "r_foot")[:3],
                              self._get_collision_force("floor", "l_foot")[:3]])

        return grf

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 6

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["joint_left_shoulder_pitch", "joint_left_shoulder_roll", "joint_left_shoulder_yaw", "joint_left_elbow_pitch", "joint_left_wrist_yaw", "joint_left_wrist_pitch", "joint_left_wrist_roll",
                                 "joint_right_shoulder_pitch", "joint_right_shoulder_roll", "joint_right_shoulder_yaw", "joint_right_elbow_pitch", "joint_right_wrist_yaw", "joint_right_wrist_pitch", "joint_right_wrist_roll"]
            
            motors_to_remove += ["link_left_shoulder_pitch", "link_left_shoulder_roll", "link_left_shoulder_yaw", "link_left_elbow_pitch", "link_left_wrist_yaw", "link_left_wrist_pitch", "link_left_wrist_roll",
                                 "link_right_shoulder_pitch", "link_right_shoulder_roll", "link_right_shoulder_yaw", "link_right_elbow_pitch", "link_right_wrist_yaw", "link_right_wrist_pitch", "link_right_wrist_roll"]
            
        if self._disable_back_joint:
            joints_to_remove += ["joint_waist_pitch", "joint_waist_roll", "joint_waist_yaw"]
            motors_to_remove += ["link_waist_pitch", "link_waist_roll", "link_waist_yaw"]
            

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    def _has_fallen(self, obs, return_err_msg=False):
        """
        Checks if a model has fallen.

        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.
            Optionally an error message is returned.

        """

        pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])
        pelvis_y_condition = (obs[0] < -0.3) or (obs[0] > 0.1)
        pelvis_tilt_condition = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        pelvis_list_condition = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 8)) or (pelvis_euler[2] > (np.pi / 8))
        pelvis_condition = (pelvis_y_condition or pelvis_tilt_condition or
                            pelvis_list_condition or pelvis_rotation_condition)

        if return_err_msg:
            error_msg = ""
            if pelvis_y_condition:
                error_msg += "pelvis_y_condition violated.\n"
            elif pelvis_tilt_condition:
                error_msg += "pelvis_tilt_condition violated.\n"
            elif pelvis_list_condition:
                error_msg += "pelvis_list_condition violated.\n"
            elif pelvis_rotation_condition:
                error_msg += "pelvis_rotation_condition violated.\n"

            return pelvis_condition, error_msg
        else:

            return pelvis_condition

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
        task (str): Main task to solve. Either "walk", "run" or "carry". The latter is walking while carrying
                an unknown weight, which makes the task partially observable.
        dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.

        """
        check_validity_task_mode_dataset(GR1T1.__name__, task, None, dataset_type,
                                         *GR1T1.valid_task_confs.get_all())
        if dataset_type == "real":
            if task == "run":
                path = "datasets/data_generation/generated_data/gr1t1_run.npz"
            else:
                path = "datasets/data_generation/generated_data/gr1t1_walk.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "disable_back_joint" in kwargs.keys():
                assert kwargs["disable_back_joint"] is False
            if "hold_weight" in kwargs.keys():
                assert kwargs["hold_weight"] is False

            if task == "run":
                path = "datasets/humanoids/perfect/gr1t1_run/perfect_expert_dataset_det.npz"
            else:
                path = "datasets/humanoids/perfect/gr1t1_walk/perfect_expert_dataset_det.npz"

        return BaseRobotHumanoid.generate(GR1T1, path, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)

    @staticmethod
    def _add_weight(xml_handle, mass, color):
        """
        Adds a weight to the Mujoco XML handle. The weight will
        be hold in front of GR1T1. Therefore, the arms will be
        reoriented.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # find pelvis handle
        pelvis = xml_handle.find("body", "base")
        pelvis.add("body", name="weight")
        weight = xml_handle.find("body", "weight")
        weight.add("geom", type="box", size="0.1 0.18 0.1", pos="0.35 0 0.1", group="0", rgba=color, mass=mass)

        return xml_handle

    @staticmethod
    def _reorient_arms(xml_handle):
        """TODO:
        Reorients the elbow to not collide with the hip.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # modify the arm orientation
        # left_shoulder_pitch_link = xml_handle.find("body", "left_shoulder_pitch_link")
        # left_shouldejoint_left_shoulder_pitch_actuatorr_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        # right_elbow_link = xml_handle.find("body", "right_elbow_link")
        # right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        # right_shoulder_pitch_link = xml_handle.find("body", "right_shoulder_pitch_link")
        # right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        # left_elbow_link = xml_handle.find("body", "left_elbow_link")
        # left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return xml_handle

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """
        observation_spec = [
                            # ------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            ("q_joint_left_hip_pitch", "joint_left_hip_pitch", ObservationType.JOINT_POS),
                            ("q_joint_left_hip_roll", "joint_left_hip_roll", ObservationType.JOINT_POS),
                            ("q_joint_left_hip_yaw", "joint_left_hip_yaw", ObservationType.JOINT_POS),
                            ("q_joint_left_knee_pitch", "joint_left_knee_pitch", ObservationType.JOINT_POS),
                            ("q_joint_left_ankle_pitch", "joint_left_ankle_pitch", ObservationType.JOINT_POS),
                            ("q_joint_left_ankle_roll", "joint_left_ankle_roll", ObservationType.JOINT_POS),
                            ("q_joint_right_hip_pitch", "joint_right_hip_pitch", ObservationType.JOINT_POS),
                            ("q_joint_right_hip_roll", "joint_right_hip_roll", ObservationType.JOINT_POS),
                            ("q_joint_right_hip_yaw", "joint_right_hip_yaw", ObservationType.JOINT_POS),
                            ("q_joint_right_knee_pitch", "joint_right_knee_pitch", ObservationType.JOINT_POS),
                            ("q_joint_right_ankle_pitch", "joint_right_ankle_pitch", ObservationType.JOINT_POS),
                            ("q_joint_right_ankle_roll", "joint_right_ankle_roll", ObservationType.JOINT_POS),
                            ("q_joint_waist_pitch", "joint_waist_pitch", ObservationType.JOINT_POS),
                            ("q_joint_waist_roll", "joint_waist_roll", ObservationType.JOINT_POS),
                            ("q_joint_waist_yaw", "joint_waist_yaw", ObservationType.JOINT_POS),
                            ("q_joint_left_shoulder_pitch", "joint_left_shoulder_pitch", ObservationType.JOINT_POS),
                            ("q_joint_left_shoulder_roll", "joint_left_shoulder_roll", ObservationType.JOINT_POS),
                            ("q_joint_left_shoulder_yaw", "joint_left_shoulder_yaw", ObservationType.JOINT_POS),
                            ("q_joint_left_elbow_pitch", "joint_left_elbow_pitch", ObservationType.JOINT_POS),
                            ("q_joint_left_wrist_yaw", "joint_left_wrist_yaw", ObservationType.JOINT_POS),
                            ("q_joint_left_wrist_pitch", "joint_left_wrist_pitch", ObservationType.JOINT_POS),
                            ("q_joint_left_wrist_roll", "joint_left_wrist_roll", ObservationType.JOINT_POS),
                            ("q_joint_right_shoulder_pitch", "joint_right_shoulder_pitch", ObservationType.JOINT_POS),
                            ("q_joint_right_shoulder_roll", "joint_right_shoulder_roll", ObservationType.JOINT_POS),
                            ("q_joint_right_shoulder_yaw", "joint_right_shoulder_yaw", ObservationType.JOINT_POS),
                            ("q_joint_right_elbow_pitch", "joint_right_elbow_pitch", ObservationType.JOINT_POS),
                            ("q_joint_right_wrist_yaw", "joint_right_wrist_yaw", ObservationType.JOINT_POS),
                            ("q_joint_right_wrist_pitch", "joint_right_wrist_pitch", ObservationType.JOINT_POS),
                            ("q_joint_right_wrist_roll", "joint_right_wrist_roll", ObservationType.JOINT_POS),


                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_joint_left_hip_pitch", "joint_left_hip_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_left_hip_roll", "joint_left_hip_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_left_hip_yaw", "joint_left_hip_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_left_knee_pitch", "joint_left_knee_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_left_ankle_pitch", "joint_left_ankle_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_left_ankle_roll", "joint_left_ankle_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_right_hip_pitch", "joint_right_hip_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_right_hip_roll", "joint_right_hip_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_right_hip_yaw", "joint_right_hip_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_right_knee_pitch", "joint_right_knee_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_right_ankle_pitch", "joint_right_ankle_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_right_ankle_roll", "joint_right_ankle_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_waist_pitch", "joint_waist_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_waist_roll", "joint_waist_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_waist_yaw", "joint_waist_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_left_shoulder_pitch", "joint_left_shoulder_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_left_shoulder_roll", "joint_left_shoulder_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_left_shoulder_yaw", "joint_left_shoulder_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_left_elbow_pitch", "joint_left_elbow_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_left_wrist_yaw", "joint_left_wrist_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_left_wrist_pitch", "joint_left_wrist_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_left_wrist_roll", "joint_left_wrist_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_right_shoulder_pitch", "joint_right_shoulder_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_right_shoulder_roll", "joint_right_shoulder_roll", ObservationType.JOINT_VEL),
                            ("dq_joint_right_shoulder_yaw", "joint_right_shoulder_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_right_elbow_pitch", "joint_right_elbow_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_right_wrist_yaw", "joint_right_wrist_yaw", ObservationType.JOINT_VEL),
                            ("dq_joint_right_wrist_pitch", "joint_right_wrist_pitch", ObservationType.JOINT_VEL),
                            ("dq_joint_right_wrist_roll", "joint_right_wrist_roll", ObservationType.JOINT_VEL)]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.
            
            NUM_ACTIONS = 29

        """
       
        action_spec = ["link_left_hip_pitch","link_left_hip_roll","link_left_hip_yaw","link_left_knee_pitch","link_left_ankle_pitch","link_left_ankle_roll",
                       "link_right_hip_pitch","link_right_hip_roll","link_right_hip_yaw","link_right_knee_pitch","link_right_ankle_pitch","link_right_ankle_roll",
                       "link_waist_pitch","link_waist_roll","link_waist_yaw",
                       "link_left_shoulder_pitch","link_left_shoulder_roll","link_left_shoulder_yaw","link_left_elbow_pitch","link_left_wrist_yaw","link_left_wrist_pitch","link_left_wrist_roll",
                       "link_right_shoulder_pitch","link_right_shoulder_roll","link_right_shoulder_yaw","link_right_elbow_pitch","link_right_wrist_yaw","link_right_wrist_pitch","link_right_wrist_roll"
                       ]

        return action_spec
