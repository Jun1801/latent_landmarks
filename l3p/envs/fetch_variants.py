"""Custom Fetch variants from the paper (Section 5.3), built on gymnasium-robotics.

  * Box-Distractor-PickAndPlace: standard pick-and-place with a box distractor in
    the middle of the table that the arm must avoid.
  * Place-Inside-Box: place the object inside an open box; curriculum with 80%
    regular pick-and-place goals and 20% inside-the-box goals.

These are NOT shipped by gymnasium-robotics, so we generate modified MuJoCo XMLs
(the original Fetch scene + an added box body, with absolute include/mesh paths
so it compiles from anywhere) and subclass MujocoFetchEnv. This is a faithful
reproduction of the *tasks described in the paper*; the exact XML geometry is
our own (the paper's original assets are not public in a portable form).
"""

from __future__ import annotations

import os

import numpy as np
import gymnasium_robotics
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv

_GR_ASSETS = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets")
_FETCH = os.path.join(_GR_ASSETS, "fetch")
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(_OUT, exist_ok=True)

_INITIAL_QPOS = {
    "robot0:slide0": 0.405, "robot0:slide1": 0.48, "robot0:slide2": 0.0,
    "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
}


def _scene_xml(extra_bodies: str) -> str:
    """Full Fetch pick-and-place scene with absolute asset paths + extra bodies."""
    return f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <compiler angle="radian" coordinate="local" meshdir="{_GR_ASSETS}/stls/fetch" texturedir="{_GR_ASSETS}/textures"></compiler>
    <option timestep="0.002"><flag warmstart="enable"></flag></option>
    <include file="{_FETCH}/shared.xml"></include>
    <worldbody>
        <geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"></geom>
        <body name="floor0" pos="0.8 0.75 0">
            <site name="target0" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
        </body>
        <include file="{_FETCH}/robot.xml"></include>
        <body pos="1.3 0.75 0.2" name="table0">
            <geom size="0.25 0.35 0.2" type="box" mass="2000" material="table_mat"></geom>
        </body>
        <body name="object0" pos="0.025 0.025 0.025">
            <joint name="object0:joint" type="free" damping="0.01"></joint>
            <geom size="0.025 0.025 0.025" type="box" condim="3" name="object0" material="block_mat" mass="2"></geom>
            <site name="object0" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
        </body>
        {extra_bodies}
        <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
    </worldbody>
    <actuator>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="30000" name="robot0:l_gripper_finger_joint" user="1"></position>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="30000" name="robot0:r_gripper_finger_joint" user="1"></position>
    </actuator>
</mujoco>'''


# A solid distractor box sitting on the table, slightly off the spawn center.
_DISTRACTOR_BODY = '''<body name="distractor_box" pos="1.32 0.68 0.45">
            <geom name="distractor_box" size="0.035 0.035 0.05" type="box" material="puck_mat" mass="1000"></geom>
        </body>'''

# Geometry exposed for visualization (matches the XML bodies below).
DISTRACTOR_XY = (1.32, 0.68)      # distractor box center (x, y)
DISTRACTOR_HALF = 0.035           # half-extent in x/y
CONTAINER_XY = (1.30, 0.92)       # target-box center (x, y)
CONTAINER_HALF = 0.066            # half-extent of the container footprint

# An open-top box container on the table (four thin walls) for Place-Inside-Box.
_BOX_CENTER = np.array([1.30, 0.92, 0.44])
_CONTAINER_BODY = f'''<body name="container" pos="{_BOX_CENTER[0]} {_BOX_CENTER[1]} {_BOX_CENTER[2]}">
            <geom type="box" size="0.06 0.006 0.03" pos="0 0.06 0" material="table_mat" mass="500"></geom>
            <geom type="box" size="0.06 0.006 0.03" pos="0 -0.06 0" material="table_mat" mass="500"></geom>
            <geom type="box" size="0.006 0.06 0.03" pos="0.06 0 0" material="table_mat" mass="500"></geom>
            <geom type="box" size="0.006 0.06 0.03" pos="-0.06 0 0" material="table_mat" mass="500"></geom>
        </body>'''


def _write_xml(name: str, extra_bodies: str) -> str:
    path = os.path.join(_OUT, name)
    with open(path, "w") as f:
        f.write(_scene_xml(extra_bodies))
    return path


class BoxDistractorFetchEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        model_path = _write_xml("box_distractor.xml", _DISTRACTOR_BODY)
        MujocoFetchEnv.__init__(
            self, model_path=model_path, has_object=True, block_gripper=False,
            n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True,
            target_offset=0.0, obj_range=0.15, target_range=0.15,
            distance_threshold=0.05, initial_qpos=_INITIAL_QPOS,
            reward_type=reward_type, **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


class PlaceInsideBoxFetchEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, reward_type="sparse", inside_ratio=0.2, **kwargs):
        model_path = _write_xml("place_inside_box.xml", _CONTAINER_BODY)
        self.inside_ratio = inside_ratio
        MujocoFetchEnv.__init__(
            self, model_path=model_path, has_object=True, block_gripper=False,
            n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True,
            target_offset=0.0, obj_range=0.15, target_range=0.15,
            distance_threshold=0.05, initial_qpos=_INITIAL_QPOS,
            reward_type=reward_type, **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, inside_ratio=inside_ratio, **kwargs)

    def _sample_goal(self):
        # Curriculum: 20% of goals are inside the box, 80% regular pick-and-place.
        if self.np_random.uniform() < self.inside_ratio:
            goal = _BOX_CENTER.copy()
            goal[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
            goal[2] = self.height_offset          # rest inside the container on the table
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3)
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        return goal.copy()


def make_fetch_variant(canonical: str, cfg, seed: int):
    from gymnasium.wrappers import TimeLimit
    if canonical == "BoxDistractorPickAndPlace":
        env = BoxDistractorFetchEnv()
    elif canonical == "PlaceInsideBox":
        env = PlaceInsideBoxFetchEnv(inside_ratio=cfg.place_inside_box_ratio)
    else:
        raise ValueError(f"Unknown fetch variant: {canonical}")
    env = TimeLimit(env, max_episode_steps=max(cfg.max_episode_steps, cfg.test_episode_steps))
    env.reset(seed=seed)
    return env
