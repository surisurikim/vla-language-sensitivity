"""
simplerenv_runner.py

SimplerEnv에서 단일 에피소드를 실행하는 로직.
phase0_baseline.py와 Phase 1 instruction variation 스크립트에서 공통으로 사용한다.
"""

import os
import sys

import numpy as np

SIMPLER_ENV_PATH = os.path.expanduser("~/SimplerEnv")
sys.path.insert(0, SIMPLER_ENV_PATH)

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video


def run_episode(
    model,
    env_name: str,
    scene_name: str,
    instruction: str,
    robot: str,
    rgb_overlay_path: str,
    max_steps: int = 80,
    save_video: bool = False,
    video_path: str = None,
) -> dict:
    """
    SimplerEnv에서 단일 에피소드를 실행하고 결과를 반환한다.

    Args:
        model: OpenVLAInference 인스턴스
        env_name: SimplerEnv 환경 이름 (예: "GraspSingleOpenedCokeCanInScene-v0")
        scene_name: 씬 이름
        instruction: 언어 명령 (실험 변수)
        robot: 로봇 이름 (예: "google_robot_static")
        rgb_overlay_path: 실제 환경 이미지 overlay 경로
        max_steps: 에피소드 최대 스텝 수
        save_video: 영상 저장 여부
        video_path: 영상 저장 경로

    Returns:
        dict:
            - success (bool): 태스크 성공 여부
            - steps (int): 실제 실행된 스텝 수
            - images (list): 각 스텝의 RGB 이미지
            - predicted_actions (list): 각 스텝의 raw action (shape: 7,)
    """
    # SimplerEnv의 get_robot_control_mode는 policy_name을 받지만 실제로는 사용하지 않음
    control_mode = get_robot_control_mode(robot, "openvla")

    env = build_maniskill2_env(
        env_name,
        obs_mode="rgbd",
        robot=robot,
        sim_freq=513,
        control_mode=control_mode,
        control_freq=3,
        max_episode_steps=max_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        # overlay 파일이 없으면 None으로 대체 (렌더링은 계속 가능)
        rgb_overlay_path=rgb_overlay_path if os.path.exists(rgb_overlay_path) else None,
    )

    obs, _ = env.reset()
    model.reset(instruction)

    images = []
    predicted_actions = []
    success = False

    for step in range(max_steps):
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)

        raw_action, action = model.step(image, instruction)
        predicted_actions.append(raw_action)

        terminate_flag = bool(action["terminate_episode"][0] > 0)
        obs, reward, done, truncated, info = env.step(
            np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"],
            ])
        )

        # 환경 종료 or 모델이 terminate 신호를 보내면 루프 종료
        if done or truncated or terminate_flag:
            success = bool(info.get("success", False))
            break

    env.close()

    if save_video and video_path is not None:
        write_video(video_path, images, fps=5)

    return {
        "success": success,
        "steps": step + 1,
        "images": images,
        "predicted_actions": predicted_actions,
    }
