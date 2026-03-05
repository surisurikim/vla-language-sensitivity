"""
libero_runner.py

OpenVLA-OFT + LIBERO 에피소드 실행 로직.
openvla-oft의 run_libero_eval.py 인프라를 그대로 사용하되,
instruction을 실험에 맞게 교체할 수 있도록 thin wrapper로 구성한다.

핵심 아이디어:
  openvla-oft의 run_episode(cfg, env, task_description, ...) 에서
  task_description만 우리가 원하는 instruction variant로 교체 → 나머지 코드 재사용
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from libero.libero import benchmark

# openvla-oft 경로 추가
OPENVLA_OFT_PATH = os.path.expanduser("~/openvla-oft")
sys.path.insert(0, OPENVLA_OFT_PATH)

from experiments.robot.libero.libero_utils import get_libero_env, save_rollout_video
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TASK_MAX_STEPS,
    run_episode,
    prepare_observation,
    process_action,
)


# LIBERO 태스크 스위트 → checkpoint 매핑
TASK_SUITE_TO_CHECKPOINT = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object":  "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal":    "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10":      "moojink/openvla-7b-oft-finetuned-libero-10",
}


def make_libero_cfg(task_suite_name: str, seed: int = 7) -> GenerateConfig:
    """
    주어진 태스크 스위트에 대한 기본 GenerateConfig를 생성한다.

    Args:
        task_suite_name: "libero_spatial", "libero_object", "libero_goal", "libero_10" 중 하나
        seed: 랜덤 시드

    Returns:
        GenerateConfig 인스턴스 (openvla-oft 기본값 유지)
    """
    checkpoint = TASK_SUITE_TO_CHECKPOINT[task_suite_name]
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name=task_suite_name,
        seed=seed,
        # 기본 설정 유지 (L1 regression, proprio 포함)
        use_l1_regression=True,
        use_proprio=True,
        num_images_in_input=2,  # front + wrist 카메라
    )
    return cfg


def load_libero_model(cfg: GenerateConfig):
    """
    openvla-oft의 get_model()로 모델 전체(vla + action_head + proprio_projector)를 로딩한다.
    모델 weights는 HuggingFace Hub에서 자동으로 다운로드된다.

    Returns:
        tuple: (model, processor, action_head, proprio_projector, resize_size)
    """
    print(f"[LIBERO] 모델 로딩: {cfg.pretrained_checkpoint}")
    model = get_model(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)

    print("[LIBERO] 모델 로딩 완료.")
    return model, processor, action_head, proprio_projector, resize_size


def run_libero_episode(
    cfg: GenerateConfig,
    env,
    instruction: str,
    model,
    processor,
    action_head,
    proprio_projector,
    resize_size: int,
    save_video: bool = False,
    video_path: str = None,
) -> dict:
    """
    LIBERO 환경에서 단일 에피소드를 실행한다.
    openvla-oft의 run_episode를 그대로 호출하되, instruction만 교체한다.

    Args:
        cfg: GenerateConfig 인스턴스
        env: LIBERO 환경 인스턴스
        instruction: 사용할 언어 명령 (기본 task_description 대신 이것을 사용)
        model, processor, action_head, proprio_projector: 모델 구성 요소
        resize_size: 이미지 리사이즈 크기
        save_video: 영상 저장 여부
        video_path: 영상 저장 경로

    Returns:
        dict:
            - success (bool): 태스크 성공 여부
            - images (list): 에피소드 이미지 리스트
    """
    # openvla-oft의 run_episode에 instruction을 직접 주입
    success, replay_images = run_episode(
        cfg=cfg,
        env=env,
        task_description=instruction,  # ← 여기서 instruction 교체
        model=model,
        resize_size=resize_size,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
    )

    if save_video and video_path is not None and replay_images:
        save_rollout_video(replay_images, video_path)

    return {
        "success": success,
        "images": replay_images,
    }


def get_libero_tasks(task_suite_name: str):
    """
    주어진 태스크 스위트의 모든 태스크와 기본 instruction을 반환한다.

    Returns:
        list of dict: [{"task": task_obj, "default_instruction": str}, ...]
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    tasks = []
    for task in task_suite.tasks:
        # 임시 환경 생성으로 기본 instruction 확인
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        env.close()
        tasks.append({
            "task": task,
            "default_instruction": task_description,
        })
    return tasks
