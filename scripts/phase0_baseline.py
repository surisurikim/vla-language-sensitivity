"""
phase0_baseline.py

Phase 0: SimplerEnv에서 OpenVLA 베이스라인 평가.

목적:
  - OpenVLA + SimplerEnv가 정상적으로 동작하는지 확인
  - 기본 instruction으로 태스크를 실행하고 성공률(baseline)을 기록
  - 이후 Phase 1 instruction variation 실험의 기준값으로 사용

사용법:
  conda activate vla-lang
  cd ~/vla-language-sensitivity
  python scripts/phase0_baseline.py

결과:
  results/phase0/ 에 성공률 JSON과 에피소드 영상(옵션) 저장
"""

import os
import sys
import json
import argparse
import numpy as np

# SimplerEnv 경로 추가
SIMPLER_ENV_PATH = os.path.expanduser("~/SimplerEnv")
sys.path.insert(0, SIMPLER_ENV_PATH)

# 프로젝트 루트 경로 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from policies.openvla_policy import OpenVLAInference
from envs.simplerenv_runner import run_episode


# ─────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────

# 평가할 태스크 목록
# SimplerEnv Google Robot 태스크 중 가장 안정적으로 테스트된 환경 선택
TASKS = [
    {
        "env_name": "GraspSingleOpenedCokeCanInScene-v0",
        "scene_name": "google_pick_coke_can_1_v4",
        "instruction": "pick coke can",
        "robot": "google_robot_static",
        "policy_setup": "google_robot",
        "rgb_overlay_path": os.path.join(
            SIMPLER_ENV_PATH,
            "ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png"
        ),
    },
    {
        "env_name": "MoveNearGoogleBakedTexInScene-v0",
        "scene_name": "google_move_near_real_eval_1",
        "instruction": "move the object near the target",
        "robot": "google_robot_static",
        "policy_setup": "google_robot",
        "rgb_overlay_path": os.path.join(
            SIMPLER_ENV_PATH,
            "ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png"
        ),
    },
]

# 에피소드 반복 횟수: 통계적 신뢰성을 위해 본 실험에서는 20 이상 권장
NUM_EPISODES = 5  # 빠른 확인용 기본값

# 결과 저장 경로
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES,
                        help="에피소드 반복 횟수")
    parser.add_argument("--save-video", action="store_true",
                        help="에피소드 영상 저장 여부")
    parser.add_argument("--model-path", type=str, default="openvla/openvla-7b",
                        help="OpenVLA 모델 경로 또는 HuggingFace ID")
    parser.add_argument("--task-idx", type=int, default=None,
                        help="특정 태스크만 실행 (0-indexed). 미입력 시 전체 실행")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tasks = TASKS if args.task_idx is None else [TASKS[args.task_idx]]

    # 모델은 전체 실험에서 한 번만 로딩 (매 태스크마다 재로딩 불필요)
    print("=" * 60)
    print(f"OpenVLA 모델 로딩: {args.model_path}")
    print("=" * 60)
    # policy_setup은 첫 번째 태스크 기준으로 설정 (같은 policy_setup 태스크끼리 실행 권장)
    model = OpenVLAInference(
        model_path=args.model_path,
        policy_setup=tasks[0]["policy_setup"],
        device="cuda:0",
    )

    all_results = {}

    for task in tasks:
        env_name = task["env_name"]
        instruction = task["instruction"]
        print(f"\n{'=' * 60}")
        print(f"태스크: {env_name}")
        print(f"Instruction: '{instruction}'")
        print(f"에피소드 수: {args.num_episodes}")
        print("=" * 60)

        successes = []
        task_dir = os.path.join(RESULTS_DIR, env_name)
        os.makedirs(task_dir, exist_ok=True)

        for ep in range(args.num_episodes):
            video_path = os.path.join(task_dir, f"ep{ep:03d}.mp4") if args.save_video else None

            result = run_episode(
                model=model,
                env_name=env_name,
                scene_name=task["scene_name"],
                instruction=instruction,
                robot=task["robot"],
                rgb_overlay_path=task["rgb_overlay_path"],
                save_video=args.save_video,
                video_path=video_path,
            )

            successes.append(result["success"])
            status = "성공" if result["success"] else "실패"
            print(f"  에피소드 {ep+1:2d}/{args.num_episodes}: {status} ({result['steps']} 스텝)")

            # 첫 번째 에피소드만 action 시각화 저장 (이후 에피소드는 생략)
            if ep == 0:
                viz_path = os.path.join(task_dir, "action_viz_ep0.png")
                model.visualize_epoch(result["predicted_actions"], result["images"], viz_path)

        success_rate = float(np.mean(successes))
        print(f"\n  >> 성공률: {success_rate:.1%} ({sum(successes)}/{args.num_episodes})")

        all_results[env_name] = {
            "instruction": instruction,
            "num_episodes": args.num_episodes,
            "successes": successes,
            "success_rate": success_rate,
        }

    # 결과를 JSON으로 저장
    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # 최종 요약 출력
    print(f"\n{'=' * 60}")
    print("PHASE 0 베이스라인 요약")
    print("=" * 60)
    for env_name, res in all_results.items():
        print(f"  {env_name:50s}  {res['success_rate']:.1%}")
    print(f"\n결과 저장 위치: {results_path}")


if __name__ == "__main__":
    main()
