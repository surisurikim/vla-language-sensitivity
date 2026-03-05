"""
phase0_libero.py

Phase 0 (LIBERO): OpenVLA-OFT + LIBERO 베이스라인 평가.

목적:
  - OpenVLA-OFT + LIBERO가 정상적으로 동작하는지 확인
  - 기본 instruction(LIBERO 내장)으로 태스크를 실행하고 성공률(baseline)을 기록
  - 이후 Phase 1 instruction variation 실험의 기준값으로 사용

모델:
  - moojink/openvla-7b-oft-finetuned-libero-{spatial|object|goal|10}
  - 첫 실행 시 HuggingFace Hub에서 자동 다운로드 (~15GB)

사용법:
  conda activate vla-lang
  cd ~/vla-language-sensitivity

  # libero_spatial 태스크 스위트, 태스크당 3회
  python scripts/phase0_libero.py --task-suite libero_spatial --num-episodes 3

  # 특정 태스크 인덱스만 실행
  python scripts/phase0_libero.py --task-suite libero_spatial --task-idx 0 --num-episodes 3

결과:
  results/phase0_libero/{task_suite}/ 에 성공률 JSON 저장
"""

import os
import sys
import json
import argparse
import numpy as np

# 프로젝트 루트 경로 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# openvla-oft 경로 추가
OPENVLA_OFT_PATH = os.path.expanduser("~/openvla-oft")
sys.path.insert(0, OPENVLA_OFT_PATH)

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import set_seed_everywhere

from envs.libero_runner import (
    make_libero_cfg,
    load_libero_model,
    run_libero_episode,
    get_libero_tasks,
)

# 결과 저장 경로
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "results", "phase0_libero")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        help="평가할 LIBERO 태스크 스위트")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="태스크당 에피소드 반복 횟수 (본 실험에서는 20 이상 권장)")
    parser.add_argument("--task-idx", type=int, default=None,
                        help="특정 태스크만 실행 (0-indexed). 미입력 시 전체 실행")
    parser.add_argument("--save-video", action="store_true",
                        help="에피소드 영상 저장 여부")
    parser.add_argument("--seed", type=int, default=7,
                        help="랜덤 시드")
    args = parser.parse_args()

    set_seed_everywhere(args.seed)

    results_dir = os.path.join(RESULTS_BASE_DIR, args.task_suite)
    os.makedirs(results_dir, exist_ok=True)

    # GenerateConfig 생성 (openvla-oft 기본 설정)
    cfg = make_libero_cfg(args.task_suite, seed=args.seed)

    # 모델 로딩 (HuggingFace Hub에서 자동 다운로드)
    print("=" * 60)
    print(f"모델 로딩: {cfg.pretrained_checkpoint}")
    print("=" * 60)
    model, processor, action_head, proprio_projector, resize_size = load_libero_model(cfg)

    # 태스크 목록 가져오기
    tasks = get_libero_tasks(args.task_suite)
    if args.task_idx is not None:
        tasks = [tasks[args.task_idx]]

    all_results = {}

    for task_info in tasks:
        task = task_info["task"]
        default_instruction = task_info["default_instruction"]

        print(f"\n{'=' * 60}")
        print(f"태스크: {default_instruction}")
        print(f"에피소드 수: {args.num_episodes}")
        print("=" * 60)

        # LIBERO 환경 생성
        env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        task_dir = os.path.join(results_dir, default_instruction.replace(" ", "_")[:50])
        os.makedirs(task_dir, exist_ok=True)

        successes = []

        for ep in range(args.num_episodes):
            video_path = os.path.join(task_dir, f"ep{ep:03d}.mp4") if args.save_video else None

            result = run_libero_episode(
                cfg=cfg,
                env=env,
                instruction=default_instruction,  # Phase 0: 기본 instruction 그대로 사용
                model=model,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                resize_size=resize_size,
                save_video=args.save_video,
                video_path=video_path,
            )

            successes.append(result["success"])
            status = "성공" if result["success"] else "실패"
            print(f"  에피소드 {ep+1:2d}/{args.num_episodes}: {status}")

        env.close()

        success_rate = float(np.mean(successes))
        print(f"\n  >> 성공률: {success_rate:.1%} ({sum(successes)}/{args.num_episodes})")

        all_results[default_instruction] = {
            "num_episodes": args.num_episodes,
            "successes": successes,
            "success_rate": success_rate,
        }

    # 결과 저장
    results_path = os.path.join(results_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 최종 요약
    avg_success = np.mean([v["success_rate"] for v in all_results.values()])
    print(f"\n{'=' * 60}")
    print(f"PHASE 0 LIBERO 베이스라인 요약 ({args.task_suite})")
    print("=" * 60)
    for instr, res in all_results.items():
        print(f"  {instr[:55]:55s}  {res['success_rate']:.1%}")
    print(f"\n  평균 성공률: {avg_success:.1%}")
    print(f"결과 저장 위치: {results_path}")


if __name__ == "__main__":
    main()
