import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualize_utils import process_all_episodes_concat, process_video_episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific episode and create video(s).")
    parser.add_argument(
        "--episode_index",
        type=int,
        required=False,
        help="The index of the episode to process (required unless --concat_all is set)",
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the pickle file")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save output videos")
    parser.add_argument("--fps", type=int, default=5, help="Video frame rate (default: 30)")
    parser.add_argument(
        "--vis_keys",
        nargs="*",
        default=None,
        help="Which observation keys to visualize. If omitted, visualize all keys except 'state'.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Index to take when observation[key] has a leading frame/batch dimension (default: 0).",
    )
    parser.add_argument(
        "--concat_all",
        action="store_true",
        help="Process every episode in order and concatenate outputs into a single video per key with episode labels.",
    )

    args = parser.parse_args()

    if args.concat_all:
        process_all_episodes_concat(
            file_path=args.input_file,
            output_dir=args.output_dir,
            vis_keys=args.vis_keys,
            fps=args.fps,
            frame_index=args.frame_index,
        )
    else:
        if args.episode_index is None:
            parser.error("--episode_index is required unless --concat_all is set")

        process_video_episode(
            episode_index=args.episode_index,
            file_path=args.input_file,
            output_dir=args.output_dir,
            vis_keys=args.vis_keys,
            fps=args.fps,
            frame_index=args.frame_index,
        )
