r"""Record videos and measure XY/yaw tracking for every command combination.

Accepts the same arguments as speed_eval.py, plus --video-dir:
    python third_prototype/eval_video.py \
        --model-path third_prototype/logs/RUN/model_4999.pt \
        --commands 0 0.4 0.8 --y-range 0 0 --yaw-commands -0.4 0 0.4 \
        --duration 30 --warmup 2 --tolerance 0.1 --repeats 1

Each deterministic rollout produces one MP4 and the same accuracy table as
speed_eval.py. Videos include warmup; metrics exclude warmup. Yaw defaults to
zero; --yaw-commands adds turning tests. The environment diagnostic snapshot
must include base_ang_vel_local. Results are saved as metrics.json unless
--json-out specifies another file. Video paths link each measurement to its
recording. Rendering includes the terminal pose before automatic reset.
Use --terrain flat for a flat-plane evaluation with the same checkpoint.
"""
from __future__ import annotations

if __package__:
    from . import speed_eval
else:
    import speed_eval


def main(argv=None):
    return speed_eval.main(argv, record_videos=True, description=__doc__)


if __name__ == "__main__":
    main()
