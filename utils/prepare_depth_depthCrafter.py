import DepthCrafter.run as DC
from fire import Fire
def main(
    idx: int,
):
    DC.main(target_fps=2, video_folder=f"/data2/videos/youtube/video{idx}", save_folder=f"/data2/videos/depth/depth{idx}", save_npz=True)

if __name__ == "__main__":
    Fire(main)