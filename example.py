import maintools as tools
import main_run

# Video location
file = 'videos/video_test_2.mp4'

# Running Marker-less | Get Hip and Knee 2D Joint Angles
main_run.run_markerless(video_path=file, side='r', save=True, show=True)