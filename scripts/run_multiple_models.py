from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from run_models.run_deeplabv3plus_all import main as deeplabRun
from run_models.run_unetpp_all import main as unetppRun
from run_models.run_segformer_all import main as segformerRun


#Make sure model config is set up first!

def main():
    #deeplabRun()
    unetppRun()
    segformerRun()



if __name__ == "__main__":
    main()