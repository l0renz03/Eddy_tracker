import numpy as np

def RejectPass(data: np.ndarray, ui: bool = False, filename: str="", threshold: float=2.) -> bool:
    try:
        bad = np.nanmax(data) > threshold or np.nanmin(data) < -threshold
    except ValueError as e:
        if ui: print('\033[93m'+str(e)+'\033[0m')
        bad = True
    if bad and ui:
        print('\033[93m'+f"Bad Pass Found:{filename}"+'\033[0m')
    return bad