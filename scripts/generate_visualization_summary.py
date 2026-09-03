import pandas as pd
import glob


RootFolder = "C:\\Users\\Administrator\\Desktop\\"
Dataset = "GynSurg"
Splits = [
        "test",
        "val",
        "train",
        ]
Models = [
    #"SAM2",
    "SAMMed2D",
    "MedSAM",
    #"SurgiSAM",
    ]

OutputSheets = []

for Model in Models:
    files = {}
    for split in Splits:
        ClassIds = []
        splitSheets = []
        prefix = RootFolder+Model+"\\visualize_metrics\\"+Dataset+"_"+split+"_"
        xlsxFiles = glob.glob(prefix+"*_summary_statistics.xlsx")
        if(len(xlsxFiles)) == 0:
            continue
        for file in xlsxFiles:
            classId = file[len(prefix):-len("_summary_statistics.xlsx")]
            if classId == "":
                #classId = "all"
                pass
            else:
                classId = f"{int(classId):03d}"
            ClassIds.append(classId)
            files[classId] = file
        ClassIds.sort()
        for classId in ClassIds:
            sheet = pd.read_excel(files[classId])
            if classId == "":
                classId = "all"
            subSheet = sheet[sheet['parameter'].isin(["dice","iou","precision","recall","specificity","inference_time_ms"])].loc[:,("parameter","mean","prompt_mode")]
            subSheet["parameter"] = classId + " " +  subSheet["parameter"]
            subSheet= subSheet.set_index([subSheet.columns[2],subSheet.columns[0]]).sort_index()
            subSheet= subSheet.rename({"mean":Model + " " + split},axis=1)
            splitSheets.append(subSheet)
        concatSheet = pd.concat(splitSheets).T
        concatSheet = concatSheet.loc[:,~concatSheet.columns.duplicated()].copy()
        OutputSheets.append(concatSheet)

pd.concat(OutputSheets).to_excel(RootFolder+'output.xlsx', sheet_name='Raw')