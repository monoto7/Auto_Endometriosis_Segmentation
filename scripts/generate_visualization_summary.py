import pandas as pd
import glob
import numpy as np


RootFolder = "/data/m_meyer/AES/OrganSegmentationResults/GynSurg255/"
Dataset = "GynSurg255"
Splits = [
        "test",
        "val",
        #"train",
        ]
Models = [
     "SegFormer",
     "DeepLabV3Plus",
     "UNetPP",
     "YOLO11s_seg",
     "nnUNetV2_2D_100epochs",
     "nnUNetV2_2D"
    # "SAM2",
    # "SurgiSam",
    # "SamMed2D",
    # "MedSAM",
    ]
MetricsToWeigh = ["dice","iou","precision","recall","specificity"]
Metrics = ["dice","iou","precision","recall","specificity","inference_time_ms"]
ClassNames = {"085":"Uterus","170":"Fallopian Tubes","255":"Ovaries"}
OutputSheets = []
Trained = True
FileSuffix = "_Trained"
for Model in Models:
    files = {}
    for split in Splits:
        ClassIds = []
        splitSheets = []
        prefix = RootFolder+Model+("/trained/" if Trained else "")+"/visualize_metrics/"+Dataset+"_"+split+"_"
        xlsxFiles = glob.glob(prefix+"*summary_statistics.xlsx")
        if(len(xlsxFiles)) == 0:
            print("no xlsx files found for " + prefix)
            continue
        for file in xlsxFiles:
            classId = file[len(prefix):-len("_summary_statistics.xlsx")]
            if classId == "":
                #classId = "all"
                pass
            else:
                #Making sure it has the correct value and is sorted properly
                classId = f"{int(classId):03d}"
            ClassIds.append(classId)
            files[classId] = file
        ClassIds.sort()
        #print(ClassIds)
        unweighedMeans = {}
        totalPerPrompt = {}
        for classId in ClassIds:
            sheet = pd.read_excel(files[classId])
            MetricsToRead = Metrics.copy()
            if classId == "":
                classId = "mean"
                if len(ClassIds) > 1:
                    MetricsToRead = ["inference_time_ms"]
                    #MetricsToRead = MetricsToWeigh  
            else:
                MetricsToRead = MetricsToWeigh                
            className = ClassNames.get(classId, classId)

            subSheet = sheet[sheet['parameter'].isin(MetricsToRead)].loc[:,("parameter","mean","prompt_mode")]
            subSheet["parameter"] = className + " " +  subSheet["parameter"]
            promptModes = subSheet["prompt_mode"].unique()
            subSheet= subSheet.set_index([subSheet.columns[2],subSheet.columns[0]]).sort_index()
            subSheet= subSheet.rename({"mean":Model + " " + split},axis=1)
            splitSheets.append(subSheet)
            if(classId != "mean"):
                for promptMode in promptModes:
                    unweighedMeans[promptMode] = unweighedMeans.get(promptMode,{})
                    unweighedMeans[promptMode][className] = {"count":sheet[sheet['parameter'].isin(["recall"])].loc[:,("n")].to_numpy()[0]}
                    for metricToWeigh in MetricsToWeigh:
                        subsheet = sheet[sheet['prompt_mode'].isin([promptMode])]
                        unweighedMeans[promptMode][className][metricToWeigh] = subsheet[subsheet['parameter'].isin([metricToWeigh])].loc[:,("mean")].to_numpy()[0]
                    originalValue = totalPerPrompt.get(promptMode,0)
                    totalPerPrompt[promptMode] = originalValue + unweighedMeans[promptMode][className]["count"]

        if(len(unweighedMeans) != 0 ):

            weigheddfs = []
            for promptName in totalPerPrompt:
                for metricToWeigh in MetricsToWeigh:

                    MetricTimesCount=0

                    for classname in unweighedMeans[promptName]:
                        MetricTimesCount+= unweighedMeans[promptName][classname]["count"] * unweighedMeans[promptName][classname][metricToWeigh]

                    
                    WeightedMetric = MetricTimesCount/totalPerPrompt[promptName]

                    ##TODO Replace 'No Prompt' with actual column name.
                    index_tuples = [
                        (promptName, f"weighted {metricToWeigh}"),
                    ]
                    data_index = pd.MultiIndex.from_tuples(
                        index_tuples, names=["prompt_mode", "parameter"]
                    )
                    weigheddfs.append(pd.DataFrame([WeightedMetric], index=data_index, columns=[f"{Model} {split}"]))


            splitSheets.insert(0,pd.concat(weigheddfs))
        
        concatSheet = pd.concat(splitSheets).T
        concatSheet = concatSheet.loc[:,~concatSheet.columns.duplicated()].copy()
        OutputSheets.append(concatSheet)
pd.concat(OutputSheets).to_excel(RootFolder+'SummaryOfSummaries'+FileSuffix+'.xlsx', sheet_name='Raw')