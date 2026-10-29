import pandas as pd
import glob


RootFolder = "/data/m_meyer/AES/OrganSegmentationResults/GynSurg/"
Dataset = "GynSurg"
Splits = [
        "test",
        "val",
        #"train",
        ]
Models = [
    "SegFormer",
    #"DeepLabV3Plus",
    #"UNetPP",
    #"YOLO11s_seg",
    #"nnUNetV2_2D_100epochs",
    #"nnUNetV2_2D"
    ]
MetricsToWeigh = ["dice","iou","precision","recall","specificity"]
Metrics = ["dice","iou","precision","recall","specificity","inference_time_ms"]

ClassNames = {"085":"Uterus","170":"Fallopian Tubes","255":"Ovaries"}
OutputSheets = []

for Model in Models:
    files = {}
    for split in Splits:
        ClassIds = []
        splitSheets = []
        prefix = RootFolder+Model+"/trained/visualize_metrics/"+Dataset+"_"+split+"_"
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
                classId = f"{int(classId):03d}"
            ClassIds.append(classId)
            files[classId] = file
        ClassIds.sort()
        #print(ClassIds)
        unweighedMeans = {}
        total = 0
        for classId in ClassIds:
            sheet = pd.read_excel(files[classId])
            MetricsToRead = Metrics.copy()
            if classId == "":
                classId = "mean"
                if len(ClassIds) > 1:
                    MetricsToRead = ["inference_time_ms"]
            else:
                MetricsToRead = MetricsToWeigh                
            className = ClassNames.get(classId, classId)

            subSheet = sheet[sheet['parameter'].isin(MetricsToRead)].loc[:,("parameter","mean","prompt_mode")]
            subSheet["parameter"] = className + " " +  subSheet["parameter"]
            
            subSheet= subSheet.set_index([subSheet.columns[2],subSheet.columns[0]]).sort_index()
            subSheet= subSheet.rename({"mean":Model + " " + split},axis=1)
            splitSheets.append(subSheet)

            if(classId != "mean"):
                unweighedMeans[className] = {"count":sheet[sheet['parameter'].isin(["recall"])].loc[:,("n")].to_numpy()[0]}
                for metricToWeigh in MetricsToWeigh:
                    unweighedMeans[className][metricToWeigh] = sheet[sheet['parameter'].isin([metricToWeigh])].loc[:,("mean")].to_numpy()[0]
                total += unweighedMeans[className]["count"]

        if(len(unweighedMeans) != 0 ):

            weigheddfs = []

            for metricToWeigh in MetricsToWeigh:

                MetricTimesCount=0

                for classname in unweighedMeans:
                    MetricTimesCount+= unweighedMeans[classname]["count"] * unweighedMeans[classname][metricToWeigh]

                
                WeightedMetric = MetricTimesCount/total

                ##TODO Replace 'No Prompt' with actual column name.
                index_tuples = [
                    ("No_prompt", f"weighted {metricToWeigh}"),
                ]
                data_index = pd.MultiIndex.from_tuples(
                    index_tuples, names=["prompt_mode", "parameter"]
                )
                weigheddfs.append(pd.DataFrame([WeightedMetric], index=data_index, columns=[f"{Model} {split}"]))


            splitSheets.insert(0,pd.concat(weigheddfs))
        
        concatSheet = pd.concat(splitSheets).T
        concatSheet = concatSheet.loc[:,~concatSheet.columns.duplicated()].copy()
        OutputSheets.append(concatSheet)
pd.concat(OutputSheets).to_excel(RootFolder+'SummaryOfSummaries.xlsx', sheet_name='Raw')