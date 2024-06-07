import os
import random
import glob
import pandas as pd

from sklearn.model_selection import train_test_split

def get_path(folder):
    path = f"D:\\ACPC\\DATA\\"

    rawctniipath, ctniipath, mrniipath, gmniipath, wmniipath, csfniipath = [], [], [], [], [], []

    for f in folder:
        frctpath = f"{path}{f}\\RCT\\" 
        for file in glob.glob(frctpath + "/*.nii.gz"):
            rawctniipath.append(file)

        fctpath = f"{path}{f}\\CT\\"
        for file in glob.glob(fctpath + "/*.nii.gz"):
            ctniipath.append(file)

        fmrpath = f"{path}{f}\\MRI\\"
        for file in glob.glob(fmrpath + "/*.nii.gz"):
            mrniipath.append(file)

        fgmpath = f"{path}{f}\\GM\\"
        for file in glob.glob(fgmpath + "/*.nii.gz"):
            gmniipath.append(file)   

        fwmpath = f"{path}{f}\\WM\\"
        for file in glob.glob(fwmpath + "/*.nii.gz"):
            wmniipath.append(file)      

        fcsfpath = f"{path}{f}\\CSF\\"
        for file in glob.glob(fcsfpath + "/*.nii.gz"):
            csfniipath.append(file)

    return rawctniipath, ctniipath, mrniipath, gmniipath, wmniipath, csfniipath

def set_train_test(args):
    try:
        traininfo = pd.read_excel(f"D:\\ACPC\\traininfo_{args.train_patient}.xlsx")
        testinfo = pd.read_excel(f"D:\\ACPC\\testinfo_{args.test_patient}.xlsx")
    except:
        path = f"D:\\ACPC\\DATANII\\"

        patient=[]

        for folder in os.listdir(path):
            if folder == '5747297' or folder =='9098907' or folder=='5548173':
                continue
            patient.append(folder)

        random.shuffle(patient)

        train_patient = patient[:args.train_patient]
        test_patient = patient[-args.test_patient:]

        traininfo = pd.DataFrame({'train_index' : train_patient})
        testinfo = pd.DataFrame({'test_index' : test_patient})

        traininfo.to_excel(f"D:\\ACPC\\traininfo_{args.train_patient}.xlsx", index=False)
        testinfo.to_excel(f"D:\\ACPC\\testinfo_{args.test_patient}.xlsx", index=False)

        traininfo = pd.read_excel(f"D:\\ACPC\\traininfo_{args.train_patient}.xlsx")
        testinfo = pd.read_excel(f"D:\\ACPC\\testinfo_{args.test_patient}.xlsx")

    train_patient = list(traininfo['train_index'])
    test_patient = list(testinfo['test_index'])

    print(f"Train Patient : {len(train_patient)}\nTest patient : {len(test_patient)}")

    train_rct, train_ct, train_mr, train_gm, train_wm, train_csf = get_path(train_patient)
    test_rct, test_ct, test_mr, test_gm, test_wm, test_csf = get_path(test_patient)

    print(f"\nTrain Slice : {len(train_ct)}\nTest Slice : {len(test_ct)}")

    return train_rct, train_ct, train_mr, train_gm, train_wm, train_csf, test_rct, test_ct, test_mr, test_gm, test_wm, test_csf