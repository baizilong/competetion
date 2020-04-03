import numpy as np
# js=0
# # ret=[]
# np.memmap('/Users/zilongbai/scripts/scriptbackup/tianwen/train.mymemmap', dtype='float32', mode='w+', shape=(193417,2600),offset=0)
# # dic={}
# sheetindex=0
with open('/Users/zilongbai/scripts/scriptbackup/tianwen/new_columns_check_sets.csv') as f:
    sheet=[]
    while True:
        a=f.readline().strip('\n')
        temp=a.split(',')
        if len(temp[-1])>20:
        #if a!='' and temp[-2] in ['qso','star','galaxy']:
            #sheetindex=sheetindex+1
            input=[float(i) for i in temp[:-1]]
            # if temp[-2]=='qso':
            #     input.append(2.)
            # if temp[-2]=='star':
            #     input.append(0.)
            # if temp[-2]=='galaxy':
            #     input.append(1.)
            sheet.append(input)
            if len(sheet)==10000:
                w=np.memmap('/Users/zilongbai/test.mymemmap', dtype='float32', mode='r+', shape=(10000, 2600), offset=sheetindex*10000*2600*4)
                for i in range(len(sheet)):
                    w[i]=sheet[i]
                sheetindex=sheetindex+1
                sheet=[]
                print(sheetindex)
        if a=='':
            w = np.memmap('/Users/zilongbai/test.mymemmap', dtype='float32', mode='r+', shape=(10000, 2600),
                          offset=sheetindex * 10000 * 2600 * 4)
            for i in range(len(sheet)):
                w[i] = sheet[i]
            break
print(sheetindex)

# w = np.memmap('/Users/zilongbai/test.mymemmap', dtype='float32', mode='r+', shape=(10000, 2601),
#                           offset=57 * 10000 * 2601 * 4)
# for i in range(len(w)):
#     print(w[i])
# # b[0] =[2.2 for i in range(1000)]
